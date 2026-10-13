"""Actual GLM forward call-site under packed CP4 partition/reference kernels.

CPU references validate glue, not CUDA kernels/collectives. GPU qualification
reuses these helpers with actual kernels and a four-rank process group.
"""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.kernels.attention.dsa import causal_ranges, get_dsa_topk_fn

ROOT = Path(__file__).resolve().parents[2]


def forward_method(namespace):
    path = ROOT / "miles_plugins/models/glm5/glm5.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DSAMultiLatentAttention")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["forward"]


def reference_attention(q, kv, indices, sm_scale, d_v):
    selected = kv[0, indices[0, :, 0].clamp(min=0).long(), 0]
    scores = torch.einsum("shd,skd->shk", q[0].float(), selected.float()) * sm_scale
    scores = scores.masked_fill(indices[0, :, 0, None] < 0, -torch.inf)
    return torch.einsum("shk,skd->shd", scores.softmax(-1), selected[..., :d_v].float()).to(q.dtype)[None]


def reference_indexer(q, k, weights, starts, ends, topk, topk_fn):
    assert weights.ndim == 2 and weights.dtype == torch.float32
    logits = (torch.einsum("shd,kd->shk", q.float(), k.float()).relu() * weights[..., None]).sum(1)
    positions = torch.arange(k.shape[0], device=k.device)[None]
    logits = logits.masked_fill((positions < starts[:, None]) | (positions >= ends[:, None]), -torch.inf)
    indices = topk_fn(logits, topk)
    return logits.gather(1, indices.clamp(min=0).long()).masked_fill(indices < 0, -torch.inf), indices


def callsite_case(rank, share, *, device="cpu", gpu_kernels=False, scatter=None):
    torch.manual_seed(98)
    total, heads, dim, d_v, index_heads, index_dim, topk = (2048, 16, 576, 512, 64, 128, 2048) if gpu_kernels else (24, 2, 6, 4, 2, 4, 4)
    dtype = torch.bfloat16 if gpu_kernels else torch.float32
    cu = [0, 333, 981, 2047, 2048] if gpu_kernels else [0, 3, 9, 16, 23, 24]
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(cu, device=device, dtype=torch.int32))
    q = torch.randn(total // 4, heads, dim, device=device, dtype=dtype, requires_grad=True)
    kv = torch.randn(total, 1, dim, device=device, dtype=dtype, requires_grad=True)
    wv = torch.randn(heads, 3, d_v, device=device, dtype=dtype)
    iq = torch.randn(total // 4, index_heads, index_dim, device=device, dtype=dtype)
    ik = torch.randn(total, 1, index_dim, device=device, dtype=dtype)
    weights = torch.rand(total // 4, index_heads, device=device)
    replay_calls, selections = [], []

    def replay(fn, return_probs):
        replay_calls.append(return_probs)
        return fn

    def indexer(*args, **kwargs):
        function = importlib.import_module("miles.kernels.attention.dsa").lighting_indexer if gpu_kernels else reference_indexer
        scores, indices = function(*args, **kwargs)
        selections.append(indices.detach())
        return scores, indices

    attention = importlib.import_module("miles.kernels.attention.dsa").sparse_attention if gpu_kernels else reference_attention
    namespace = dict(torch=torch, causal_ranges=causal_ranges, get_dsa_topk_fn=get_dsa_topk_fn,
                     lighting_indexer=indexer, sparse_attention=attention,
                     indexer_replay_manager=SimpleNamespace(enabled=False, get_topk_fn=replay),
                     parallel_state=SimpleNamespace(get_context_parallel_group=lambda: None),
                     scatter_to_sequence_parallel_region=scatter or (lambda value, **kw: value.chunk(4)[rank]))
    forward = forward_method(namespace)
    model = SimpleNamespace(config=SimpleNamespace(kv_lora_rank=d_v),
                            get_absorb_query_key_value_tensors=lambda *a, **kw: (q, kv, wv, iq, ik, weights),
                            linear_proj=lambda x: (x, None), index_topk=topk, topk_backend="torch",
                            index_share=share, skip_topk=False, layer_number=2, _source_layer=2,
                            index_topk_freq=2, skip_topk_offset=0,
                            _HOLDER_ATTR="_dsa_index_share_topk_holder", recompute_up_proj=False, softmax_scale=dim**-.5)
    out, _ = forward(model, None, None, packed_seq_params=packed)
    starts, ends = causal_ranges(packed.cu_seqlens_q)
    chosen = selections[0]
    mask = chosen >= 0
    assert ((chosen >= starts.chunk(4)[rank][:, None]) | ~mask).all()
    assert ((chosen < ends.chunk(4)[rank][:, None]) | ~mask).all()
    assert (chosen == -1).any()  # short segments exercise padded top-k
    ref = reference_attention(q[None], kv[None], chosen[:, None][None], model.softmax_scale, d_v)[0]
    ref = torch.einsum("shm,hdm->shd", ref, wv).reshape(total // 4, 1, -1)
    torch.testing.assert_close(out, ref, atol=.12 if gpu_kernels else 1e-5, rtol=.02 if gpu_kernels else 1e-5)
    out.float().square().mean().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()
    if share:
        model.skip_topk = True
        model.layer_number = 3
        shared, _ = forward(model, None, None, packed_seq_params=packed)
        torch.testing.assert_close(shared, out)
        assert len(selections) == 1  # skip layer really reused indices
        isolated = SimpleNamespace(cu_seqlens_q=packed.cu_seqlens_q)
        with pytest.raises(AssertionError, match="did not run"):
            forward(model, None, None, packed_seq_params=isolated)
        model.skip_topk = False
        recomputed, _ = forward(model, None, None, packed_seq_params=packed)
        torch.testing.assert_close(recomputed, out)
    assert replay_calls and not any(replay_calls)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("share", [False, True])
def test_actual_glm_cp4_packing_padding_and_shared_indices(rank, share):
    callsite_case(rank, share)


def test_kernel_compiler_safety_and_single_active_glm_path():
    path = ROOT / "miles/kernels/attention/dsa/tilelang/sparse_attention_bwd.py"
    tree = ast.parse(path.read_text())
    settings = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
    assert any(isinstance(k, ast.Attribute) and k.attr == "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE" and isinstance(v, ast.Constant) and v.value is False for node in settings for k, v in zip(node.keys, node.values))
    model = (ROOT / "miles_plugins/models/glm5/glm5.py").read_text()
    assert "SparseMLA.apply" not in model and "from .ops.indexer" not in model
    assert "miles.utils.hf_config import load_hf_config" in model
