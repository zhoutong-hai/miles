"""K3 KDA Q/K/V LoRA through the actual TP/SP layer and recurrence, against merged local weights.

Run with ``torchrun --nproc_per_node=2 tests/fast-gpu/test_lora_native_kimi_k3.py``.
The input collective belongs to LinearAttentionLayer: an adapter-local TP copy
would double-reduce dx, which this full-layer gradient comparison detects.
"""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci
from tests.fast.backends.megatron_utils.lora.native.naming_stub import FixedPrefixNaming

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.spec.attention import KimiK3AttentionSpec
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.backends.megatron_utils.lora.utils import reduce_marked_lora_grads
from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LinearAttnHeads

sys.path.insert(0, os.path.dirname(__file__))
from linear_attn_reference import ReplicatedKDA, build_layer, packed, rel_err  # noqa: E402

register_cuda_ci(
    est_time=180, suite="stage-b-2-gpu-h200", labels=["lora"], hardware=["hopper", "blackwell"], num_gpus=2
)

PREFIX = "language_model.model.layers.0.self_attn."
LIVE_LAYERS = []  # reduce_marked_lora_grads caches by model identity


def check(sequence_parallel, world):
    hidden, rank, alpha = 256, 128, 256
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
        tensor_model_parallel_size=world,
        sequence_parallel=sequence_parallel,
    )
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    reference = ReplicatedKDA(hidden, LinearAttnHeads(4, 4, 64, 64), torch.float32).cuda()
    model, merged = build_layer(reference, config), build_layer(reference, config)
    LIVE_LAYERS.extend([model, merged])
    for m in (model, merged):
        for parameter in m.parameters():
            parameter.requires_grad_(False)
    tp = parallel_state.get_tensor_model_parallel_group()
    context = AttachContext(
        lora=LoRAConfig(
            rank=rank, alpha=alpha, dropout=0.0, target_modules=tuple(PREFIX + f"{v}_proj" for v in ("q", "k", "v"))
        ),
        transformer_config=config,
        tp_size=world,
        tp_rank=tp.rank(),
        naming=FixedPrefixNaming(PREFIX),
    )
    block = SimpleNamespace(is_kda=True, linear_attn=model.linear_attn)
    assert KimiK3AttentionSpec().attach(block, None, context) == 3
    generator = torch.Generator(device="cuda").manual_seed(123)
    full_x = torch.randn(128, 1, hidden, generator=generator, device="cuda") * 0.1
    full_dy = torch.randn(full_x.shape, generator=generator, device="cuda")
    x = (full_x.chunk(world)[tp.rank()] if sequence_parallel else full_x).detach().clone().requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    dy = full_dy.chunk(world)[tp.rank()] if sequence_parallel else full_dy
    params = packed(torch.tensor([0, 128], dtype=torch.int32, device="cuda"))
    with torch.no_grad():
        actual_zero = model(x, packed_seq_params=params)[0]
        expected_zero = merged(expected_x, packed_seq_params=params)[0]
        torch.testing.assert_close(actual_zero, expected_zero, rtol=0, atol=0)
    adapters = []
    for leaf in ("q", "k", "v"):
        adapter = getattr(model.linear_attn, f"lora_{leaf}_adapter")
        a, b = getattr(adapter, f"{leaf}_A"), getattr(adapter, f"{leaf}_B")
        full_a = torch.randn(a.shape, generator=generator, device="cuda") * 0.02
        full_b = torch.randn((b.shape[0] * world, b.shape[1]), generator=generator, device="cuda") * 0.02
        with torch.no_grad():
            a.copy_(full_a)
            b.copy_(full_b.chunk(world)[tp.rank()])
            weight = getattr(merged.linear_attn, f"{leaf}_proj").weight
            weight.add_(b @ a, alpha=alpha / rank)
        weight.requires_grad_(True)
        adapters.append((leaf, a, b, weight))
    actual = model(x, packed_seq_params=params)[0]
    expected = merged(expected_x, packed_seq_params=params)[0]
    (actual * dy).sum().backward()
    (expected * dy).sum().backward()
    reduce_marked_lora_grads([model])
    errors = {"output": rel_err(expected, actual), "dx": rel_err(expected_x.grad, x.grad)}
    for leaf, a, b, weight in adapters:
        expected_a = (alpha / rank) * (b.detach().t() @ weight.grad)
        expected_b = (alpha / rank) * (weight.grad @ a.detach().t())
        dist.all_reduce(expected_a, group=tp)
        errors[f"{leaf}.A"] = rel_err(expected_a, a.grad)
        errors[f"{leaf}.B"] = rel_err(expected_b, b.grad)
    assert all(torch.isfinite(torch.tensor(v)) and v < 5e-3 for v in errors.values()), errors
    if dist.get_rank() == 0:
        print({"TP": world, "SP": sequence_parallel, "rank": rank, "alpha": alpha, "relative_errors": errors})


def main():
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(42)
    for sequence_parallel in (False, True):
        check(sequence_parallel, world)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
