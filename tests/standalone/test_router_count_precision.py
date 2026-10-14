"""Native count-buffer regression on CPU; no GPU, Ray, credentials or model run."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e010_learning_observation import MEGATRON, actual_functions, observer


def _native_class_methods(path, class_name, methods, bases=(), extra=None):
    tree = ast.parse(path.read_text())
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected} == set(methods)
    for node in selected:
        node.decorator_list = []
    cls = ast.ClassDef(name=class_name, bases=list(bases), keywords=[], body=selected, decorator_list=[])
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch, **(extra or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[class_name]


def _native_router(monkeypatch, dtype):
    # Only replace CUDA device selection; buffer construction/counting are pinned native source.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    native = _native_class_methods(MEGATRON / "megatron/core/transformer/moe/router.py", "TopKRouter",
                                  {"_init_routing_mode", "_maintain_float32_expert_bias", "_apply_expert_bias"})
    router = torch.nn.Module()
    router.config = SimpleNamespace(dsv4_n_hash_layers=0, moe_router_enable_expert_bias=True,
                                   num_moe_experts=256, freeze_e_score_correction_bias=False)
    router.num_experts, router.bias, router._routing_mode_initialized = 256, None, False
    router.weight = torch.nn.Parameter(torch.ones(256, 2), requires_grad=False)
    native._init_routing_mode(router, layer_number=4)
    router.expert_bias.copy_(torch.linspace(-.25, .25, 256))
    if dtype != torch.float32:
        class Base(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
        wrapper = _native_class_methods(
            MEGATRON / "megatron/core/transformer/module.py", "Float16Module", {"__init__"},
            bases=[ast.Name(id="Base", ctx=ast.Load())], extra={"Base": Base},
        )
        wrapper(SimpleNamespace(fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
                                virtual_pipeline_model_parallel_size=None), router)
    native._maintain_float32_expert_bias(router)
    assert router.local_tokens_per_expert.dtype == dtype
    assert router.expert_bias.dtype == torch.float32
    return router, native._apply_expert_bias


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_prepare_restores_only_native_empty_nonpersistent_counter(monkeypatch, dtype):
    router, _ = _native_router(monkeypatch, dtype)
    original_counter = router.local_tokens_per_expert
    optimizer = SimpleNamespace(param_groups=[{"params": [torch.nn.Parameter(torch.ones(1))]}])
    snapshot = observer.require_frozen_router(router, optimizer)
    state_keys = set(router.state_dict())
    before = observer.prepare_router_count_buffer(router)
    assert before == str(dtype)
    assert router.local_tokens_per_expert.dtype == torch.float32
    assert router.local_tokens_per_expert.device == original_counter.device
    assert dict(router.named_buffers())["local_tokens_per_expert"] is router.local_tokens_per_expert
    assert "local_tokens_per_expert" not in router.state_dict()
    assert set(router.state_dict()) == state_keys
    assert observer.check_unchanged(snapshot)
    assert not router.weight.requires_grad
    if dtype == torch.float32:
        assert router.local_tokens_per_expert is original_counter


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_actual_accumulation_reproduces_rounding_and_fix_is_exact_across_updates(monkeypatch, dtype):
    broken, apply_count = _native_router(monkeypatch, dtype)
    fixed, _ = _native_router(monkeypatch, dtype)
    observer.prepare_router_count_buffer(fixed)
    expected = torch.zeros(256, dtype=torch.int64)
    # Odd, large microbatch counts expose both BF16 and FP16 integer rounding.
    assignments = torch.arange(256) % 101 + 4001
    routing = torch.arange(int(assignments.max())).unsqueeze(1) < assignments.unsqueeze(0)
    for update in range(2):
        for _ in range(3):
            apply_count(broken, routing)
            apply_count(fixed, routing)
            expected += assignments
        assert not torch.equal(broken.local_tokens_per_expert.to(torch.int64), expected)
        assert torch.equal(fixed.local_tokens_per_expert.to(torch.int64), expected)
        # Casting at the observer's old capture point cannot repair the lost integers.
        assert not torch.equal(broken.local_tokens_per_expert.float().to(torch.int64), expected)
        broken.local_tokens_per_expert.zero_()
        fixed.local_tokens_per_expert.zero_()
        expected.zero_()
        assert fixed.local_tokens_per_expert.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nonempty_buffer_cannot_be_repaired_after_accumulation(monkeypatch, dtype):
    router, _ = _native_router(monkeypatch, dtype)
    router.local_tokens_per_expert[0] = 1
    before = router.local_tokens_per_expert.clone()
    with pytest.raises(ValueError, match="before accumulation"):
        observer.prepare_router_count_buffer(router)
    assert router.local_tokens_per_expert.dtype == dtype
    assert torch.equal(router.local_tokens_per_expert, before)


@pytest.mark.parametrize("invalid", ["unregistered", "shape", "requires_grad", "dtype"])
def test_incompatible_buffer_rejected(monkeypatch, invalid):
    router, _ = _native_router(monkeypatch, torch.bfloat16)
    if invalid == "unregistered":
        del router.local_tokens_per_expert
        router.local_tokens_per_expert = torch.zeros(256)
    elif invalid == "shape":
        router.local_tokens_per_expert = torch.zeros(255)
    elif invalid == "requires_grad":
        router.local_tokens_per_expert.requires_grad_(True)
    else:
        router.local_tokens_per_expert = torch.zeros(256, dtype=torch.int64)
    with pytest.raises(ValueError):
        observer.prepare_router_count_buffer(router)


def test_repaired_native_stacked_reduction_keeps_existing_one_collective_and_bias(monkeypatch):
    routers = [_native_router(monkeypatch, torch.bfloat16)[0] for _ in range(2)]
    for router in routers:
        observer.prepare_router_count_buffer(router)
        router.local_tokens_per_expert.copy_(torch.arange(256) + 4001)
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, group: (calls.append(group), value.mul_(2)))
    native = actual_functions(MEGATRON / "megatron/core/transformer/moe/moe_utils.py", ["get_updated_expert_bias"],
                              {"parallel_state": SimpleNamespace(get_tensor_and_data_parallel_group=lambda **kw: "native")})
    obj = observer.RouterLoadObserver.__new__(observer.RouterLoadObserver)
    obj.original, obj.representative = native["get_updated_expert_bias"], True
    counts = torch.stack([router.local_tokens_per_expert for router in routers])
    bias = torch.stack([router.expert_bias for router in routers])
    before = bias.clone()
    result = obj._capture(counts, bias, 0.)
    assert calls == ["native"]
    assert obj.pending_dtype == "torch.float32"
    assert torch.equal(obj.pending, 2 * (torch.arange(256) + 4001).expand(2, -1))
    assert torch.equal(result, before) and torch.equal(bias, before)


def test_preparation_is_once_at_opt_in_initialization_not_each_update():
    tree = ast.parse(Path(observer.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RouterLoadObserver")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assert ast.unparse(init).count("prepare_router_count_buffer") == 1
    assert ast.unparse(cls).count("prepare_router_count_buffer") == 1
    assert observer.get_router_load_observer(SimpleNamespace(observe_router_load=False), None, None, None) is None


def test_actual_observer_initialization_prepares_native_buffer_once(monkeypatch):
    router, _ = _native_router(monkeypatch, torch.bfloat16)
    block = torch.nn.Module()
    block.router, block.local_expert_indices, block.ep_group = router, list(range(16)), object()
    mpu = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    state = SimpleNamespace(tp=SimpleNamespace(rank=0), effective_dp_cp=SimpleNamespace(rank=0),
                            cp=SimpleNamespace(size=4))
    monkeypatch.setitem(sys.modules, "miles.backends.training_utils.parallel",
                        SimpleNamespace(get_parallel_state=lambda: state))
    finalize = SimpleNamespace(get_updated_expert_bias=lambda *args: args[1])
    monkeypatch.setitem(sys.modules, "megatron.core.distributed.finalize_model_grads", finalize)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 16)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    args = SimpleNamespace(observe_router_load=True, recompute_method="uniform", recompute_num_layers=1,
                           qkv_format="thd", allgather_cp=True, sequence_parallel=True)
    config = SimpleNamespace(moe_router_freeze_gate=True, moe_router_enable_expert_bias=True,
                             moe_router_bias_update_rate=0, moe_aux_loss_coeff=0,
                             freeze_e_score_correction_bias=False)
    optimizer = SimpleNamespace(param_groups=[{"params": [torch.nn.Parameter(torch.ones(1))]}])
    obj = observer.get_router_load_observer(args, config, [block], optimizer)
    assert obj.counter_dtypes == ("torch.bfloat16",)
    assert router.local_tokens_per_expert.dtype == torch.float32
    assert observer.check_unchanged(obj.snapshot)
    # Reuse the already constructed observer, even after counting has started.
    router.local_tokens_per_expert[0] = 4001
    assert observer.get_router_load_observer(args, config, [block], optimizer) is obj
    assert router.local_tokens_per_expert[0].item() == 4001
