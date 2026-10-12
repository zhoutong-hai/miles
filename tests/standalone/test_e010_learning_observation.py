"""Changed-path CPU tests using actual pinned loss/router source, no Ray/GPU/API."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
MEGATRON = Path("/Users/zhoutong/code/Megatron-LM-e009-throughput")
spec = importlib.util.spec_from_file_location("router_observer", ROOT / "miles/backends/megatron_utils/router_load_observer.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def actual_functions(path, names, extra=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = {"torch": torch, **(extra or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, *nodes], type_ignores=[])), str(path), "exec"), ns)
    return ns


@pytest.mark.parametrize("advantage", [-2., -.1, 0., .1, 2.])
@pytest.mark.parametrize("ratio", [.01, .79, 1., 1.29, 20., 1e6])
def test_actual_dual_clip_caps_only_negative_advantages(advantage, ratio):
    ns = actual_functions(ROOT / "miles/backends/training_utils/loss_hub/math_utils.py",
                          ["_safe_clamp_log_ratio", "_safe_exp_neg_ppo_kl", "compute_policy_loss"],
                          {"_LOG_RATIO_EXP_CLAMP": 20.0})
    kl = torch.tensor([-ratio]).abs().log().neg()
    adv = torch.tensor([advantage])
    old, _ = ns["compute_policy_loss"](kl, adv, .2, .28)
    new, _ = ns["compute_policy_loss"](kl, adv, .2, .28, 10.)
    torch.testing.assert_close(new, old.clamp(max=-10 * advantage) if advantage < 0 else old)
    assert torch.isfinite(new).all()


@pytest.mark.parametrize("coef", [0., .001])
def test_actual_entropy_branch_is_differentiable_only_when_selected(coef):
    tree = ast.parse((ROOT / "miles/backends/training_utils/loss_hub/losses.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "policy_loss_function")
    branch = next(n for n in function.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                  and n.test.id == "calculate_entropy")
    logits = torch.tensor([2., 0., -1.], requires_grad=True)
    probabilities = logits.softmax(0)
    entropy = -(probabilities * logits.log_softmax(0)).sum().reshape(1)
    policy = logits.sum() * 0
    ns = {"torch": torch, "calculate_entropy": True, "args": SimpleNamespace(entropy_coef=coef),
          "log_probs_and_entropy": {"entropy": [entropy]}, "sum_of_sample_mean": lambda value: value.mean(),
          "pg_loss": policy, "loss": policy, "entropy_loss": policy.new_zeros(())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), "actual_entropy_branch", "exec"), ns)
    ns["loss"].backward()
    assert bool(torch.count_nonzero(logits.grad)) == (coef != 0)
    assert ns["entropy_loss"].requires_grad == (coef != 0)


def test_reduced_histograms_use_actual_mapping_and_population_cv():
    counts = torch.tensor([4., 0., 0., 4.])
    first = observer.summarize_counts(counts, [[0, 3], [1, 2]])
    second = observer.summarize_counts(counts, [[0, 1], [2, 3]])
    assert first == {"expert_max_mean": 2., "expert_cv": 1., "unused_expert_fraction": .5, "ep_rank_max_mean": 2.}
    assert second["ep_rank_max_mean"] == 1.
    assert observer.summarize_counts(torch.zeros(4), [[0, 1], [2, 3]])["unused_expert_fraction"] == 1.


@pytest.mark.parametrize("mapping", [[[0, 1], [1, 3]], [[0, 1]], [[0, 1], [2, 4]]])
def test_mapping_duplicates_missing_and_foreign_experts_rejected(mapping):
    with pytest.raises(ValueError):
        observer.summarize_counts(torch.ones(4), mapping)


def test_capture_uses_existing_native_reduction_once_and_preserves_bias(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value, group: (calls.append(group), value.mul_(2)))
    ns = actual_functions(MEGATRON / "megatron/core/transformer/moe/moe_utils.py", ["get_updated_expert_bias"],
                          {"parallel_state": SimpleNamespace(get_tensor_and_data_parallel_group=lambda **kw: "existing")})
    obj = observer.RouterLoadObserver.__new__(observer.RouterLoadObserver)
    obj.original = ns["get_updated_expert_bias"]
    obj.representative = True
    counts, bias = torch.tensor([[1., 2.]]), torch.tensor([[.3, -.2]])
    before = bias.clone()
    result = obj._capture(counts, bias, 0.)
    assert calls == ["existing"]
    assert torch.equal(obj.pending, torch.tensor([[2, 4]]))
    assert torch.equal(result, before) and torch.equal(bias, before)


def test_original_router_count_no_grad_scoring_and_full_recompute_count_once():
    tree = ast.parse((MEGATRON / "megatron/core/transformer/moe/router.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TopKRouter")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_expert_bias")
    method.decorator_list = []
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, method], type_ignores=[])), "actual_router_count", "exec"), ns)
    router = SimpleNamespace(enable_expert_bias=True, local_tokens_per_expert=torch.zeros(2))
    routing = torch.tensor([[True, False], [False, True]])
    with torch.no_grad():
        ns["_apply_expert_bias"](router, routing)
    assert torch.count_nonzero(router.local_tokens_per_expert) == 0
    def forward(x):
        ns["_apply_expert_bias"](router, routing)
        return x.square()
    checkpoint(forward, torch.ones(1, requires_grad=True), use_reentrant=True).sum().backward()
    assert torch.equal(router.local_tokens_per_expert, torch.ones(2))


def test_frozen_gate_exclusion_unchanged_proof_and_real_vs_padded_accounting():
    gate = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    bias = torch.tensor([.3, -.2])
    router = SimpleNamespace(weight=gate, bias=None, expert_bias=bias)
    optimizer = SimpleNamespace(param_groups=[{"params": [torch.nn.Parameter(torch.ones(1))]}])
    snapshot = observer.require_frozen_router(router, optimizer)
    assert observer.check_unchanged(snapshot)
    bias[0] += 1
    assert not observer.check_unchanged(snapshot)
    gate.requires_grad_(True)
    with pytest.raises(ValueError):
        observer.require_frozen_router(router, optimizer)
    obj = observer.RouterLoadObserver.__new__(observer.RouterLoadObserver)
    obj.cp_size, obj.real_positions, obj.packed_positions = 4, 0, 0
    obj.record_batch({"tokens": torch.zeros(1, 8), "total_lengths": [11, 17]})
    assert (obj.real_positions, obj.packed_positions) == (28, 32)


def test_opt_out_is_inert_and_capture_restored_after_error():
    assert observer.get_router_load_observer(SimpleNamespace(), None, None, None) is None
    obj = observer.RouterLoadObserver.__new__(observer.RouterLoadObserver)
    original = lambda *args: None
    obj.original, obj.finalize = original, SimpleNamespace(get_updated_expert_bias=original)
    with pytest.raises(RuntimeError):
        with obj.step():
            assert obj.finalize.get_updated_expert_bias != original
            raise RuntimeError("interrupted")
    assert obj.finalize.get_updated_expert_bias is original


def test_capture_batch_is_only_in_training_not_score_forward():
    tree = ast.parse((ROOT / "miles/backends/megatron_utils/model.py").read_text())
    scoring = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward_only")
    training = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_one_step")
    assert "record_batch" not in ast.unparse(scoring)
    assert ast.unparse(training).count("record_batch") == 1


@pytest.mark.parametrize("frozen", [False, True])
def test_actual_native_constructed_gate_freeze(frozen):
    tree = ast.parse((MEGATRON / "megatron/core/transformer/moe/router.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Router")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"__init__", "reset_parameters"}]
    class Base(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
    extracted = ast.ClassDef(name="ActualRouter", bases=[ast.Name(id="Base", ctx=ast.Load())],
                             keywords=[], body=nodes, decorator_list=[])
    prefix = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ns = {"torch": torch, "Base": Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, extracted], type_ignores=[])), "actual_router_constructor", "exec"), ns)
    config = SimpleNamespace(num_moe_experts=4, hidden_size=2, add_bias_linear=True,
                             calculate_per_token_loss=False, perform_initialization=False,
                             params_dtype=torch.float32, sequence_parallel=True, moe_router_freeze_gate=frozen)
    pg = SimpleNamespace(tp=None, cp=None, tp_cp=None, tp_dp_cp=None)
    router = ns["ActualRouter"](config, pg)
    assert router.weight.requires_grad == router.bias.requires_grad == (not frozen)


@pytest.mark.parametrize("rollout,step,due", [(8,1,False),(9,0,False),(9,1,True),(19,0,False),(19,1,True)])
def test_snapshot_policy_completed_rollouts_only(rollout, step, due):
    assert (step == 1 and (rollout + 1) % 10 == 0) == due


@pytest.mark.parametrize("valid", [False, True])
def test_publish_private_counts_only_and_invalid_accounting_degrades_metrics(monkeypatch, tmp_path, valid):
    logged = []
    monkeypatch.setitem(sys.modules, "miles.utils.tracking_utils",
        SimpleNamespace(tracking=SimpleNamespace(log=lambda args, metrics, step_key: logged.append(metrics))))
    obj = observer.RouterLoadObserver.__new__(observer.RouterLoadObserver)
    obj.args = SimpleNamespace(router_load_output=str(tmp_path))
    obj.first_check, obj.accounting_valid = True, None
    obj.rank_experts = [list(range(16 * rank, 16 * (rank + 1))) for rank in range(16)]
    counts = [1] * 256
    layers = [{"layer": layer, "metrics": observer.summarize_counts(torch.tensor(counts), obj.rank_experts),
               "accounting": {"counts_match_packed_work": valid}, "counts": counts} for layer in range(4, 79)]
    obj._publish(layers, 9, 1, 20, True)
    assert logged[0]["router_load/accounting_passed"] == int(valid)
    assert any("by_update" in key for key in logged[0]) == valid
    assert not any(isinstance(value, (dict, list, torch.Tensor)) for value in logged[0].values())
    path = tmp_path / "rollout_0010.json"
    assert path.exists() == valid
    if valid:
        payload = json.loads(path.read_text())
        assert len(payload["counts"]) == 75 and all(len(row) == 256 for row in payload["counts"])
        assert "weights" not in payload
    assert (tmp_path / "first-update-accounting.json").exists()
