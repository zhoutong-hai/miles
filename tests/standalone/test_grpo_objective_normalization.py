"""CPU checks of actual Miles reducers and pinned Megatron schedule scaling.

No model, credentials, or GPU fixture. CP and DP are emulated at the reducer
boundary; numerical reference uses a single unpartitioned global batch.
"""

import argparse
import ast
from collections.abc import Callable
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
MEGATRON = Path(os.environ.get("MILES_TEST_MEGATRON_ROOT", ROOT.parent / "Megatron-LM-e009-throughput"))


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def boundary(cp_size, rank):
    state = SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=rank), intra_dp=SimpleNamespace(size=1), intra_dp_cp=SimpleNamespace(size=cp_size))
    ns = {"torch": torch, "Callable": Callable, "get_parallel_state": lambda: state, "Namespace": argparse.Namespace, "RolloutBatch": dict}
    functions(
        ROOT / "miles/backends/training_utils/cp_utils.py",
        {
            "get_sum_of_sample_mean",
            "_slice_loss_mask_for_local_cp",
            "get_logits_and_tokens_offset_with_cp",
        },
        ns,
    )
    functions(ROOT / "miles/backends/training_utils/loss.py", {"loss_function"}, ns)
    ns.update(Namespace=argparse.Namespace, RolloutBatch=dict, is_multi_lora_enabled=lambda args: False)
    return ns


@pytest.fixture
def native_schedule(monkeypatch):
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.multi_token_prediction", SimpleNamespace(MTPLossAutoScaler=SimpleNamespace()))
    return functions(MEGATRON / "megatron/core/pipeline_parallel/schedules.py", {"forward_step_calc_loss"}, {"torch": torch})["forward_step_calc_loss"]


@pytest.mark.parametrize("mode", ["sample", "token", "fixed"])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("partition", [[0, 1, 2, 3], [[0], [1, 2], [3]]])
def test_actual_global_loss_and_grad_are_partition_invariant(native_schedule, mode, cp_size, partition):
    lengths = [3, 7, 2, 0]
    masks = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]), torch.zeros(2), torch.zeros(0)]
    values = [torch.linspace(0.2, 1.2, n) for n in lengths]
    groups = [partition] if isinstance(partition[0], int) else partition
    theta = torch.tensor(0.7, requires_grad=True)
    combined = theta * 0
    token_count = 0
    log_rows = []
    for rank in range(cp_size):
        ns = boundary(cp_size, rank)
        for indices in groups:
            batch = {"total_lengths": [lengths[i] + 11 for i in indices], "response_lengths": [lengths[i] for i in indices], "loss_masks": [masks[i] for i in indices]}
            local_values = []
            for i in indices:
                v = values[i] if cp_size == 1 else ns["_slice_loss_mask_for_local_cp"](lengths[i] + 11, lengths[i], values[i], "thd", None)
                local_values.append(v)
            x = torch.cat(local_values) * theta
            args = SimpleNamespace(
                calculate_per_token_loss=mode == "token", qkv_format="thd", grpo_fixed_response_normalizer=512 if mode == "fixed" else None, grpo_std_normalization=mode != "fixed", recompute_loss_function=False, allgather_cp=False, global_batch_size=4, use_dynamic_global_batch_size=False, true_on_policy_mode=False
            )
            ns["get_loss_function"] = lambda args: lambda a, b, logits, reducer: (reducer(logits - 0.005 * logits.square()), {"loss": reducer(logits).detach()})
            loss, count, log = ns["loss_function"](args, batch, len(groups), x, apply_megatron_loss_scaling=True, num_rollouts=4)
            log_rows.append(log)
            loss, count = native_schedule(SimpleNamespace(), loss, lambda output, count=count: (output, count, {}), SimpleNamespace(calculate_per_token_loss=mode == "token", timers=None), None, False, len(groups), [], cp_group_size=cp_size, is_last_stage=True)
            combined = combined + loss
            token_count += count
    # Native token mode SUM-reduces gradients and the full response count is
    # replicated CP times. Sample mode averages dense grads over DP×CP.
    actual = combined / (token_count if mode == "token" else cp_size)
    numerator = [(v * theta - 0.005 * (v * theta).square()) * m for v, m in zip(values, masks, strict=True)]
    if mode == "token":
        expected = sum(v.sum() for v in numerator) / sum(m.sum() for m in masks)
    elif mode == "fixed":
        expected = sum(v.sum() for v in numerator) / (4 * 512)
    else:
        expected = sum(v.sum() / m.sum().clamp_min(1) for v, m in zip(numerator, masks, strict=True)) / 4
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, theta)[0], torch.autograd.grad(expected, theta)[0])
    if mode != "sample":
        ns["get_parallel_state"] = lambda: SimpleNamespace(cp=SimpleNamespace(size=cp_size), effective_dp_cp=SimpleNamespace(groups_inner_to_outer=[]))
        functions(ROOT / "miles/backends/training_utils/log_utils.py", {"aggregate_train_losses"}, ns)
        metrics = ns["aggregate_train_losses"](log_rows, None if mode == "token" else 4)
        assert metrics["objective_token_mean"] == int(mode == "token")
        assert metrics["objective_fixed_response_normalizer"] == (512 if mode == "fixed" else 0)
        assert metrics["objective_reward_std"] == int(mode == "token")


@pytest.mark.parametrize("fixed", [None, 512])
def test_actual_policy_tis_rebuild_keeps_differentiable_entropy_denominator(fixed):
    ns = boundary(1, 0)
    functions(ROOT / "miles/backends/training_utils/cp_utils.py", {"get_local_response_loss_masks"}, ns)
    functions(ROOT / "miles/backends/training_utils/loss_hub/losses.py", {"policy_loss_function"}, ns)
    theta = torch.tensor(0.6, requires_grad=True)
    logits = theta * torch.tensor([0.2, 0.4, 0.8, 1.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rejected = torch.tensor([1.0, 0.0, 1.0, 0.0])
    args = SimpleNamespace(
        calculate_per_token_loss=False,
        qkv_format="thd",
        grpo_fixed_response_normalizer=fixed,
        entropy_coef=0.005,
        observe_training_entropy=True,
        use_rollout_logprobs=False,
        skip_actor_forward_only=False,
        advantage_estimator="grpo",
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=10.0,
        get_mismatch_metrics=False,
        use_tis=True,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        use_kl_loss=False,
    )
    batch = dict(advantages=[torch.ones(4)], log_probs=[torch.zeros(4)], rollout_log_probs=[torch.zeros(4)], loss_masks=[mask], total_lengths=[15], response_lengths=[4], unconcat_tokens=[torch.ones(4)], rollout_mask_sums=[torch.tensor(4.0)])
    ns.update(
        get_log_probs_and_entropy=lambda logits, **kwargs: {"log_probs": [logits], "entropy": [logits.square()]},
        compute_policy_loss=lambda kl, adv, *args: (-kl * adv, torch.zeros_like(kl)),
        vanilla_tis_function=lambda **kwargs: (kwargs["pg_loss"], [rejected], {}),
        compute_ess_ratio_contribution=lambda **kwargs: torch.tensor(1.0),
        compute_approx_kl=lambda x, y, **kwargs: x - y,
    )
    reducer = ns["get_sum_of_sample_mean"]([15], [4], [mask], denominators=batch["rollout_mask_sums"], fixed_response_normalizer=fixed)
    loss, metrics = ns["policy_loss_function"](args, batch, logits, reducer)
    denom = fixed or 4
    expected = ((logits - 0.005 * logits.square()) * rejected).sum() / denom
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, theta, retain_graph=True)[0], torch.autograd.grad(expected, theta)[0])
    torch.testing.assert_close(metrics["entropy_loss"], (logits.square() * rejected).sum() / denom)
    assert theta.grad is None


@pytest.mark.parametrize("cp_size", [1, 4])
def test_fixed_budget_survives_tis_mask_rebuild_and_ignores_rollout_denominator(cp_size):
    masks = [torch.tensor([1.0, 0.0, 1.0, 0.0])]
    total = torch.tensor(0.0)
    for rank in range(cp_size):
        ns = boundary(cp_size, rank)
        x = torch.arange(1.0, 5.0)
        if cp_size > 1:
            x = ns["_slice_loss_mask_for_local_cp"](15, 4, x, "thd", None)
        reducer = ns["get_sum_of_sample_mean"]([15], [4], masks, denominators=[torch.tensor(99.0)], fixed_response_normalizer=512)
        total += reducer(x)
    torch.testing.assert_close(total, torch.tensor(4.0 / 512))
    tree = ast.parse((ROOT / "miles/backends/training_utils/loss_hub/losses.py").read_text())
    rebuild = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "get_sum_of_sample_mean"]
    assert any(any(k.arg == "fixed_response_normalizer" for k in n.keywords) for n in rebuild)


@pytest.mark.parametrize("budget,token", [(0, False), (-1, False), (512, True)])
def test_incompatible_reducer_fails_closed(budget, token):
    with pytest.raises(ValueError):
        boundary(1, 0)["get_sum_of_sample_mean"]([5], [1], [torch.ones(1)], calculate_per_token_loss=token, fixed_response_normalizer=budget)


def objective_guard(args):
    path = ROOT / "miles/utils/arguments.py"
    tree = ast.parse(path.read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "miles_validate_args")
    # Invoke the actual isolated guard, not an independently retyped validation.
    nodes = f.body[1:3]
    assert ast.unparse(nodes[0]).startswith("fixed_normalizer =")
    f.body = nodes
    ns = {}
    exec(compile(ast.Module(body=[f], type_ignores=[]), str(path), "exec"), ns)
    return ns[f.name](args)


def test_actual_parser_registers_objective_flags_with_correct_defaults():
    tree = ast.parse((ROOT / "miles/utils/arguments.py").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--calculate-per-token-loss", action="store_true")
    flags = {"--grpo-fixed-response-normalizer", "--disable-grpo-std-normalization"}
    expressions = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and n.value.args and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value in flags]
    assert len(expressions) == 2
    exec(compile(ast.Module(body=expressions, type_ignores=[]), "actual_argument_registration", "exec"), {"parser": parser})
    default = parser.parse_args([])
    assert default.grpo_fixed_response_normalizer is None and default.grpo_std_normalization is True
    token = parser.parse_args(["--calculate-per-token-loss"])
    assert token.calculate_per_token_loss and token.grpo_std_normalization
    dr = parser.parse_args(["--disable-grpo-std-normalization", "--grpo-fixed-response-normalizer", "512"])
    assert dr.grpo_fixed_response_normalizer == 512 and not dr.grpo_std_normalization
    assert not dr.calculate_per_token_loss


@pytest.mark.parametrize(
    "change", [{}, {"calculate_per_token_loss": True}, {"grpo_std_normalization": True}, {"normalize_advantages": True}, {"rollout_max_response_len": 256}, {"grpo_fixed_response_normalizer": -1}, {"rewards_normalization": False}, {"advantage_estimator": "ppo"}, {"custom_pg_loss_reducer_function_path": "other"}]
)
def test_actual_argument_guard(change):
    args = SimpleNamespace(**(dict(grpo_fixed_response_normalizer=512, advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False, calculate_per_token_loss=False, normalize_advantages=False, custom_pg_loss_reducer_function_path=None, rollout_max_response_len=512) | change))
    if change:
        with pytest.raises(ValueError):
            objective_guard(args)
    else:
        objective_guard(args)


def test_empty_response_has_zero_valid_tokens_and_gradient():
    ns = boundary(1, 0)
    x = torch.ones(3, requires_grad=True)
    args = SimpleNamespace(calculate_per_token_loss=True, qkv_format="thd", grpo_std_normalization=True, recompute_loss_function=False, allgather_cp=False, global_batch_size=1, use_dynamic_global_batch_size=False)
    ns["get_loss_function"] = lambda args: lambda a, b, logits, reducer: (reducer(logits), {})
    loss, count, _ = ns["loss_function"](args, {"total_lengths": [10], "response_lengths": [3], "loss_masks": [torch.zeros(3)]}, 1, x)
    assert count.item() == 0 and loss.item() == 0
    loss.backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
