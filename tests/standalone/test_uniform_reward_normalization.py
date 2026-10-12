"""CPU-only actual-helper regression; no unrelated Ray fixture or GPU job."""

import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE = "miles/ray/rollout/train_data_conversion.py"
BASE = "424168348b2d0780441dd00a829dacc7c0c51149"
FUNCTIONS = {"_reward_group_segments", "_normalize_rewards_by_rollout", "_post_process_rewards"}


@dataclass
class Sample:
    reward: Any
    index: int | None
    rollout_id: int | None = None
    group_index: int | None = None

    def get_reward_value(self, args):
        return self.reward[args.reward_key] if isinstance(self.reward, dict) else self.reward


def load_helpers(source):
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert {node.name for node in selected} == FUNCTIONS
    namespace = {"torch": torch, "Any": Any, "Sample": Sample}
    exec(compile(ast.Module(body=selected, type_ignores=[]), MODULE, "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def helpers():
    current = load_helpers((ROOT / MODULE).read_text())
    original = load_helpers(subprocess.check_output(["git", "show", f"{BASE}:{MODULE}"], cwd=ROOT, text=True))
    return original, current


def arguments(**changes):
    values = dict(advantage_estimator="grpo", grpo_std_normalization=True, rewards_normalization=True,
                  n_samples_per_prompt=16, rollout_batch_size=1, reward_key="reward")
    return SimpleNamespace(**(values | changes))


def samples(values, **changes):
    return [Sample(value, index=i, **changes) for i, value in enumerate(values)]


def normalize(helper, values, args=None, rows=None, sizes=None):
    return helper["_normalize_rewards_by_rollout"](args or arguments(), rows or samples(values), values, sizes)


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, 0.1, 0.3, 0.6, 0.7, 0.8, 0.9, -0.3])
@pytest.mark.parametrize("size", [1, 2, 3, 16, 64])
@pytest.mark.parametrize("mode", ["grpo", "gspo", "reinforce_plus_plus_baseline", "mean_only"])
def test_equal_finite_rewards_are_exact_zero(helpers, value, size, mode):
    args = arguments(advantage_estimator=mode, grpo_std_normalization=mode != "mean_only")
    values = [value] * size
    assert normalize(helpers[1], values, args=args) == [0.0] * size


@pytest.mark.parametrize("values", [
    [0.3] * 15 + [0.3000001], [0.3] * 15 + [0.300000000001],
    [0.0, 1.0], [-0.3, -0.2, 0.1], [0.3, 0.7, 0.2, 0.8],
])
@pytest.mark.parametrize("estimator", ["grpo", "gspo", "reinforce_plus_plus_baseline"])
@pytest.mark.parametrize("std", [False, True])
def test_nonuniform_formula_is_bitwise_unchanged(helpers, values, estimator, std):
    args = arguments(advantage_estimator=estimator, grpo_std_normalization=std)
    assert normalize(helpers[1], values, args=args) == normalize(helpers[0], values, args=args)


def test_original_distinct_values_are_not_guarded_after_fp32_collapse(helpers, monkeypatch):
    values = [0.3] * 15 + [0.300000000001]
    assert torch.tensor(values).unique().numel() == 1
    expected = normalize(helpers[0], values)

    def forbidden(*args, **kwargs):
        raise AssertionError("Distinct original rewards must use the original arithmetic path")

    monkeypatch.setattr(torch, "zeros_like", forbidden)
    assert normalize(helpers[1], values) == expected


def test_irregular_explicit_groups_empty_groups_and_raw_rewards(helpers):
    values = [0.3] * 3 + [0.7] + [0.2, 0.8] * 8
    rows = samples(values)
    args = arguments()
    sizes = [0, 3, 1, 0, 16]
    old_raw, old = helpers[0]["_post_process_rewards"](args, rows, None, sizes)
    new_raw, new = helpers[1]["_post_process_rewards"](args, rows, None, sizes)
    assert new[:4] == [0.0] * 4
    assert new[4:] == old[4:]
    assert new_raw == old_raw == values
    assert [row.reward for row in rows] == values


def test_unique_rollouts_and_unequal_siblings_preserve_broadcast(helpers):
    rows = [Sample(0.3, i, rid, group) for i, rid, group in
            [(0, 5, 0), (1, 5, 0), (2, 5, 0), (3, 6, 0)]]
    rows += [Sample(0.2, 4, 7, 1), Sample(0.8, 5, 8, 1), Sample(0.8, 6, 8, 1)]
    values = [row.reward for row in rows]
    old, new = [normalize(h, values, rows=rows) for h in helpers]
    assert new[:4] == [0.0] * 4
    assert new[4:] == old[4:]
    assert new[5] == new[6]


def test_inconsistent_sibling_error_is_preserved(helpers):
    rows = [Sample(0.3, 0, 5, 0), Sample(0.7, 1, 5, 0)]
    for helper in helpers:
        with pytest.raises(ValueError, match="must share one reward"):
            normalize(helper, [0.3, 0.7], rows=rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_nonfinite_and_fp32_overflow_not_masked(helpers, value):
    values = [value] * 16
    results = []
    for helper in helpers:
        try:
            results.append(normalize(helper, values))
        except (ValueError, RuntimeError, OverflowError) as error:
            results.append(type(error))
    if isinstance(results[0], type):
        assert results[0] is results[1]
    else:
        old, new = map(torch.tensor, results)
        torch.testing.assert_close(new, old, atol=0, rtol=0, equal_nan=True)
        assert not bool(torch.isfinite(new).all())


@pytest.mark.parametrize("config", [dict(rewards_normalization=False), dict(advantage_estimator="ppo")])
def test_bypasses_and_reward_selection_remain_unchanged(helpers, config):
    args = arguments(**config)
    rows = samples([{"reward": 0.3, "unused": 0.9}] * 16)
    for helper in helpers:
        raw, output = helper["_post_process_rewards"](args, rows, None)
        assert raw == output == [0.3] * 16


def test_custom_hook_and_empty_handling_remain_unchanged(helpers):
    sentinel = object()
    rows = samples([0.3] * 16)
    for helper in helpers:
        assert helper["_post_process_rewards"](arguments(), rows, lambda a, s: sentinel) is sentinel
        assert helper["_normalize_rewards_by_rollout"](arguments(), [], [], [0]) == []
        assert helper["_post_process_rewards"](arguments(), [], None) == ([], [])


def test_invalid_group_size_and_missing_reward_errors_preserved(helpers):
    for helper in helpers:
        with pytest.raises(AssertionError):
            normalize(helper, [0.3] * 3, sizes=[2])
        with pytest.raises(KeyError):
            helper["_post_process_rewards"](arguments(), samples([{"other": 0.3}]), None)
