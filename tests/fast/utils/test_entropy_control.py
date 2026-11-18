from copy import deepcopy
from dataclasses import replace
import ast
import asyncio
import json
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from miles.utils.entropy_control import EntropyController, EntropyControlSpec, decide, merge_response_fragments, summarize_responses
from miles.utils.entropy_control import digest, spec_from_args


def controller(*, mode="direct", horizon=40):
    return EntropyController("test", "a" * 40, EntropyControlSpec(horizon, 48, 16, mode))


def rows(*, entropy=0.4, selected=16):
    return [dict(sample_id=g * 16 + i, group_index=g, theme=f"theme{g % 3}", scenario="scenario", entropy_sum=entropy * (i + 1), original_tokens=i + 1, original_advantage=(1 if i == 0 else -1 if i == 1 else 0) if g < selected else 0) for g in range(48) for i in range(16)]


def observation(c, **kwargs):
    packet = c.snapshot
    return dict(run=c.run, source=c.source, completed_update=packet["for_update"], used_sha256=packet["sha256"], responses=rows(**kwargs))


def step(c, **kwargs):
    c.observe(observation(c, **kwargs))
    return c.commit_publication(completed_update=c.completed + 1)


def test_two_batch_warmup_and_publication_boundary():
    c = controller()
    used = c.snapshot
    c.observe(observation(c, entropy=0.45))
    assert c.snapshot == used
    assert c.completed == 0
    with pytest.raises(ValueError):
        c.observe(observation(c))
    with pytest.raises(ValueError):
        c.commit_publication(completed_update=2)
    event = c.commit_publication(completed_update=1)
    assert event["next"]["scale"] == 0.8
    assert event["reason"] == "warmup_hold"
    event = step(c, entropy=0.45)
    assert event["used"]["scale"] == 0.8
    assert event["rolling2_entropy"] == pytest.approx(0.45)
    assert event["next"]["scale"] == pytest.approx(0.95)
    assert not event["guard"]


def test_selected_groups_include_zero_advantage_siblings_and_equal_responses():
    r = rows()
    # A zero-advantage sibling in a selected group still contributes to feedback.
    r[2]["entropy_sum"] = 3.0
    result = summarize_responses(r, controller().spec)
    assert result["selected_group_entropy"] == pytest.approx(0.4 + 0.6 / 256)
    assert result["all_response_entropy"] == pytest.approx(0.4 + 0.6 / 768)
    assert result["selected_group_count"] == 16
    assert result["coverage_valid"]
    # Selected entropy is response-mean, not token-pooled entropy.
    r = rows(entropy=0.0)
    r[0].update(entropy_sum=1.0, original_tokens=1)
    r[1].update(entropy_sum=100.0, original_tokens=100)
    assert summarize_responses(r, controller().spec)["selected_group_entropy"] == pytest.approx(2 / 256)


def test_insufficient_latest_batch_holds_without_older_backfill():
    c = controller()
    step(c, entropy=0.4)
    step(c, entropy=0.45)
    used = c.scale
    event = step(c, entropy=0.1, selected=15)
    assert event["reason"] == "insufficient_feedback_groups"
    assert c.scale == used
    event = step(c, entropy=0.1)
    assert not event["coverage_valid"]
    assert c.scale == used
    event = step(c, entropy=0.1)
    assert event["coverage_valid"]
    assert c.scale == 0.6


@pytest.mark.parametrize("used,expected", [(0.6, 1.0), (1.05, 1.05), (1.1, 1.1)])
def test_independent_current_global_guard_preserves_used_scale(used, expected):
    spec = controller().spec
    history = (dict(selected_group_entropy=None, coverage_valid=False, all_response_entropy=0.0), dict(selected_group_entropy=None, coverage_valid=False, all_response_entropy=0.51))
    result = decide(history, completed=2, used_scale=used, spec=spec)
    assert result["guard"]
    assert result["final_scale"] == expected
    history[-1]["all_response_entropy"] = 0.5
    assert not decide(history, completed=2, used_scale=used, spec=spec)["guard"]
    history[-1]["all_response_entropy"] = 0.9
    assert not decide(history[-1:], completed=1, used_scale=used, spec=spec)["guard"]


def test_observe_arm_preserves_unit_scale_for_every_update():
    c = controller(mode="observe")
    for _ in range(40):
        assert c.snapshot["scale"] == 1.0
        assert step(c, entropy=0.1)["reason"] == "observe_only"
    assert c.snapshot is None


def test_pending_save_restore_and_finite_terminal_no_update41():
    c = controller()
    for _ in range(19):
        step(c)
    c.observe(observation(c))
    state = c.state_dict()
    assert state["pending"]["used"]["for_update"] == 20
    restored = EntropyController.restore(state)
    assert restored.state_dict() == state
    assert c.commit_publication(completed_update=20) == restored.commit_publication(completed_update=20)
    for _ in range(20):
        assert step(c) == step(restored)
    assert c.completed == 40 and c.snapshot is None
    assert EntropyController.restore(c.state_dict()).snapshot is None
    with pytest.raises(ValueError):
        c.observe({})
    with pytest.raises(ValueError):
        c.commit_publication(completed_update=41)


@pytest.mark.parametrize("field,value", [("run", "other"), ("source", "b" * 40), ("completed_update", 2), ("used_sha256", "bad")])
def test_mixed_observation_identity_fails(field, value):
    c = controller()
    v = observation(c)
    v[field] = value
    with pytest.raises(ValueError):
        c.observe(v)


@pytest.mark.parametrize("change", ["missing", "duplicate", "group", "category", "empty", "nan"])
def test_partial_or_invalid_populations_fail(change):
    c = controller()
    v = observation(c)
    r = v["responses"]
    if change == "missing":
        r.pop()
    elif change == "duplicate":
        r[-1] = deepcopy(r[0])
    elif change == "group":
        r[0]["group_index"] = 100
    elif change == "category":
        r[0]["theme"] = "other"
    elif change == "empty":
        r[0]["original_tokens"] = 0
    else:
        r[0]["entropy_sum"] = float("nan")
    with pytest.raises(ValueError):
        c.observe(v)


def test_missing_or_tampered_checkpoint_history_fails():
    c = controller()
    step(c)
    step(c)
    state = c.state_dict()
    for key, value in (("history", []), ("next", {}), ("scale", 0.7), ("completed", 41)):
        broken = deepcopy(state)
        broken[key] = value
        with pytest.raises(ValueError):
            EntropyController.restore(broken)
    broken = deepcopy(state)
    broken["history"][0]["completed_update"] = 0
    with pytest.raises(ValueError):
        EntropyController.restore(broken)


def test_nonfinite_constants_rejected():
    with pytest.raises(ValueError):
        replace(controller().spec, gain=float("nan"))


def test_dp2_cp2_aggregation_uses_token_sums_before_response_means():
    originals = rows()
    fragments = []
    # Unequal CP fragments, including entirely empty response slices on CP rank1.
    for row in originals:
        count = row["original_tokens"]
        for cp, n in ((0, (count + 1) // 2), (1, count // 2)):
            fragments.append(
                {
                    **{key: row[key] for key in ("sample_id", "group_index", "theme", "scenario", "original_tokens")},
                    "cp_rank": cp,
                    "local_tokens": n,
                    "prompt_loss_scale": 1.0,
                    "advantage_representation": "torch.float32",
                    "original_advantage": row["original_advantage"] if n else None,
                    "statistics": {"entropy_sum": 0.4 * n},
                }
            )
    # Simulate length-balanced DP ownership and arbitrary arrival order.
    fragments = fragments[::2] + list(reversed(fragments[1::2]))
    merged = merge_response_fragments(fragments, cp_size=2)
    assert summarize_responses(merged, controller().spec) == summarize_responses(originals, controller().spec)
    for invalid in (fragments[:-1], fragments + [fragments[0]]):
        with pytest.raises(ValueError):
            merge_response_fragments(invalid, cp_size=2)
    bad = deepcopy(fragments)
    bad[0]["original_tokens"] += 1
    with pytest.raises(ValueError):
        merge_response_fragments(bad, cp_size=2)


def _native_entropy_coordinator():
    # Exercise the actual native coordinator without importing CUDA/Ray/adapter model dependencies.
    source = Path(__file__).resolve().parents[3] / "miles/ray/train/entropy.py"
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(
        asyncio=asyncio,
        json=json,
        math=math,
        os=os,
        Path=Path,
        quote=quote,
        EntropyController=EntropyController,
        digest=digest,
        spec_from_args=spec_from_args,
        TrainStepOutcome=SimpleNamespace(NORMAL="NORMAL"),
        logger=logging.getLogger(__name__),
        read_training_state=lambda path: json.loads((path / "training_state.json").read_text()),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), namespace)
    return namespace["EntropyCoordinator"]


def _statistical_observation(c):
    value = observation(c)
    for row in value["responses"]:
        n, a = row["original_tokens"], row["original_advantage"]
        row.update(prompt_loss_scale=1.0, tis_weight_sum=n, tis_weight_squared_sum=n, train_rollout_abs_diff_sum=0.0, train_rollout_k3_sum=0.0, tis_below_floor_tokens=0, tis_above_cap_tokens=0, low_probability_negative_tokens=0, pg_coefficient_mass=abs(a) * n, applied_pg_coefficient_mass=abs(a) * n)
        for sign, active in (("positive", a > 0), ("negative", a < 0), ("zero", a == 0)):
            row[sign + "_tokens"] = n if active else 0
            row[sign + "_entropy_sum"] = row["entropy_sum"] if active else 0.0
    return value


@pytest.mark.parametrize("mode", ["observe", "direct"])
@pytest.mark.parametrize("tracking_error", [False, True])
def test_native_coordinator_all128_pending_save_restore_and_publication(tmp_path, mode, tracking_error):
    cls = _native_entropy_coordinator()
    args = SimpleNamespace(wandb_run_id="test", entropy_control_source_commit="a" * 40, num_rollout=40, rollout_batch_size=48, n_samples_per_prompt=16, entropy_control_mode=mode, backend=SimpleNamespace(save=str(tmp_path / "checkpoints"), world_size=128), lora_adapter_path=None)
    coordinator = cls(args)
    coordinator.validate_loaded_cursors([0] * 128)
    recorded = []

    class Cell:
        async def execute(self, method, **kwargs):
            recorded.append((method, kwargs))
            if method == "set_entropy_packet":
                return [kwargs["packet"]["sha256"]] * 128
            if method == "set_entropy_checkpoint_state":
                return [kwargs["state"]["pending"]["used"]["sha256"]] * 128
            if tracking_error:
                assert kwargs["kill_on_failure"] is False
                raise RuntimeError("Transient tracking transport failure")
            return [kwargs["event"]["completed_update"]] * 128

    cells = [Cell()]
    asyncio.run(coordinator.prepare(cells, 0))
    outputs = [SimpleNamespace(outcome="NORMAL", optimizer_step_valid=True, entropy_observation=None) for _ in range(128)]
    outputs[90].entropy_observation = _statistical_observation(coordinator.controller)
    coordinator.accept(outputs)
    with pytest.raises(ValueError):
        asyncio.run(coordinator.prepare(cells, 0))
    with pytest.raises(ValueError):
        asyncio.run(coordinator.published(cells, rollout_id=0, weight_version=1, failed_cell_ids=("failed",)))
    assert coordinator.controller.completed == 0
    asyncio.run(coordinator.prepare_save(cells))
    saved = recorded[-1][1]["state"]
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "training_state.json").write_text(json.dumps({"iteration": 0, "entropy_control": saved}))
    args.lora_adapter_path = str(adapter)
    restored = cls(args)
    restored.validate_loaded_cursors([1] * 128)
    with pytest.raises(ValueError):
        restored.validate_loaded_cursors([1] * 127 + [0])
    # Re-publishing restored native weights commits pending1 before update2 is prepared.
    asyncio.run(restored.published(cells, rollout_id=None, weight_version=1, failed_cell_ids=()))
    assert restored.controller.completed == 1
    asyncio.run(restored.prepare(cells, 1))
    assert recorded[-1][1]["packet"]["for_update"] == 2
    event = json.loads((tmp_path / "entropy/update-0001-decision.json").read_text())
    assert event["published_weight_version"] == 1
    assert event["used"]["scale"] == (1.0 if mode == "observe" else 0.8)


def test_native_coordinator_refuses_missing_or_invalid_optimizer_results(tmp_path):
    cls = _native_entropy_coordinator()
    args = SimpleNamespace(wandb_run_id="test", entropy_control_source_commit="a" * 40, num_rollout=40, rollout_batch_size=48, n_samples_per_prompt=16, entropy_control_mode="direct", backend=SimpleNamespace(save=str(tmp_path / "checkpoints"), world_size=128), lora_adapter_path=None)
    c = cls(args)
    results = [SimpleNamespace(outcome="NORMAL", optimizer_step_valid=True, entropy_observation=None) for _ in range(128)]
    results[0].entropy_observation = _statistical_observation(c.controller)
    with pytest.raises(ValueError):
        c.accept(results[:-1])
    results[-1].optimizer_step_valid = False
    with pytest.raises(ValueError):
        c.accept(results)
    results[-1].optimizer_step_valid = True
    results[-1].entropy_observation = results[0].entropy_observation
    with pytest.raises(ValueError):
        c.accept(results)
