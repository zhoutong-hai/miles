"""Exercise sealing and metric publication with the v0.2 orchestrator slice."""

import json
from types import SimpleNamespace

import pytest

from miles.utils.reinforce_observations import finalize


def _packet(tmp_path):
    root = tmp_path / "update-000001" / "attempt-0"
    root.mkdir(parents=True)
    for rank in range(2):
        (root / f"rank-{rank}-binding.json").write_text(json.dumps({"rank": rank, "world_size": 2}))
        (root / f"rank-{rank}-complete.json").write_text(json.dumps({"rank": rank, "completed_update": 1, "outcome": "NORMAL"}))
    inputs, frames = [], []
    for sample, advantage in enumerate([1.0, -1.0, 0.0, 0.0]):
        inputs.append({"sample": sample, "group": sample // 2, "group_tokens": 2, "factor": 1.0, "tags": {"theme": "fixture"}, "q": [-1.0], "advantage": [advantage], "mask": [1]})
        frames.append({"sample": sample, "group": sample // 2, "factor": [1.0], "q": [-1.0], "A": [advantage], "mask": [1], "p": [-1.0], "ratio": [1.0], "weight": [1.0], "entropy": [0.5]})
    (root / "rank-0-input.json").write_text(json.dumps({"rows": inputs}))
    (root / "rank-0-frames.jsonl").write_text(json.dumps({"rows": frames}) + "\n")
    values = dict(reinforce_diagnostics_dir=str(tmp_path), global_batch_size=4, rollout_batch_size=2, n_samples_per_prompt=2, mis_positive_low=0.2, mis_positive_high=5.0, mis_negative_low=0.5, mis_negative_high=2.0)
    return root, values


def _tracking(monkeypatch):
    # Tracking is a deployment dependency; these tests inspect the exact handoff.
    import sys
    import types

    calls = []
    module = types.ModuleType("miles.utils.tracking_utils")
    module.tracking = SimpleNamespace(log=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "miles.utils.tracking_utils", module)
    return calls


def test_seal_and_tracking_without_launcher_scaling_fields(tmp_path, monkeypatch):
    root, values = _packet(tmp_path)
    calls = _tracking(monkeypatch)
    finalize(SimpleNamespace(**values), 0)
    seal = json.loads((root / "complete.json").read_text())
    assert seal["native_NORMAL_ranks"] == 2
    assert seal["learned_publication_returned"] is True
    assert seal["metrics"]["train/completed_optimizer_updates"] == 1
    assert seal["metrics"]["train/sfc/raw_original_tokens"] == 4
    assert seal["metrics"]["train/sfc/entropy_equal_response_mean"] == 0.5
    assert len(calls) == 1 and calls[0][1] == {"step_key": "train/step"}


def test_actual_typed_orchestrator_config(tmp_path, monkeypatch):
    from miles.utils.args.runtime import OrchestratorConfig

    root, values = _packet(tmp_path)
    _tracking(monkeypatch)
    args = OrchestratorConfig.model_construct(**values)
    assert not hasattr(args, "actor_num_nodes")
    finalize(args, 0)
    assert (root / "complete.json").exists()


@pytest.mark.parametrize("corrupt", ["missing_binding", "wrong_world", "missing_completion", "wrong_rank", "not_normal", "wrong_update", "missing_response", "changed_q"])
def test_incomplete_or_corrupt_evidence_never_seals(tmp_path, monkeypatch, corrupt):
    root, values = _packet(tmp_path)
    calls = _tracking(monkeypatch)
    if corrupt == "missing_binding":
        (root / "rank-1-binding.json").unlink()
    elif corrupt == "wrong_world":
        (root / "rank-1-binding.json").write_text(json.dumps({"rank": 1, "world_size": 3}))
    elif corrupt == "missing_completion":
        (root / "rank-1-complete.json").unlink()
    elif corrupt in {"wrong_rank", "not_normal", "wrong_update"}:
        path = root / "rank-1-complete.json"
        row = json.loads(path.read_text())
        row.update({"wrong_rank": {"rank": 0}, "not_normal": {"outcome": "SKIPPED"}, "wrong_update": {"completed_update": 2}}[corrupt])
        path.write_text(json.dumps(row))
    else:
        path = root / "rank-0-frames.jsonl"
        data = json.loads(path.read_text())
        if corrupt == "missing_response":
            data["rows"].pop()
        else:
            data["rows"][0]["q"] = [-2.0]
        path.write_text(json.dumps(data) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize(SimpleNamespace(**values), 0)
    assert not (root / "complete.json").exists() and not calls
