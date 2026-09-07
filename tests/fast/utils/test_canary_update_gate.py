import json
from types import SimpleNamespace

import pytest

from miles.utils.canary_update_gate import (
    GATE_PATH_ENV,
    METRIC_DIR_ENV,
    RUN_ID_ENV,
    REQUIRED_METRICS,
    validate_and_write_update1_gate,
)


def _fixture(tmp_path, monkeypatch):
    run_id = "e006-test-canary"
    root = tmp_path / run_id
    gate_path = root / "update1-gate.json"
    metric_dir = root / "runtime-evidence" / "ci-metrics"
    metric_dir.mkdir(parents=True)
    metrics = {
        "train/grad_norm": 1.0,
        "train/loss": 0.5,
        "train/ppo_kl": 0.01,
        "train/tis_clipfrac": 0.02,
        "train/train_rollout_kl": 0.01,
        "train/train_rollout_logprob_abs_diff": 0.04,
    }
    with (metric_dir / "rank0.jsonl").open("w") as stream:
        for key, value in metrics.items():
            stream.write(json.dumps({"metric": key, "series": [[0, value]]}) + "\n")

    save = root / "checkpoints" / "model"
    receipt_dir = save.parent / "receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "native-adapter-rollout-0.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "rollout_id": 0,
                "ranks": 128,
                "adapter_tensors_per_rank": 1206,
                "shared_factors_per_rank": 462,
                "shared_factors_replica_equal": True,
                "all_values_finite": True,
                "nonzero_b_tensors_across_rank_shards": 3,
            }
        )
    )
    args = SimpleNamespace(
        save=str(save),
        start_rollout_id=0,
        num_rollout=2,
        num_steps_per_rollout=1,
        save_interval=1,
        skip_actor_forward_only=False,
        use_tis=True,
        check_lora_weight_equal=True,
        check_initial_lora_zero_effect=True,
    )
    smoke = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "post_update_1",
        "engine_count": 4,
        "request_count": 32,
        "output_count": 144,
        "token_count": 288,
        "evidence_sha256": "a" * 64,
    }
    monkeypatch.setenv(GATE_PATH_ENV, str(gate_path))
    monkeypatch.setenv(METRIC_DIR_ENV, str(metric_dir))
    monkeypatch.setenv(RUN_ID_ENV, run_id)
    return gate_path, metric_dir, args, smoke


def test_update1_gate_authorizes_only_second_update(tmp_path, monkeypatch):
    gate_path, _, args, smoke = _fixture(tmp_path, monkeypatch)

    receipt = validate_and_write_update1_gate(args, smoke)

    assert receipt["status"] == "PASS"
    assert receipt["completed_update"] == 1
    assert receipt["authorized_next_update"] == 2
    assert set(receipt["metrics"]) == REQUIRED_METRICS
    assert json.loads(gate_path.read_text()) == receipt


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("train/train_rollout_logprob_abs_diff", 0.061, "difference exceeds 0.06"),
        ("train/ppo_kl", 0.031, "KL exceeds 0.03"),
        ("train/tis_clipfrac", 0.051, "TIS clip fraction exceeds 0.05"),
    ],
)
def test_update1_gate_rejects_threshold_breach(tmp_path, monkeypatch, metric, value, message):
    _, metric_dir, args, smoke = _fixture(tmp_path, monkeypatch)
    path = metric_dir / "rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["metric"] == metric:
            row["series"] = [[0, value]]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match=message):
        validate_and_write_update1_gate(args, smoke)


def test_update1_gate_rejects_nonfinite_metric(tmp_path, monkeypatch):
    _, metric_dir, args, smoke = _fixture(tmp_path, monkeypatch)
    path = metric_dir / "rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["metric"] == "train/loss":
            row["series"] = [[0, "NaN"]]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match="non-numeric"):
        validate_and_write_update1_gate(args, smoke)
