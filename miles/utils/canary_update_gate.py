"""Fail-closed inter-update gate for a bounded mechanics canary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


GATE_PATH_ENV = "MILES_CANARY_UPDATE1_GATE_PATH"
METRIC_DIR_ENV = "MILES_CI_GATE_RECORD_DIR"
RUN_ID_ENV = "MILES_EXPERIMENT_RUN_ID"
REQUIRED_METRICS = {
    "train/grad_norm",
    "train/loss",
    "train/ppo_kl",
    "train/tis_clipfrac",
    "train/train_rollout_kl",
    "train/train_rollout_logprob_abs_diff",
}


def _load_step_metrics(record_dir: Path, *, step: int) -> dict[str, float]:
    if not record_dir.is_dir():
        raise RuntimeError(f"canary metric record directory is missing: {record_dir}")
    observed: dict[str, list[float]] = {key: [] for key in REQUIRED_METRICS}
    paths = sorted(record_dir.glob("*.jsonl"))
    if not paths:
        raise RuntimeError("canary metric record directory contains no JSONL records")
    for path in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed canary metric record {path}:{line_number}") from exc
            metric = record.get("metric")
            if metric not in observed:
                continue
            series = record.get("series")
            if not isinstance(series, list):
                raise RuntimeError(f"canary metric {metric!r} has malformed series")
            for point in series:
                if not isinstance(point, list) or len(point) != 2:
                    raise RuntimeError(f"canary metric {metric!r} has malformed point")
                if point[0] == step:
                    value = point[1]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise RuntimeError(f"canary metric {metric!r} step={step} is non-numeric")
                    value = float(value)
                    if not math.isfinite(value):
                        raise RuntimeError(f"canary metric {metric!r} step={step} is non-finite")
                    observed[metric].append(value)

    result = {}
    for metric, values in observed.items():
        unique = set(values)
        if len(unique) != 1:
            raise RuntimeError(f"canary metric {metric!r} step={step} must have one unique value; observed={values}")
        result[metric] = values[0]
    return result


def _validate_native_save(args: Any, *, rollout_id: int) -> dict[str, Any]:
    path = Path(args.save).parent / "receipts" / f"native-adapter-rollout-{rollout_id}.json"
    if not path.is_file():
        raise RuntimeError(f"canary native adapter receipt is missing: {path}")
    value = json.loads(path.read_text())
    if not (
        value.get("schema_version") == 1
        and value.get("status") == "PASS"
        and value.get("rollout_id") == rollout_id
        and value.get("ranks") == 128
        and value.get("adapter_tensors_per_rank") == 1206
        and value.get("shared_factors_per_rank") == 462
        and value.get("shared_factors_replica_equal") is True
        and value.get("all_values_finite") is True
        and value.get("nonzero_b_tensors_across_rank_shards", 0) > 0
    ):
        raise RuntimeError(f"canary native adapter receipt failed: {value}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ranks": value["ranks"],
        "shared_factors_replica_equal": True,
        "all_values_finite": True,
        "nonzero_b_tensors_across_rank_shards": value["nonzero_b_tensors_across_rank_shards"],
    }


def validate_and_write_update1_gate(args: Any, smoke_receipt: dict[str, Any]) -> dict[str, Any]:
    """Authorize only update 2 after update 1 passes every bounded gate."""

    gate_path_value = os.environ.get(GATE_PATH_ENV)
    metric_dir_value = os.environ.get(METRIC_DIR_ENV)
    run_id = os.environ.get(RUN_ID_ENV)
    if not gate_path_value or not metric_dir_value or not run_id:
        raise RuntimeError(f"{GATE_PATH_ENV}, {METRIC_DIR_ENV}, and {RUN_ID_ENV} are required")
    gate_path = Path(gate_path_value)
    metric_dir = Path(metric_dir_value)
    if gate_path.parent.name != run_id:
        raise RuntimeError("canary update-1 gate path is outside the immutable run namespace")
    if metric_dir.parent.parent != gate_path.parent:
        raise RuntimeError("canary metric directory is outside the immutable run namespace")
    if gate_path.exists():
        raise RuntimeError(f"canary update-1 gate receipt already exists: {gate_path}")

    expected_args = {
        "start_rollout_id": 0,
        "num_rollout": 2,
        "num_steps_per_rollout": 1,
        "save_interval": 1,
        "skip_actor_forward_only": False,
        "use_tis": True,
        "check_lora_weight_equal": True,
        "check_initial_lora_zero_effect": True,
    }
    observed_args = {key: getattr(args, key, None) for key in expected_args}
    if observed_args != expected_args:
        raise RuntimeError(f"canary update-1 gate argument mismatch: expected={expected_args} observed={observed_args}")

    metrics = _load_step_metrics(metric_dir, step=0)
    if not 0 <= metrics["train/train_rollout_logprob_abs_diff"] <= 0.06:
        raise RuntimeError("update 1 train/rollout logprob difference exceeds 0.06")
    if abs(metrics["train/ppo_kl"]) > 0.03 or abs(metrics["train/train_rollout_kl"]) > 0.03:
        raise RuntimeError("update 1 KL exceeds 0.03")
    if not 0 <= metrics["train/tis_clipfrac"] <= 0.05:
        raise RuntimeError("update 1 TIS clip fraction exceeds 0.05")
    if metrics["train/grad_norm"] < 0:
        raise RuntimeError("update 1 gradient norm is negative")

    if not (
        smoke_receipt.get("schema_version") == 1
        and smoke_receipt.get("status") == "PASS"
        and smoke_receipt.get("stage") == "post_update_1"
        and smoke_receipt.get("engine_count") == 4
        and smoke_receipt.get("request_count") == 32
        and smoke_receipt.get("output_count") == 144
        and smoke_receipt.get("token_count", 0) > 0
        and isinstance(smoke_receipt.get("evidence_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", smoke_receipt["evidence_sha256"])
    ):
        raise RuntimeError(f"post-update operational smoke failed: {smoke_receipt}")

    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "run_id": run_id,
        "completed_update": 1,
        "authorized_next_update": 2,
        "publication_completed_by_blocking_return": True,
        "frozen_old_policy_path": "default_actor_forward_before_optimizer_step",
        "metrics": metrics,
        "native_adapter_save": _validate_native_save(args, rollout_id=0),
        "post_update_operational_smoke": smoke_receipt,
        "scope": "BOUNDED_TWO_ROLLOUT_TWO_UPDATE_MECHANICS_ONLY",
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = gate_path.with_suffix(gate_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(gate_path)
    return receipt
