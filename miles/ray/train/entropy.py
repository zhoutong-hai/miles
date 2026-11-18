"""Native coordinator integration for finite original-response entropy feedback."""

import asyncio
import json
import logging
import math
import os
from pathlib import Path
from urllib.parse import quote

from miles.backends.training_utils.checkpoint.adapter.adapter_dir import read_training_state
from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.entropy_control import EntropyController, digest, spec_from_args

logger = logging.getLogger(__name__)


def _persist(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Conflicting immutable entropy artifact: {path.name}")
        return
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def observation_metrics(observation: dict, event: dict) -> dict:
    rows = observation["responses"]
    count = sum(row["original_tokens"] for row in rows)

    def total(key):
        return math.fsum(row[key] for row in rows)

    summary = event["latest_two_batches"][-1]
    metrics = {
        "entropy/all_equal_response": summary["all_response_entropy"],
        "entropy/all_pooled_token": total("entropy_sum") / count,
        "entropy/selected_groups": summary["selected_group_count"],
        "entropy/selected_group_fraction": summary["selected_group_count"] / event["used"]["spec"]["groups_per_batch"],
        "entropy/original_tokens": count,
        "entropy/original_responses": len(rows),
        "entropy/zero_denominator_responses": sum(row["original_tokens"] == 0 for row in rows),
        "entropy_control/lambda_used": event["used"]["scale"],
        "entropy_control/lambda_next": None if event["next"] is None else event["next"]["scale"],
        "entropy_control/guard": int(event["guard"]),
        "entropy_control/coverage_valid": int(event["coverage_valid"]),
        "entropy_control/warmup": int(event["reason"] == "warmup_hold"),
        "entropy_control/coverage_hold": int(event["reason"] == "insufficient_feedback_groups"),
        "entropy_control/lower_saturated": int(event["lower_saturated"]),
        "entropy_control/upper_saturated": int(event["upper_saturated"]),
        "entropy_control/closed": int(event["closed"]),
        "entropy/selected_group_equal_response": summary["selected_group_entropy"],
        "entropy/selected_group_rolling2": event["rolling2_entropy"],
        "entropy_control/formula": event["formula_result"],
    }
    for key in ("train_rollout_abs_diff", "train_rollout_k3", "tis_weight", "tis_weight_squared"):
        metrics[f"entropy/{key}_mean"] = total(key + "_sum") / count
    sum_w, sum_w2 = total("tis_weight_sum"), total("tis_weight_squared_sum")
    metrics["entropy/tis_ess_fraction"] = sum_w * sum_w / (count * sum_w2) if sum_w2 else 0.0
    for key in ("tis_below_floor", "tis_above_cap", "low_probability_negative"):
        metrics[f"entropy/{key}_fraction"] = total(key + "_tokens") / count
    normalizer = sum(max(row["original_tokens"], 1) for row in rows)
    for key in ("pg_coefficient_mass", "applied_pg_coefficient_mass"):
        metrics[f"entropy/{key}"] = math.fsum(row[key] * row["prompt_loss_scale"] / normalizer for row in rows)
    for sign in ("positive", "negative", "zero"):
        n = total(sign + "_tokens")
        metrics[f"entropy/{sign}/token_fraction"] = n / count
        metrics[f"entropy/{sign}/entropy_mass_fraction"] = total(sign + "_entropy_sum") / total("entropy_sum") if total("entropy_sum") else 0.0
        if n:
            metrics[f"entropy/{sign}/conditional_entropy"] = total(sign + "_entropy_sum") / n
    for category in ("theme", "scenario"):
        for label in sorted({row[category] for row in rows}):
            selected = [row for row in rows if row[category] == label]
            prefix = f"entropy/{category}/{quote(label, safe='-_')}"
            metrics[prefix + "/groups"] = len({row["group_index"] for row in selected})
            metrics[prefix + "/responses"] = len(selected)
            metrics[prefix + "/equal_response"] = math.fsum(row["entropy_sum"] / row["original_tokens"] / len(selected) for row in selected)
    return {key: value for key, value in metrics.items() if value is not None}


class EntropyCoordinator:
    def __init__(self, args):
        self.args = args
        self.controller = EntropyController(args.wandb_run_id, args.entropy_control_source_commit, spec_from_args(args))
        self.root = Path(args.backend.save).parent / "entropy"
        self.expected_workers = args.backend.world_size
        if args.lora_adapter_path is not None:
            state = read_training_state(Path(args.lora_adapter_path))
            if state is None or "entropy_control" not in state:
                raise ValueError("Selected adapter checkpoint lacks native entropy controller state")
            restored = EntropyController.restore(state["entropy_control"])
            if restored.run != self.controller.run or restored.source != self.controller.source or restored.spec != self.controller.spec or restored.completed + (restored.pending is not None) != state["iteration"] + 1:
                raise ValueError("Native optimizer/controller source, specification or cursor differs")
            self.controller = restored

    def validate_loaded_cursors(self, cursors):
        expected = self.controller.completed + (self.controller.pending is not None)
        if len(cursors) != self.expected_workers or set(cursors) != {expected}:
            raise ValueError("Native optimizer cursors differ from saved entropy state")

    async def _all(self, cells, method, *, kill_on_failure=True, timeout=None, **kwargs):
        responses = await asyncio.gather(*(cell.execute(method, kill_on_failure=kill_on_failure, timeout=timeout, **kwargs) for cell in cells))
        flat = [item for response in responses for item in response]
        if len(flat) != self.expected_workers:
            raise ValueError("Entropy operation did not acknowledge every training worker")
        return flat

    async def prepare(self, cells, rollout_id):
        packet = self.controller.snapshot
        if self.controller.pending is not None or packet is None or packet["for_update"] != rollout_id + 1:
            raise ValueError("Unpublished, skipped or terminal entropy update")
        acknowledgements = await self._all(cells, "set_entropy_packet", packet=packet)
        if set(acknowledgements) != {packet["sha256"]}:
            raise ValueError("Training workers disagree on the entropy packet")

    def accept(self, results):
        if len(results) != self.expected_workers or any(result.outcome != TrainStepOutcome.NORMAL or result.optimizer_step_valid is not True for result in results):
            raise ValueError("Entropy observation requires every successful finite optimizer step")
        observations = [result.entropy_observation for result in results if result.entropy_observation is not None]
        if len(observations) != 1:
            raise ValueError("Entropy observation requires exactly one complete DP×CP writer")
        (observation,) = observations
        self.controller.observe(observation)
        _persist(self.root / f"update-{self.controller.completed + 1:04d}-observation.json", observation)

    async def prepare_save(self, cells):
        state = self.controller.state_dict()
        acknowledgements = await self._all(cells, "set_entropy_checkpoint_state", state=state)
        if set(acknowledgements) != {state["pending"]["used"]["sha256"]}:
            raise ValueError("Native save workers disagree on the entropy state")

    async def published(self, cells, *, rollout_id, weight_version, failed_cell_ids):
        pending = self.controller.pending
        if pending is None:
            if rollout_id is not None:
                raise ValueError("Trained entropy publication lacks pending optimizer feedback")
            return
        completed = pending["used"]["for_update"]
        if (rollout_id is not None and completed != rollout_id + 1) or weight_version is None or failed_cell_ids:
            raise ValueError("Entropy feedback requires successful publication to every selected engine")
        observation = pending["observation"]
        if digest(observation) != pending["observation_sha256"]:
            raise ValueError("Restored pending entropy observation seal differs")
        event = self.controller.commit_publication(completed_update=completed, published_weight_version=weight_version)
        _persist(self.root / f"update-{completed:04d}-decision.json", event)
        metrics = observation_metrics(observation, event)
        _persist(self.root / f"update-{completed:04d}-metrics.json", metrics)
        try:
            acknowledgements = await self._all(cells, "log_entropy_publication", event=event, metrics=metrics, kill_on_failure=False, timeout=60.0)
            if set(acknowledgements) != {completed}:
                raise ValueError("Entropy publication logging acknowledgement differs")
        except Exception:
            # Publication and optimizer work already succeeded. Keep durable metrics for reconciliation;
            # a tracking failure must not kill trained workers or roll the entropy controller back.
            logger.exception("Entropy tracking DEGRADED at update%d; private decision/metrics preserved", completed)
