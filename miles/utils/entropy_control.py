"""Delayed entropy feedback from original response statistics, independent of loss diagnostics.

The native coordinator owns this state. Training produces a pending observation;
only successful weight publication commits it and authorizes the next packet.
"""

import hashlib
import json
import math
import struct
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace


def spec_from_args(args) -> "EntropyControlSpec":
    return EntropyControlSpec(horizon=args.num_rollout, groups_per_batch=args.rollout_batch_size, responses_per_group=args.n_samples_per_prompt, mode=args.entropy_control_mode)


def validate_entropy_control_args(args) -> None:
    if args.entropy_control_continuation is not None and (
        args.entropy_control_mode == "off" or args.lora_adapter_path is None or args.load is None
        or args.finetune or args.no_load_optim or args.no_load_rng
    ):
        raise ValueError("Entropy continuation requires an enabled controller and full native adapter resume")
    if args.entropy_control_mode == "off":
        return
    spec_from_args(args)
    if (
        args.train_backend != "megatron"
        or args.policy_loss_aggregation != "prompt-mean"
        or not args.observe_training_entropy
        or not args.observe_advantage_sign_metrics
        or args.indep_dp
        or args.fully_async
        or args.use_critic
        or args.multi_lora
        or args.lora_rank <= 0
        or args.num_steps_per_rollout != 1
        or args.debug_disable_optimizer
        or args.save is None
        or not args.wandb_run_id
        or not args.entropy_control_source_commit
        or len(args.entropy_control_source_commit) != 40
        or any(c not in "0123456789abcdef" for c in args.entropy_control_source_commit)
    ):
        raise ValueError("Entropy feedback requires synchronous one-step single-LoRA Megatron prompt-mean PG, native entropy/sign observation, optimizer and immutable run/source identities")


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class EntropyControlSpec:
    horizon: int
    groups_per_batch: int
    responses_per_group: int
    mode: str
    nominal_lambda: float = 0.8
    entropy_reference: float = 0.4
    gain: float = 3.0
    lower: float = 0.6
    upper: float = 1.1
    threshold: float = 0.05
    warmup_updates: int = 2
    minimum_selected_groups: int = 16
    guard_entropy_gt: float = 0.5
    guard_minimum_lambda: float = 1.0

    def __post_init__(self):
        if self.mode not in ("observe", "direct"):
            raise ValueError("Entropy mode must be observe or direct")
        for value in (self.horizon, self.groups_per_batch, self.responses_per_group, self.minimum_selected_groups):
            if type(value) is not int or value < 1:
                raise ValueError("Entropy horizon and population sizes must be positive integers")
        if self.minimum_selected_groups > self.groups_per_batch or self.warmup_updates != 2:
            raise ValueError("Entropy coverage must fit the batch; two warmup updates are required")
        if not all(math.isfinite(value) for value in asdict(self).values() if isinstance(value, float)):
            raise ValueError("Entropy constants must be finite")
        if not (0 < self.threshold < 1 and 0 < self.lower <= self.nominal_lambda <= self.upper):
            raise ValueError("Invalid entropy probability threshold or lambda bounds")
        if not self.lower <= 1.0 <= self.upper or not self.lower <= self.guard_minimum_lambda <= self.upper:
            raise ValueError("Lambda bounds must include the observation arm and guard")


def summarize_responses(rows: list[dict], spec: EntropyControlSpec) -> dict:
    expected = spec.groups_per_batch * spec.responses_per_group
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise ValueError("Entropy feedback requires every unique original response")
    groups = Counter(row["group_index"] for row in rows)
    if len(groups) != spec.groups_per_batch or set(groups.values()) != {spec.responses_per_group}:
        raise ValueError("Entropy feedback requires complete original sibling groups")
    for row in rows:
        if type(row["original_tokens"]) is not int or row["original_tokens"] <= 0 or not math.isfinite(row["entropy_sum"]) or not math.isfinite(row["original_advantage"]):
            raise ValueError("Missing original valid entropy tokens or nonfinite actual advantage")
    for group in groups:
        labels = {(row["theme"], row["scenario"]) for row in rows if row["group_index"] == group}
        if len(labels) != 1:
            raise ValueError("Entropy sibling theme/scenario identities differ")
    eligible = {row["group_index"] for row in rows if row["original_advantage"] != 0}
    # Eligibility precedes TIS/probability/lambda. Include every eligible group's sibling, even A==0.
    selected = [row for row in rows if row["group_index"] in eligible]
    return {
        "all_response_entropy": math.fsum(row["entropy_sum"] / row["original_tokens"] / expected for row in rows),
        "selected_group_entropy": (math.fsum(row["entropy_sum"] / row["original_tokens"] / len(selected) for row in selected) if selected else None),
        "selected_group_count": len(eligible),
        "coverage_valid": len(eligible) >= spec.minimum_selected_groups,
    }


def merge_response_fragments(fragments: list[dict], *, cp_size: int) -> list[dict]:
    """Sum disjoint CP token fragments before response means; reject duplicate DP ownership."""
    by_sample = {}
    identity_fields = ("sample_id", "group_index", "theme", "scenario", "original_tokens", "prompt_loss_scale", "advantage_representation")
    for fragment in fragments:
        identity = fragment["sample_id"]
        by_sample.setdefault(identity, []).append(fragment)
    responses = []
    for _identity, pieces in sorted(by_sample.items()):
        if len(pieces) != cp_size or {piece["cp_rank"] for piece in pieces} != set(range(cp_size)):
            raise ValueError("Missing/duplicated entropy CP fragment or duplicated DP sample")
        base = pieces[0]
        if any(any(piece[key] != base[key] for key in identity_fields) for piece in pieces):
            raise ValueError("Entropy CP sample identity, original mask or normalization differs")
        count = sum(piece["local_tokens"] for piece in pieces)
        if count != base["original_tokens"]:
            raise ValueError("Entropy CP token coverage differs from the original response mask")
        advantages = [piece["original_advantage"] for piece in pieces if piece["local_tokens"]]
        if not advantages or len(set(advantages)) != 1:
            raise ValueError("Entropy requires one finite native GRPO advantage per original response")
        additive = {key: math.fsum(piece["statistics"][key] for piece in pieces) for key in base["statistics"]}
        responses.append({**{key: base[key] for key in identity_fields}, "original_advantage": advantages[0], **additive})
    return responses


def validate_summary(summary: dict, spec: EntropyControlSpec) -> None:
    count, entropy = summary["selected_group_count"], summary["selected_group_entropy"]
    if type(count) is not int or not 0 <= count <= spec.groups_per_batch or summary["coverage_valid"] != (count >= spec.minimum_selected_groups) or ((entropy is None) != (count == 0)) or (entropy is not None and not math.isfinite(entropy)) or not math.isfinite(summary["all_response_entropy"]):
        raise ValueError("Invalid entropy feedback coverage/history")


def make_packet(*, run: str, source: str, spec: EntropyControlSpec, for_update: int, scale: float) -> dict:
    if type(for_update) is not int or not 1 <= for_update <= spec.horizon or isinstance(scale, bool) or not math.isfinite(scale) or not spec.lower <= scale <= spec.upper:
        raise ValueError("Entropy packet update or scale exceeds selected bounds")
    body = dict(protocol="entropy_low_negative_pg_v1", run=run, source=source, spec=asdict(spec), for_update=for_update, scale=scale, applied_scale_fp32=struct.unpack("f", struct.pack("f", scale))[0])
    return {**body, "sha256": digest(body)}


def decide(history: tuple[dict, ...], *, completed: int, used_scale: float, spec: EntropyControlSpec) -> dict:
    """FP64 mapping, independent current-batch guard, no older feedback backfill."""
    available = len(history) == 2 and all(row["selected_group_entropy"] is not None for row in history)
    smoothed = math.fsum(row["selected_group_entropy"] / 2 for row in history) if available else None
    coverage = len(history) == 2 and all(row["coverage_valid"] for row in history)
    formula = math.fsum((spec.nominal_lambda, spec.gain * (smoothed - spec.entropy_reference))) if available else None
    mapped = min(spec.upper, max(spec.lower, formula)) if formula is not None else None
    ordinary, reason = used_scale, "warmup_hold"
    if completed >= spec.warmup_updates:
        if coverage:
            ordinary, reason = mapped, "direct_entropy_mapping"
        else:
            reason = "insufficient_feedback_groups"
    guard = completed >= spec.warmup_updates and history[-1]["all_response_entropy"] > spec.guard_entropy_gt
    guarded = max(ordinary, used_scale, spec.guard_minimum_lambda) if guard else ordinary
    final = min(spec.upper, max(spec.lower, guarded))
    if spec.mode == "observe":
        final, reason = 1.0, "observe_only"
    return dict(
        rolling2_entropy=smoothed,
        coverage_valid=coverage,
        formula_result=formula,
        formula_clamped=mapped,
        ordinary_result=ordinary,
        guarded_result=guarded,
        final_scale=final,
        reason=reason,
        guard=guard,
        lower_saturated=formula is not None and formula < spec.lower,
        upper_saturated=formula is not None and formula > spec.upper,
    )


@dataclass
class EntropyController:
    run: str
    source: str
    spec: EntropyControlSpec
    completed: int = 0
    scale: float | None = None
    history: tuple[dict, ...] = ()
    last_event_sha256: str | None = None
    last_used_packet: dict | None = None
    pending: dict | None = field(default=None, repr=False)
    continuation: dict | None = field(default=None, repr=False)

    def __post_init__(self):
        if not self.run or len(self.source) != 40 or any(c not in "0123456789abcdef" for c in self.source):
            raise ValueError("Entropy controller requires immutable run/source identities")
        if self.scale is None:
            self.scale = 1.0 if self.spec.mode == "observe" else self.spec.nominal_lambda

    @property
    def snapshot(self) -> dict | None:
        if self.completed == self.spec.horizon:
            return None
        return make_packet(run=self.run, source=self.source, spec=self.spec, for_update=self.completed + 1, scale=self.scale)

    def observe(self, observation: dict) -> None:
        """Stage only after every training rank reports a successful optimizer step."""
        used = self.snapshot
        if used is None or self.pending is not None:
            raise ValueError("Terminal or unresolved entropy update cannot accept another observation")
        expected = dict(completed_update=used["for_update"], used_sha256=used["sha256"], run=self.run, source=self.source)
        if any(observation.get(key) != value for key, value in expected.items()):
            raise ValueError("Mixed entropy update, source, run or used-packet identity")
        summary = summarize_responses(observation["responses"], self.spec)
        self.pending = dict(used=deepcopy(used), summary=summary, observation_sha256=digest(observation), observation=deepcopy(observation))

    def commit_publication(self, *, completed_update: int, published_weight_version: int | None = None) -> dict:
        if self.pending is None or completed_update != self.completed + 1:
            raise ValueError("Entropy commit requires the matching pending trained publication")
        used = self.snapshot
        if self.pending["used"] != used:
            raise ValueError("Pending entropy packet differs from current controller state")
        history = (*self.history[-1:], {"completed_update": completed_update, **self.pending["summary"]})
        result = decide(history, completed=completed_update, used_scale=used["scale"], spec=self.spec)
        self.completed, self.scale, self.history = completed_update, result["final_scale"], history
        event = dict(
            protocol="entropy_low_negative_pg_v1",
            completed_update=completed_update,
            used=used,
            observation_sha256=self.pending["observation_sha256"],
            latest_two_batches=list(history),
            measurement="pre_optimizer_training_forward_original_valid_tokens",
            published_weight_version=published_weight_version,
            **result,
            closed=self.completed == self.spec.horizon,
            next=self.snapshot,
        )
        self.last_used_packet, self.last_event_sha256 = deepcopy(used), digest(event)
        self.pending = None
        return event

    def state_dict(self) -> dict:
        # Native saves precede publication. Preserve pending as well as committed feedback.
        state = dict(protocol="entropy_low_negative_pg_v1", run=self.run, source=self.source, spec=asdict(self.spec), completed=self.completed, scale=self.scale, history=list(self.history), last_event_sha256=self.last_event_sha256, last_used_packet=self.last_used_packet, pending=self.pending, next=self.snapshot)
        if self.continuation is not None:
            state["continuation"] = self.continuation
        return deepcopy(state)

    def continued(self, *, horizon: int, source: str, authorization_sha256: str) -> "EntropyController":
        """Start a new finite segment without rewriting the sealed terminal prefix."""
        if self.pending is not None or self.completed != self.spec.horizon or horizon <= self.spec.horizon:
            raise ValueError("Entropy continuation requires a published terminal prefix and a larger horizon")
        origin = self.state_dict()
        continued = EntropyController(
            run=self.run, source=source, spec=replace(self.spec, horizon=horizon),
            completed=self.completed, scale=self.scale, history=deepcopy(self.history),
            last_event_sha256=self.last_event_sha256, last_used_packet=deepcopy(self.last_used_packet),
            continuation=dict(origin=origin, origin_sha256=digest(origin), authorization_sha256=authorization_sha256),
        )
        continued._validate_restored_state()
        return continued

    @classmethod
    def restore(cls, state: dict) -> "EntropyController":
        if state.get("protocol") != "entropy_low_negative_pg_v1":
            raise ValueError("Unknown entropy state protocol")
        restored = cls(
            run=state["run"],
            source=state["source"],
            spec=EntropyControlSpec(**state["spec"]),
            completed=state["completed"],
            scale=state["scale"],
            history=tuple(deepcopy(state["history"])),
            last_event_sha256=state["last_event_sha256"],
            last_used_packet=deepcopy(state["last_used_packet"]),
            pending=deepcopy(state["pending"]),
            continuation=deepcopy(state.get("continuation")),
        )
        restored._validate_restored_state()
        if restored.state_dict() != state:
            raise ValueError("Entropy saved state or next-packet hash differs")
        return restored

    def _validate_restored_state(self) -> None:
        origin = self._continuation_origin()
        if type(self.completed) is not int or not 0 <= self.completed <= self.spec.horizon:
            raise ValueError("Entropy cursor exceeds the finite horizon")
        if len(self.history) != min(2, self.completed):
            raise ValueError("Missing entropy history")
        for index, row in enumerate(self.history):
            if row["completed_update"] != self.completed - len(self.history) + index + 1:
                raise ValueError("Entropy history is not consecutive")
            validate_summary(row, self.spec)
        initial = 1.0 if self.spec.mode == "observe" else self.spec.nominal_lambda
        if not self.completed:
            if self.scale != initial or self.last_used_packet is not None or self.last_event_sha256 is not None:
                raise ValueError("Fresh entropy state must not inherit history")
        else:
            used = self.last_used_packet
            # Only the boundary's historical packet belongs to the sealed previous segment.
            owner = origin if origin is not None and self.completed == origin.completed else self
            owner._validate_packet(used, for_update=self.completed)
            expected = decide(self.history, completed=self.completed, used_scale=used["scale"], spec=self.spec)
            if self.scale != expected["final_scale"] or not isinstance(self.last_event_sha256, str) or len(self.last_event_sha256) != 64:
                raise ValueError("Saved entropy used/next decision differs")
        if self.pending is not None:
            if self.pending["used"] != self.snapshot or self.snapshot is None:
                raise ValueError("Pending entropy cursor or packet differs")
            validate_summary(self.pending["summary"], self.spec)
            observation = self.pending["observation"]
            if digest(observation) != self.pending["observation_sha256"] or summarize_responses(observation["responses"], self.spec) != self.pending["summary"]:
                raise ValueError("Pending original entropy statistics or summary seal differs")

    def _continuation_origin(self) -> "EntropyController | None":
        if self.continuation is None:
            return None
        if not isinstance(self.continuation, dict) or set(self.continuation) != {"origin", "origin_sha256", "authorization_sha256"}:
            raise ValueError("Invalid entropy continuation provenance")
        if digest(self.continuation["origin"]) != self.continuation["origin_sha256"]:
            raise ValueError("Entropy continuation origin seal differs")
        authorization = self.continuation["authorization_sha256"]
        if not isinstance(authorization, str) or len(authorization) != 64 or any(c not in "0123456789abcdef" for c in authorization):
            raise ValueError("Entropy continuation lacks its authorization seal")
        origin = EntropyController.restore(self.continuation["origin"])
        if (
            origin.pending is not None or origin.completed != origin.spec.horizon
            or self.completed < origin.completed or self.spec.horizon <= origin.spec.horizon
            or self.run != origin.run or replace(self.spec, horizon=origin.spec.horizon) != origin.spec
        ):
            raise ValueError("Entropy continuation changed the prefix, population or controller constants")
        if self.completed == origin.completed and any(
            getattr(self, key) != getattr(origin, key)
            for key in ("scale", "history", "last_used_packet", "last_event_sha256")
        ):
            raise ValueError("Entropy continuation changed terminal feedback or its historical packet")
        return origin

    def _validate_packet(self, value: dict, *, for_update: int) -> None:
        if not isinstance(value, dict):
            raise ValueError("Missing entropy packet")
        expected = make_packet(run=self.run, source=self.source, spec=self.spec, for_update=for_update, scale=value["scale"])
        if value != expected:
            raise ValueError("Entropy packet identity, constants, scale or hash differs")
