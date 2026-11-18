"""Minimal original-token entropy statistics from the normal training forward.

Only host scalars survive the microbatch. No additional forward, vocabulary
dump, logits audit or score-centering collector participates.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

_capture = ContextVar("entropy_observation", default=None)


def negative_pg_multiplier(log_probs, advantages, active, *, threshold, scale):
    """Inclusive FP32 probability comparison; both the predicate and lambda are detached."""
    probabilities = log_probs.detach().float().exp()
    threshold_tensor = probabilities.new_tensor(threshold)
    eligible = active & (advantages.detach() < 0) & (probabilities <= threshold_tensor)
    return torch.where(eligible, probabilities.new_tensor(scale), probabilities.new_ones(())).detach(), eligible


@dataclass
class EntropyCapture:
    packet: dict
    cp_rank: int
    record_enabled: bool
    tis_low: float
    tis_high: float
    fragments: list[dict] = field(default_factory=list)

    def record(self, *, batch, layout, entropy, advantages, active, ratio, multiplier, eligible, log_probs, q):
        masks = layout.per_sample(active)
        per_sample = [layout.per_sample(value) for value in (entropy, advantages, ratio, multiplier, eligible, log_probs, q)]
        for index, (sample_id, metadata, mask, original, factor) in enumerate(
            zip(
                batch["sample_indices"],
                batch["entropy_metadata"],
                masks,
                batch["loss_masks"],
                batch["prompt_loss_scale"],
                strict=True,
            )
        ):
            values = [column[index][mask] for column in per_sample]
            h, advantage, ratio_i, scale_i, chosen, logp, logq = values
            if not all(torch.isfinite(value).all() for value in (h, advantage, logp, logq)):
                raise ValueError("Nonfinite original entropy feedback token")
            if advantage.numel() and not torch.equal(advantage, advantage[0].expand_as(advantage)):
                raise ValueError("Entropy feedback requires a constant actual native response advantage")
            weight = ratio_i.clamp(self.tis_low, self.tis_high)
            coefficient = advantage.detach().abs().float() * weight
            difference = logp.detach().float() - logq.detach().float()
            k3_gap = (-difference).clamp(-10, 10)
            statistics = {
                "entropy_sum": h.detach().double().sum(),
                "train_rollout_abs_diff_sum": difference.abs().double().sum(),
                "train_rollout_k3_sum": (k3_gap.exp() - k3_gap - 1).double().sum(),
                "tis_weight_sum": weight.double().sum(),
                "tis_weight_squared_sum": weight.double().square().sum(),
                "tis_below_floor_tokens": (ratio_i < self.tis_low).sum(),
                "tis_above_cap_tokens": (ratio_i > self.tis_high).sum(),
                "low_probability_negative_tokens": chosen.sum(),
                "pg_coefficient_mass": coefficient.double().sum(),
                "applied_pg_coefficient_mass": (coefficient * scale_i).double().sum(),
            }
            for sign, condition in (("positive", advantage > 0), ("negative", advantage < 0), ("zero", advantage == 0)):
                statistics[f"{sign}_tokens"] = condition.sum()
                statistics[f"{sign}_entropy_sum"] = h.detach()[condition].double().sum()
            names = list(statistics)
            scalar_values = torch.stack([statistics[key].double() for key in names]).tolist()
            self.fragments.append(
                {
                    "sample_id": int(sample_id),
                    "cp_rank": self.cp_rank,
                    **metadata,
                    "original_tokens": int(original.sum().item()),
                    "advantage_representation": str(advantages.dtype),
                    "local_tokens": int(mask.sum().item()),
                    "prompt_loss_scale": float(factor),
                    "original_advantage": float(advantage[0].item()) if advantage.numel() else None,
                    "statistics": dict(zip(names, scalar_values, strict=True)),
                }
            )


@contextmanager
def capture_entropy(packet, *, cp_rank, record_enabled, tis_low, tis_high):
    capture = EntropyCapture(packet, cp_rank, record_enabled, tis_low, tis_high)
    token = _capture.set(capture)
    try:
        yield capture
    finally:
        _capture.reset(token)


def apply_entropy_control(pg_loss, *, batch, layout, entropy, advantages, active, log_probs, rollout_log_probs):
    capture = _capture.get()
    if capture is None:
        raise ValueError("Entropy-controlled PG requires the worker's acknowledged update packet")
    packet = capture.packet
    multiplier, eligible = negative_pg_multiplier(
        log_probs,
        advantages,
        active,
        threshold=packet["spec"]["threshold"],
        scale=packet["scale"],
    )
    if capture.record_enabled:
        ratio = torch.exp(log_probs.detach().float() - rollout_log_probs.detach().float())
        capture.record(batch=batch, layout=layout, entropy=entropy, advantages=advantages, active=active, ratio=ratio, multiplier=multiplier, eligible=eligible, log_probs=log_probs, q=rollout_log_probs)
    return pg_loss * multiplier
