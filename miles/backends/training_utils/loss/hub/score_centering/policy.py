"""Center the effective token PPO + detached old-policy TIS score.

This is an additive gradient correction, not an unchanged PPO surrogate.
For sign-dependent active branch g, the effective mass is
q * min(p_old / q, c) * (p / p_old) * g. Both old-policy and sampler tails
are modeled as proportional to the current trainer tail. All masses detach.
"""

import math
from dataclasses import dataclass
from functools import cache

import torch

from miles.backends.training_utils.loss.hub.policy_ratio import LOG_RATIO_EXP_CLAMP
from miles.backends.training_utils.loss.hub.score_centering.masks import (
    head_probs,
    sanitize_head_log_probs,
    scored_log_probs,
)


@dataclass(frozen=True)
class PolicyScoreCenteringInputs:
    train_head_log_probs: torch.Tensor
    old_head_log_probs: torch.Tensor
    rollout_head_log_probs: torch.Tensor
    head_mask: torch.Tensor
    advantages: torch.Tensor
    eps_clip: float = 0.2
    eps_clip_high: float = 0.2
    tis_clip: float = 2.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        shape = self.train_head_log_probs.shape
        if (
            len(shape) != 2
            or self.old_head_log_probs.shape != shape
            or self.rollout_head_log_probs.shape != shape
            or self.head_mask.shape != shape
            or self.advantages.shape != shape[:1]
        ):
            raise ValueError("Policy centering requires [tokens, candidates] heads and [tokens] advantages")
        if not (
            math.isfinite(self.eps_clip)
            and 0 <= self.eps_clip < 1
            and math.isfinite(self.eps_clip_high)
            and self.eps_clip_high >= 0
            and math.isfinite(self.tis_clip)
            and self.tis_clip > 0
            and math.isfinite(self.eps)
            and 0 < self.eps < 1
        ):
            raise ValueError("Invalid policy-centering clipping or tail bounds")


@cache
def _clamp_boundary_slope(dtype: torch.dtype) -> float:
    # PyTorch versions differ at clamp's exact endpoints. Probe that stable
    # scalar convention once, so the correction follows native PPO's autograd.
    with torch.enable_grad():
        point = torch.tensor(0.0, dtype=dtype, requires_grad=True)
        return torch.autograd.grad(point.clamp(0.0, 1.0), point)[0].item()


def _active_policy_branch(log_ratio: torch.Tensor, positive: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Derivative multiplier of the native clipped surrogate, including ties."""
    finite = torch.isfinite(log_ratio)
    ratio = log_ratio.clamp(-LOG_RATIO_EXP_CLAMP, LOG_RATIO_EXP_CLAMP).exp()
    multiplier = torch.where(positive, ratio <= 1 + high, ratio >= 1 - low).to(ratio.dtype)
    boundary_slope = _clamp_boundary_slope(log_ratio.dtype)
    # At a clipping tie, maximum averages both branches' derivatives.
    tie = (ratio == 1 - low) | (ratio == 1 + high)
    multiplier = torch.where(tie, (1 + boundary_slope) / 2, multiplier)
    exponent_slope = torch.where(log_ratio.abs() == LOG_RATIO_EXP_CLAMP, boundary_slope, 1.0)
    return torch.where(finite & (log_ratio.abs() <= LOG_RATIO_EXP_CLAMP), multiplier * exponent_slope, 0.0)


def policy_score_centering_correction(
    inputs: PolicyScoreCenteringInputs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the additive per-token loss and detached diagnostics.

    Native PPO+TIS owns the sampled loss. Add this term *after* sampled TIS,
    before the common reducer; do not multiply it by sampled TIS a second time.
    Exact head coverage gives exact centering. A partial head approximates two
    tails, and cannot promise exact cancellation for arbitrary omitted tokens.
    """
    active = inputs.advantages.detach() != 0
    if not torch.isfinite(inputs.advantages).all():
        raise ValueError("Policy-centering advantages must be finite")
    mask = inputs.head_mask
    train = sanitize_head_log_probs(inputs.train_head_log_probs, mask, active)
    old = sanitize_head_log_probs(inputs.old_head_log_probs.detach(), mask, active)
    rollout = sanitize_head_log_probs(inputs.rollout_head_log_probs.detach(), mask, active)
    with torch.no_grad():
        if any(torch.isnan(value[mask]).any() for value in (train, old, rollout)):
            raise ValueError("Active policy-centering candidate log-probability contains NaN")
        p, p_old, q = (head_probs(value, mask) for value in (train, old, rollout))
        p_mass, old_mass, q_mass = (value.sum(-1) for value in (p, p_old, q))
        if any(not torch.isfinite(mass).all() or (mass > 1.0001).any() for mass in (p_mass, old_mass, q_mass)):
            raise ValueError("Policy-centering candidate probability mass must be finite and at most one")
        p_tail, old_tail, q_tail = ((1 - mass).clamp_min(inputs.eps) for mass in (p_mass, old_mass, q_mass))
        positive = inputs.advantages.detach() >= 0
        ratio = train - old
        gate = mask * _active_policy_branch(ratio, positive[:, None], inputs.eps_clip, inputs.eps_clip_high)
        # q*TIS = min(p_old, c*q); log space avoids 0*inf at zero-probability candidates.
        log_mass = torch.minimum(old, rollout + math.log(inputs.tis_clip)) + torch.where(gate != 0, ratio, 0.0)
        mass = torch.where(gate != 0, log_mass.exp() * gate, 0.0)
        tail_ratio = p_tail.log() - old_tail.log()
        tail_gate = _active_policy_branch(tail_ratio, positive, inputs.eps_clip, inputs.eps_clip_high)
        alpha = tail_gate * (inputs.tis_clip * q_tail / old_tail).clamp_max(1)
        residual = mass - alpha[:, None] * p
        if not torch.isfinite(residual).all():
            raise ValueError("Non-finite policy-centering effective mass")
    correction = (residual * scored_log_probs(train, residual)).sum(-1)
    loss = inputs.advantages.detach() * correction
    metrics = {
        "sc_correction": correction.detach(),
        "sc_correction_abs": correction.detach().abs(),
        "sc_train_head_mass": p_mass,
        "sc_old_head_mass": old_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_policy_ratio": p_tail / old_tail,
        "sc_tail_sampler_ratio": q_tail / p_tail,
        "sc_effective_head_mass": mass.sum(-1),
    }
    return loss, metrics
