"""Importance sampling for score centering (arXiv:2609.20807, Appendix A.2/A.3).

Each scheme applies an importance-sampling weight function f to the ratio
r = p / q. It supplies the terms of Eq. 12, whose head coefficient is
q * f(p / q) - alpha * p with tail scale alpha = rho * f(1 / rho), and the
sampled-token weight of the loss:

- ``head_mass(p, q, log_ratio)``: q * f(p / q) on the head candidates;
- ``tail_scale(rho, dtype)``: alpha = rho * f(1 / rho), where rho is the
  sampler-to-trainer tail-mass ratio;
- ``sample_weight(log_ratio)``: f(r) for the sampled token.

No term exponentiates an unbounded ratio.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoImportanceSampling:
    """Vanilla score centering: f(r) = 1."""

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        return q

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return rho

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(log_ratio)


@dataclass(frozen=True)
class TruncatedImportanceSampling:
    """Truncated importance sampling (TIS): f(r) = min(r, clip)."""

    clip: float

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        # q * min(p/q, c), including q=0, without 0 * inf.
        return torch.minimum(p, self.clip * q)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (self.clip * rho).clamp_max(1)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return log_ratio.clamp(max=math.log(self.clip)).exp()


@dataclass(frozen=True)
class MaskedImportanceSampling:
    """Masked importance sampling (MIS): f(r) = r for low <= r <= high, else 0."""

    low: float
    high: float

    def _inside(self, log_ratio: torch.Tensor) -> torch.Tensor:
        """Whether low <= r <= high, tested on log(r)."""
        return (log_ratio >= math.log(self.low)) & (log_ratio <= math.log(self.high))

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        return torch.where(self._inside(log_ratio), p, 0.0)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return ((rho >= 1 / self.high) & (rho <= 1 / self.low)).to(dtype)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        return torch.where(self._inside(log_ratio), log_ratio.clamp(max=math.log(self.high)).exp(), 0.0)


@dataclass(frozen=True)
class AdvantageMaskedImportanceSampling:
    """The same sign-specific MIS function in sampled, head and tail terms."""

    intervals: tuple[float, float, float, float]
    advantages: torch.Tensor

    def _bounds(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive = self.advantages.detach() > 0
        low = torch.where(positive, self.intervals[0], self.intervals[2]).to(value)
        high = torch.where(positive, self.intervals[1], self.intervals[3]).to(value)
        while low.ndim < value.ndim:
            low, high = low.unsqueeze(-1), high.unsqueeze(-1)
        return low, high

    def _ratio_and_mask(self, log_ratio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inactive = self.advantages.detach() == 0
        while inactive.ndim < log_ratio.ndim:
            inactive = inactive.unsqueeze(-1)
        log_ratio = torch.where(inactive & torch.isnan(log_ratio), 0.0, log_ratio)
        if torch.isnan(log_ratio).any():
            raise ValueError("Score-centering importance log-ratio contains NaN")
        low, high = self._bounds(log_ratio)
        # Compare the actual FP32/FP64 exp ratio inclusively, including boundary
        # rounding. Capping only above the largest accepted value avoids overflow.
        ratio = log_ratio.clamp_max(high.log() + 1).exp()
        return ratio, (ratio >= low) & (ratio <= high)

    def head_mass(self, p: torch.Tensor, q: torch.Tensor, log_ratio: torch.Tensor) -> torch.Tensor:
        _, keep = self._ratio_and_mask(log_ratio)
        return torch.where(keep, p, 0.0)

    def tail_scale(self, rho: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        low, high = self._bounds(rho)
        return ((rho >= 1 / high) & (rho <= 1 / low)).to(dtype)

    def sample_weight(self, log_ratio: torch.Tensor) -> torch.Tensor:
        ratio, keep = self._ratio_and_mask(log_ratio)
        return torch.where(keep, ratio, 0.0)


def importance_sampling(
    mode: str, *, tis_clip: float, mis_low: float, mis_high: float,
    mis_intervals: tuple[float, float, float, float] | None = None, advantages: torch.Tensor | None = None,
) -> NoImportanceSampling | TruncatedImportanceSampling | MaskedImportanceSampling | AdvantageMaskedImportanceSampling:
    """Validate the parameters of ``mode`` and return its importance sampling."""
    if mode == "tis" and (not math.isfinite(tis_clip) or tis_clip <= 0):
        raise ValueError("Score-centering TIS clip must be positive and finite")
    if mode == "mis" and (not math.isfinite(mis_low) or not math.isfinite(mis_high) or not 0 < mis_low <= mis_high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    if mode == "none":
        return NoImportanceSampling()
    if mode == "tis":
        return TruncatedImportanceSampling(clip=tis_clip)
    if mis_intervals is not None:
        if (mode != "mis" or len(mis_intervals) != 4 or advantages is None
            or not all(math.isfinite(x) for x in mis_intervals)
            or not (0 < mis_intervals[0] <= mis_intervals[1] and 0 < mis_intervals[2] <= mis_intervals[3])):
            raise ValueError("Sign-specific SFC MIS needs four finite ordered bounds and advantages")
        return AdvantageMaskedImportanceSampling(intervals=mis_intervals, advantages=advantages)
    if mode == "mis":
        return MaskedImportanceSampling(low=mis_low, high=mis_high)
    raise ValueError(f"Unknown score-centering importance weighting: {mode}")


def importance_weights(
    log_ratio: torch.Tensor,
    mode: str,
    *,
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
) -> torch.Tensor:
    """Evaluate token-level weights without exponentiating unbounded ratios."""
    sampling = importance_sampling(mode, tis_clip=tis_clip, mis_low=mis_low, mis_high=mis_high)
    return sampling.sample_weight(log_ratio)
