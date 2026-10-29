from argparse import Namespace
import math
from typing import Any

import torch


def mis_intervals(args: Namespace) -> tuple[float, float, float, float]:
    """Resolve fixed sign-specific bounds, preserving legacy MIS by default."""
    values = tuple(getattr(args, name, None) for name in (
        "mis_positive_low", "mis_positive_high", "mis_negative_low", "mis_negative_high"))
    if all(value is None for value in values):
        return (args.tis_clip_low, args.tis_clip) * 2
    if any(value is None for value in values):
        raise ValueError("Specify all four fixed MIS advantage bounds together")
    if not all(math.isfinite(value) for value in values) or not (
        0 < values[0] <= values[1] and 0 < values[2] <= values[3]
    ):
        raise ValueError("MIS advantage bounds must be finite positive ordered intervals")
    return values


def mis_weights(args: Namespace, ratio: torch.Tensor, advantages: torch.Tensor | None = None) -> torch.Tensor:
    """Mask a detached ratio; zero advantages use the negative interval for diagnostics."""
    positive_low, positive_high, negative_low, negative_high = mis_intervals(args)
    if advantages is None:
        if (positive_low, positive_high) != (negative_low, negative_high):
            raise ValueError("Asymmetric MIS requires aligned token advantages")
        keep = (ratio >= negative_low) & (ratio <= negative_high)
    else:
        positive = advantages.detach() > 0
        keep = torch.where(positive,
                           (ratio >= positive_low) & (ratio <= positive_high),
                           (ratio >= negative_low) & (ratio <= negative_high))
    return torch.where(keep, ratio.detach(), torch.zeros_like(ratio))


def vanilla_tis_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """Truncated importance sampling: clamp `exp(train - rollout)` to
    `[tis_clip_low, tis_clip]` and multiply into `pg_loss`. `loss_masks` is
    passed through unchanged; metrics report the pre-clamp ratio.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args: Namespace,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """IS clip-or-pop: zero out tokens whose `exp(train - rollout)` is outside
    `[tis_clip_low, tis_clip]` and pass the in-range ratio through unchanged.
    Same return shape as `vanilla_tis_function`.
    """
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = mis_weights(args, ice_ratio, kwargs.get("advantages"))
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics
