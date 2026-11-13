"""Original-token prompt normalization and detached fixed-MIS REINFORCE arithmetic."""

import math
from collections import Counter, defaultdict

import torch


def prompt_factors(group_ids, masks, *, groups, siblings):
    """Derive N/(G*N_g) before DP partitioning; never use the post-MIS population."""
    if len(group_ids) != len(masks) or len(group_ids) != groups * siblings:
        raise ValueError("Prompt normalization needs the complete optimizer batch")
    if any(type(g) is not int or g < 0 for g in group_ids):
        raise ValueError("Missing original prompt presentation identities")
    counts = Counter(group_ids)
    if len(counts) != groups or set(counts.values()) != {siblings}:
        raise ValueError("Prompt normalization requires every original sibling")
    totals = defaultdict(int)
    for group, mask in zip(group_ids, masks, strict=True):
        if not mask or any(x not in (0, 1, False, True) for x in mask) or not sum(mask):
            raise ValueError("REINFORCE requires nonempty original binary action masks")
        totals[group] += sum(mask)
    total = sum(totals.values())
    return [total / (groups * totals[g]) for g in group_ids], [totals[g] for g in group_ids]


def fixed_mis_loss(logp, original_q, advantage, mask, factors, bounds):
    """Return the unreduced loss and actual weights; padding is sanitized before exp."""
    if not (logp.shape == original_q.shape == advantage.shape == mask.shape == factors.shape):
        raise ValueError("Unaligned original-token REINFORCE inputs")
    if any(not math.isfinite(x) for x in bounds) or not (0 < bounds[0] <= bounds[1] and 0 < bounds[2] <= bounds[3]):
        raise ValueError("Invalid inclusive fixed-MIS intervals")
    valid = mask.bool()
    for value in (logp, original_q, advantage, factors):
        if not torch.isfinite(value[valid]).all():
            raise ValueError("Nonfinite original valid-token input")
    if (factors[valid] <= 0).any():
        raise ValueError("Invalid original prompt normalizer")
    p, q, a = logp.detach().float(), original_q.detach().float(), advantage.detach().float()
    ratio = torch.where(valid, p - q, 0.0).exp()
    if not torch.isfinite(ratio[valid]).all() or not (ratio[valid] > 0).all():
        raise ValueError("Invalid original valid-token ratio; no numeric clamp is allowed")
    keep = torch.where(a > 0, (ratio >= bounds[0]) & (ratio <= bounds[1]), (ratio >= bounds[2]) & (ratio <= bounds[3]))
    weight = torch.where(valid & keep, ratio, 0.0).detach()
    safe_logp = torch.where(valid, logp, 0.0)
    safe_advantage = torch.where(valid, a, 0.0)
    safe_factor = torch.where(valid, factors.detach(), 0.0)
    loss = -safe_logp * safe_advantage * weight * safe_factor
    return loss, weight, torch.where(valid, ratio, 0.0)
