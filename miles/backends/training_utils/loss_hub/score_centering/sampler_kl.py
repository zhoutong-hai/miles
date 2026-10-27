"""Sampler-to-trainer KL with the transported head and a modeled sampler tail.

The sampler is fixed data. Outside the head, q is approximated as rho*p,
using the same proportional-tail assumption as native score centering. The
surrogate's gradient treats that reconstructed q as fixed, not as a function
of the current trainer. No importance ratio weights the reward gradient.
"""

import torch


def sampler_kl_loss(
    train_head_log_probs: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    head_mask: torch.Tensor,
    active: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return per-token KL values with gradient KL(q_hat || p).

    Head tensors have shape [tokens, candidates], active has shape [tokens].
    Full-head coverage is exact. For a partial head, the result is exact only
    for the modeled tail, not for an arbitrary omitted sampler distribution.
    The forward value is the head-plus-tail KL estimate; the backward pass is
    the fixed-q gradient, including when reward advantages are zero.
    """
    if train_head_log_probs.ndim != 2 or rollout_head_log_probs.shape != train_head_log_probs.shape or head_mask.shape != train_head_log_probs.shape or active.shape != train_head_log_probs.shape[:1]:
        raise ValueError("Sampler KL requires [tokens, candidates] heads and a [tokens] active mask")
    valid = head_mask.bool() & active.bool().unsqueeze(-1)
    train = torch.where(valid, train_head_log_probs, 0.0)
    with torch.no_grad():
        rollout = torch.where(valid, rollout_head_log_probs.detach(), 0.0)
        p = train.detach().exp().masked_fill(~valid, 0.0)
        q = rollout.exp().masked_fill(~valid, 0.0)
        p_tail = (1 - p.sum(-1)).clamp_min(eps)
        q_tail = (1 - q.sum(-1)).clamp_min(0.0)
        rho = q_tail / p_tail
        residual = q - rho.unsqueeze(-1) * p
        logq = torch.where(q > 0, rollout, 0.0)
        logp = torch.where(q > 0, train.detach(), 0.0)
        tail_log_ratio = torch.where(q_tail > 0, rho.clamp_min(torch.finfo(rho.dtype).tiny).log(), 0.0)
        value = (q * (logq - logp)).sum(-1) + q_tail * tail_log_ratio
    # This normalized-logprob surrogate includes the dense normalization
    # gradient without gathering vocabulary-wide logits across TP ranks.
    scored = torch.where(residual != 0, train, 0.0)
    surrogate = -(residual * scored).sum(-1)
    result = surrogate + (value - surrogate).detach()
    return torch.where(active.bool(), result, 0.0)
