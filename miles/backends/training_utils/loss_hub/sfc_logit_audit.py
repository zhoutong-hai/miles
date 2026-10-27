"""Detached output-logit component gradients, not model-parameter gradients."""

import torch
import torch.distributed as dist


def component_gradients(logits, head_ids, head_logq, sampled_ids, sampled_logq, advantages,
                        *, mode, beta, temperature=1.0, vocab_size=None, group=None, denominator=1):
    """Closed forms of the native detached-head SFC/KL derivatives on a TP shard.

    This is a diagnostic calculation on detached copies. It neither calls
    backward on the model nor changes the configured objective.
    """
    rank = dist.get_rank(group) if group is not None else 0
    size = dist.get_world_size(group) if group is not None else 1
    width = logits.size(-1)
    vocabulary = vocab_size or width * size
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    work = logits.detach().to(dtype) / temperature
    padded = torch.arange(width, device=work.device) + rank * width >= vocabulary
    work = work.masked_fill(padded, -torch.inf)
    maximum = work.amax(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    exponential = (work - maximum).exp()
    normalizer = exponential.sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(normalizer, group=group)
    p = exponential / normalizer
    logp = work - maximum - normalizer.log()
    safe_logp = torch.where(p > 0, logp, 0.0)
    entropy = -(p * safe_logp).sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(entropy, group=group)
    valid = head_ids >= 0
    local_ids = head_ids - rank * width
    owned = valid & (local_ids >= 0) & (local_ids < width)
    head_p = torch.where(owned, p.gather(-1, local_ids.clamp(0, width-1)), 0.0)
    if group is not None:
        dist.all_reduce(head_p, group=group)
    q = head_logq.detach().to(dtype).exp().masked_fill(~valid, 0.0)
    rho = (1-q.sum(-1)).clamp_min(1e-6) / (1-head_p.sum(-1)).clamp_min(1e-6)
    sampled_local = sampled_ids - rank * width
    sample_owned = (sampled_local >= 0) & (sampled_local < width)
    sample_logp = torch.where(sample_owned, logp.gather(-1, sampled_local.clamp(0, width-1)[:, None]).squeeze(-1), 0.0)
    if group is not None:
        dist.all_reduce(sample_logp, group=group)
    weight = (sample_logp - sampled_logq.detach()).clamp_max(torch.log(work.new_tensor(2.0))).exp()
    score = -p.clone()
    score.scatter_add_(-1, sampled_local.clamp(0, width-1)[:, None], sample_owned.to(dtype)[:, None])

    def residual_score(residual):
        sparse = torch.zeros_like(p)
        sparse.scatter_add_(-1, local_ids.clamp(0, width-1), residual.masked_fill(~owned, 0.0))
        return sparse - p * residual.sum(-1, keepdim=True)

    advantage = advantages.detach().to(dtype)[:, None]
    none_residual = q - rho[:, None] * head_p
    tis_residual = torch.minimum(head_p, 2*q) - (2*rho).clamp_max(1)[:, None] * head_p
    unweighted = -advantage * (score - residual_score(none_residual))
    weighted = -advantage * (weight[:, None]*score - residual_score(tis_residual))
    selected = weighted if mode == "tis" else unweighted
    uncentered = -advantage * (weight[:, None] if mode == "tis" else 1) * score
    # The KL tail uses q_tail>=0, whereas native SFC uses its Appendix floor.
    kl_rho = (1-q.sum(-1)).clamp_min(0) / (1-head_p.sum(-1)).clamp_min(1e-6)
    kl_gradient = -beta * residual_score(q-kl_rho[:, None]*head_p)
    scale = temperature * denominator
    return {
        "unweighted_sfc_pg": unweighted / scale,
        "importance_change": (selected-unweighted) / scale,
        "weighted_sampler_kl": kl_gradient / scale,
        "sfc_change_from_uncentered": (selected-uncentered) / scale,
        "entropy": -p*(safe_logp+entropy) / temperature,
    }


def component_statistics(components, *, group=None):
    """Combine TP shard dot products; keep individual audited positions."""
    names = list(components)
    dots = torch.stack([torch.stack([(components[a]*components[b]).sum(-1) for b in names], -1) for a in names], -2)
    if group is not None:
        dist.all_reduce(dots, group=group)
    norms = dots.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    denominator = norms.unsqueeze(-1) * norms.unsqueeze(-2)
    cosines = torch.where(denominator > 0, dots/denominator.clamp_min(torch.finfo(dots.dtype).tiny), 0.0)
    return {"components": names, "norms": norms.cpu(), "dot_products": dots.cpu(), "cosines": cosines.cpu(),
            "cosine_defined": (denominator > 0).cpu(),
            "component_to_pg_norm_ratio": torch.where(norms[..., :1] > 0, norms[..., :-1]/norms[..., :1].clamp_min(torch.finfo(dots.dtype).tiny), 0.0).cpu(),
            "pg_norm_nonzero": (norms[..., 0] > 0).cpu(),
            "entropy_change_per_unit_logit_gradient_step": -dots[..., :-1, -1].cpu(),
            "scope": "partial output-logit gradients at these positions; not parameter gradients"}
