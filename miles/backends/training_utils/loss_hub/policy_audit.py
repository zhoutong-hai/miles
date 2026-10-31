"""Detached TP-aware distribution and sparse output-logit diagnostics."""

import torch
import torch.distributed as dist


@torch.no_grad()
def distribution_statistics(logits, *, vocab_size, group=None, temperature=1.0, with_top_mass=True):
    rank = dist.get_rank(group) if group is not None else 0
    work = logits.detach().float() / temperature
    ids = torch.arange(work.size(-1), device=work.device) + rank * work.size(-1)
    work = work.masked_fill(ids >= vocab_size, -torch.inf)
    maximum = work.amax(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    exps = (work - maximum).exp()
    total = exps.sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(total, group=group)
    p = exps / total
    logp = work - maximum - total.log()
    safe = torch.where(p > 0, logp, 0.0)
    entropy = -(p * safe).sum(-1)
    if group is not None:
        dist.all_reduce(entropy, group=group)
    if not with_top_mass:
        return entropy, None, p, safe
    top = p.topk(min(128, p.size(-1)), dim=-1).values
    if group is not None:
        parts = [torch.empty_like(top) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, top, group=group)
        top = torch.cat(parts, dim=-1)
    mass = top.topk(min(128, vocab_size), dim=-1).values.sum(-1)
    return entropy, mass, p, safe


@torch.no_grad()
def component_statistics(
    logits, tokens, advantage, ratio, weight, *, vocab_size, temperature, denominator, group=None
):
    entropy, _, entropy_p, entropy_logp = distribution_statistics(
        logits, vocab_size=vocab_size, group=group, temperature=temperature
    )
    # PG keeps the native scoring normalizer, including any padded logits.
    # The separately labelled entropy uses only the true tokenizer vocabulary.
    native_width = logits.size(-1) * (dist.get_world_size(group) if group is not None else 1)
    _, _, p, _ = distribution_statistics(
        logits, vocab_size=native_width, group=group, temperature=temperature, with_top_mass=False
    )
    rank = dist.get_rank(group) if group is not None else 0
    local_ids = tokens - rank * p.size(-1)
    owns = (local_ids >= 0) & (local_ids < p.size(-1))
    score = -p.clone()
    score.scatter_add_(-1, local_ids.clamp(0, p.size(-1) - 1)[:, None], owns.float()[:, None])
    base = -advantage[:, None] * score / (temperature * denominator)
    mis = torch.where((ratio >= 0.5) & (ratio <= 2.0), ratio, 0.0)
    # Each vector is a LOSS derivative. A descent step is the negative vector.
    components = {
        "actual_reinforce_tis": weight[:, None] * base,
        "unweighted_pg": base,
        "tis_change": (weight - 1)[:, None] * base,
        "counterfactual_mis_minus_tis": (mis - weight)[:, None] * base,
    }
    entropy_derivative = -entropy_p * (entropy_logp + entropy[:, None]) / temperature
    components["unit_entropy_derivative"] = entropy_derivative
    names = list(components)
    matrix = torch.stack([components[name] for name in names], dim=1)
    norm2 = matrix.square().sum(-1)
    entropy_direction = -(matrix * entropy_derivative[:, None]).sum(-1)
    if group is not None:
        dist.all_reduce(norm2, group=group)
        dist.all_reduce(entropy_direction, group=group)
    return {
        "components": names,
        "norm2": norm2.cpu(),
        "entropy_direction": entropy_direction.cpu(),
        "scope": "Sparse loss derivatives at output logits, normalized by original global tokens; not adapter gradients",
        "entropy_regularizer_coefficient": 0.0,
    }
