"""Detached observations for the native policy-loss MIS screen."""

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import all_gather_with_cp
from miles.backends.training_utils.loss_hub import sfc_observations
from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from miles.backends.training_utils.parallel import get_parallel_state


@torch.no_grad()
def component_gradients(logits, sampled_ids, logq, old_logp, advantages, *, args, denominator):
    """Exact native PPO derivative on detached sampled scores; no model backward.

    Only the small sampled-score leaves use autograd, preserving native clipping
    and tie derivatives without differentiating an extra vocabulary/model graph.
    """
    parallel = get_parallel_state()
    group = parallel.tp.group if parallel.tp.size > 1 else None
    width = logits.size(-1)
    rank = parallel.tp.rank
    work = logits.detach().float()
    ids = torch.arange(width, device=work.device) + rank * width
    work = work.masked_fill(ids >= args.vocab_size, -torch.inf)
    maximum = work.amax(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    exponential = (work-maximum).exp()
    normalizer = exponential.sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(normalizer, group=group)
    p = exponential/normalizer
    logp = work-maximum-normalizer.log()
    safe_logp = torch.where(p > 0, logp, 0.0)
    entropy = -(p*safe_logp).sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(entropy, group=group)
    local_ids = sampled_ids-rank*width
    owned = (local_ids >= 0) & (local_ids < width)
    sampled_logp = torch.where(owned, logp.gather(-1, local_ids.clamp(0, width-1)[:, None]).squeeze(-1), 0.)
    if group is not None:
        dist.all_reduce(sampled_logp, group=group)
    score = -p.clone()
    score.scatter_add_(-1, local_ids.clamp(0, width-1)[:, None], owned.float()[:, None])
    with torch.enable_grad():
        leaf = sampled_logp.detach().requires_grad_()
        loss, _ = compute_policy_loss(old_logp.detach()-leaf, advantages.detach(),
                                      args.eps_clip, args.eps_clip_high, args.eps_clip_c)
        coefficient, = torch.autograd.grad(loss.sum(), leaf)
    ratio = (old_logp.detach()-logq.detach()).exp()
    tis = ratio.clamp(args.tis_clip_low, args.tis_clip)
    mis = torch.where((ratio >= args.tis_clip_low) & (ratio <= args.tis_clip), ratio, 0.)
    base = coefficient[:, None]*score/(args.rollout_temperature*denominator)
    return {"counterfactual_tis_pg": tis[:, None]*base,
            "mis_minus_tis": (mis-tis)[:, None]*base,
            "entropy": -p*(safe_logp+entropy)/args.rollout_temperature}


@torch.no_grad()
def observe(args, batch, logits, *, logp, trainer_logp, rollout_logp, advantages, active, entropy, token_loss):
    if not sfc_observations.enabled(args):
        return
    old = torch.cat(trainer_logp).detach()
    q = torch.cat(rollout_logp).detach()
    ratio = (old-q).exp()
    weights = torch.where((ratio >= args.tis_clip_low) & (ratio <= args.tis_clip), ratio, 0.)
    empty_ids = torch.empty((len(logp), 0), dtype=torch.long, device=logp.device)
    empty_head = logp.new_empty((len(logp), 0))
    sfc_observations.capture(args, batch, logp[:, None], q, advantages, active, empty_ids, empty_head,
                             None, entropy, token_loss, torch.zeros_like(logp),
                             applied_weight=weights, importance_logp=old, importance_ratio=ratio)
    if not sfc_observations.audit_due(args):
        return
    # Precomputed logprobs use zigzag CP ownership, whereas retained logits may
    # use contiguous CP. All CP ranks reconstruct anchors before sparse selection.
    anchors = [all_gather_with_cp(value, total, response) for value, total, response in
               zip(trainer_logp, batch["total_lengths"], batch["response_lengths"], strict=True)]
    chunks = _iter_response_chunks(logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
                                  total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
                                  max_seq_lens=batch.get("max_seq_lens"), include_response_indices=True)
    for sample, (chunk, tokens, indices) in enumerate(chunks):
        sfc_observations.logit_audit(args, batch, sample, chunk, tokens, indices,
                                    policy_old_logp=anchors[sample])
