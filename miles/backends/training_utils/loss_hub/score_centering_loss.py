"""Integrate score centering with response alignment and Miles loss reduction."""

from argparse import Namespace
from collections.abc import Callable
import math

import torch

from miles.backends.training_utils.cp_utils import (
    allgather_cp_redistribute,
    get_local_response_loss_masks,
    slice_log_prob_with_cp,
)
from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub import sfc_observations
from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss_hub.score_centering import (
    ScoreCenteringInputs,
    score_centering_loss,
    selected_log_probs_and_entropy,
)
from miles.backends.training_utils.loss_hub.score_centering.sampler_kl import sampler_kl_loss
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def _candidate_log_probs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    parallel = get_parallel_state()
    result = {"selected": []}
    with_entropy = args.entropy_coef != 0 or args.observe_training_entropy or sfc_observations.enabled(args)
    if with_entropy:
        result["entropy"] = []
    chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=True,
    )
    for i, (chunk, tokens, indices) in enumerate(chunks):
        sfc_observations.logit_audit(args, batch, i, chunk, tokens, indices)
        indices = torch.as_tensor(list(indices), dtype=torch.long)
        ids = torch.as_tensor(batch["rollout_topk_token_ids"][i])[indices].to(logits.device).long()
        ids = torch.cat((tokens.unsqueeze(-1), ids), dim=-1)
        selected, entropy = selected_log_probs_and_entropy(
            chunk,
            ids,
            group=parallel.tp.group if parallel.tp.size > 1 else None,
            vocab_size=getattr(args, "vocab_size", None),
            temperature=1.0,  # The retained response iterator already applies temperature.
            chunk_size=args.log_probs_chunk_size,
            with_entropy=with_entropy,
        )
        result["selected"].append(selected)
        if with_entropy:
            result["entropy"].append(entropy if args.entropy_coef else entropy.detach())
    if args.allgather_cp and parallel.cp.size > 1:
        allgather_cp_redistribute(
            result, logits=logits, args=args, total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"], max_seq_lens=batch.get("max_seq_lens"),
        )
    return result


def _local_candidates(args: Namespace, batch: RolloutBatch, key: str, device: torch.device) -> torch.Tensor:
    values = []
    for i, (total, response) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], strict=True)):
        maximum = batch["max_seq_lens"][i] if batch.get("max_seq_lens") is not None else None
        # Slice on CPU before copying to the device; each CP rank needs only its rows.
        value = slice_log_prob_with_cp(torch.as_tensor(batch[key][i]), total, response, args.qkv_format, maximum)
        values.append(value.to(device))
    return torch.cat(values)


def _regularization(
    args: Namespace,
    batch: RolloutBatch,
    entropy: torch.Tensor | None,
    log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    active: torch.Tensor,
    reduce: Callable[[torch.Tensor], torch.Tensor],
    kl_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = log_probs.new_zeros(())
    metrics = {}
    if entropy is not None:
        entropy = reduce(entropy)
        loss = loss - args.entropy_coef * entropy
        metrics["entropy_loss"] = entropy.detach()
    if args.use_kl_loss:
        reference = torch.cat(batch["ref_log_probs"]).detach()
        actor = log_probs if kl_log_probs is None else kl_log_probs
        # Mask before exponentiation: replacing infinity afterward can still give
        # NaN gradients through exp. Two bounded exponentials can multiply safely
        # in float32 (exp(40) * exp(40) < float32's maximum).
        valid = active & torch.isfinite(actor) & torch.isfinite(reference)
        difference = actor - reference
        valid = valid & (difference.abs() <= 40)
        ratio_log = log_probs - rollout_log_probs
        if args.use_unbiased_kl:
            valid = valid & torch.isfinite(ratio_log) & (ratio_log.abs() <= 40)
        actor = torch.where(valid, actor, 0.0)
        reference = torch.where(valid, reference, 0.0)
        # Keep the existing Miles KL estimator's gradient through the ratio.
        # This ratio uses the actual replayed sampling distribution even though
        # the reference penalty compares full-vocabulary probabilities.
        ratio = torch.where(valid, ratio_log, 0.0).exp() if args.use_unbiased_kl else None
        kl = compute_approx_kl(actor, reference, args.kl_loss_type, importance_ratio=ratio)
        kl = reduce(torch.where(valid, torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0), 0.0))
        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl
        metrics["kl_loss"] = kl.detach()
        metrics["kl_invalid_fraction"] = reduce((active & ~valid).float()).detach()
    return loss, metrics


def _importance_diagnostics(train, rollout, active, advantages, metrics, mode, clip, reduce):
    """Original-token denominators; raw tails remain visible with weighting off."""
    with torch.no_grad():
        log_ratio = train.detach() - rollout
        tails = {
            "lt_01": active & (log_ratio < math.log(0.1)),
            "lt_05": active & (log_ratio < math.log(0.5)),
            "gt_2": active & (log_ratio > math.log(2)),
            "gt_5": active & (log_ratio > math.log(5)),
        }
        log = {f"sc_ratio_{name}_fraction": reduce(tail.float()) for name, tail in tails.items()}
        signs = {"positive": advantages > 0, "negative": advantages < 0, "zero": advantages == 0}
        for name, tail in tails.items():
            for label, sign in signs.items():
                log[f"sc_ratio_{name}_adv_{label}_fraction"] = reduce((tail & sign).float())
        cap_hits = active & (log_ratio > math.log(clip)) if mode == "tis" else torch.zeros_like(active)
        log["sc_tis_clip_fraction"] = reduce(cap_hits.float())
        log["sc_tis_floor_fraction"] = reduce(torch.zeros_like(train))
        log["sc_correction_abs"] = reduce(metrics.correction.abs())
        log["sc_importance_weight_squared"] = reduce(metrics.importance_weight.square())
        change = (metrics.importance_weight - 1) * advantages
        log["sc_weight_advantage_change"] = reduce(change)
        log["sc_weight_advantage_change_abs"] = reduce(change.abs())
        return {key: value.detach() for key, value in log.items()}


def score_centering_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probabilities = _candidate_log_probs(args, batch, logits)
    selected = torch.cat(probabilities["selected"])
    active = torch.cat(
        get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            args.qkv_format,
            batch.get("max_seq_lens"),
        )
    ).bool()
    rollout = torch.where(active, torch.cat(batch["rollout_log_probs"]).detach(), 0.0)
    advantages = torch.where(active, torch.cat(batch["advantages"]).detach(), 0.0)
    ids = _local_candidates(args, batch, "rollout_topk_token_ids", logits.device)
    head = _local_candidates(args, batch, "rollout_topk_log_probs", logits.device)
    token_loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=selected[:, 0],
            train_head_log_probs=selected[:, 1:],
            rollout_log_probs=rollout,
            rollout_head_log_probs=head,
            head_mask=(ids >= 0) & active.unsqueeze(-1),
            advantages=advantages,
            mode=args.score_centering_is,
            tis_clip=args.score_centering_tis_clip,
            mis_low=args.score_centering_mis_low,
            mis_high=args.score_centering_mis_high,
        ),
    )
    pg_loss = sum_of_sample_mean(token_loss)
    entropy = torch.cat(probabilities["entropy"]) if "entropy" in probabilities else None
    kl_log_probs = selected[:, 0] if args.use_kl_loss else None
    loss, log = _regularization(args, batch, entropy, selected[:, 0], rollout, active, sum_of_sample_mean, kl_log_probs)
    loss = loss + pg_loss
    sampler_kl_coef = getattr(args, "sampler_kl_coef", 0.0)
    sampler_kl_tokens = (sampler_kl_loss(selected[:, 1:], head, ids >= 0, active)
                         if sampler_kl_coef != 0 or sfc_observations.enabled(args) else torch.zeros_like(token_loss))
    if sampler_kl_coef != 0:
        sampler_kl = sum_of_sample_mean(sampler_kl_tokens)
        # This is a constant coefficient, including fresh-sampler tokens.
        # Reference KL and the detached rollout-KL diagnostic remain separate.
        loss = loss + sampler_kl_coef * sampler_kl
        log["sampler_kl_loss"] = sampler_kl.detach()
        log["sampler_kl_weighted_loss"] = (sampler_kl_coef * sampler_kl).detach()
        # Native token-mean logging divides accumulated metrics by the original
        # valid-token count. A scalar here would falsely shrink the coefficient.
        log["sampler_kl_coef"] = sum_of_sample_mean(torch.full_like(token_loss, sampler_kl_coef)).detach()
    log.update({key: sum_of_sample_mean(value).detach() for key, value in metrics.as_log_dict().items()})
    log.update(_importance_diagnostics(
        selected[:, 0], rollout, active, advantages, metrics, args.score_centering_is,
        args.score_centering_tis_clip, sum_of_sample_mean,
    ))
    log.update(loss=loss.detach(), pg_loss=pg_loss.detach())
    train_log_probs = torch.where(active, selected[:, 0].detach(), 0.0)
    log["train_rollout_logprob_abs_diff"] = sum_of_sample_mean((train_log_probs - rollout).abs()).detach()
    # Match the policy-loss diagnostic: sampled-token k3 estimate of KL(rollout || train).
    rollout_train_kl = compute_approx_kl(rollout, train_log_probs, kl_loss_type="low_var_kl")
    rollout_train_kl = torch.where(
        active,
        torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    log["train_rollout_kl"] = sum_of_sample_mean(rollout_train_kl).detach()
    sfc_observations.capture(args, batch, selected, rollout, advantages, active, ids, head,
                             metrics, entropy, token_loss, sampler_kl_tokens)
    return loss, log
