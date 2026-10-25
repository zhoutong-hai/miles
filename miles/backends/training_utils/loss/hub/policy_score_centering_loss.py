"""Opt-in additive centering for native token PPO + old-policy TIS."""

from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
from miles.backends.training_utils.loss.hub.score_centering.policy import (
    PolicyScoreCenteringInputs,
    policy_score_centering_correction,
)
from miles.backends.training_utils.loss.hub.score_centering_loss import _candidate_log_probs, _local_candidates
from miles.utils.types import RolloutBatch


def old_policy_log_probs_with_candidates(
    logits: torch.Tensor,
    *,
    args: Namespace,
    batch: RolloutBatch,
    callback: Callable[[torch.Tensor], dict[str, list[torch.Tensor]]],
) -> dict[str, list[torch.Tensor]]:
    """Capture candidate scores in the existing frozen old-trainer forward.

    The native callback remains authoritative for sampled old-policy scores.
    Candidate scores share its padded-vocabulary distribution and fixed temperature.
    """
    if batch.get("multimodal_train_inputs") is not None:
        raise ValueError("Policy score centering currently requires text-only response alignment")
    result = callback(logits)
    result["old_policy_topk_log_probs"] = [
        value[:, 1:].detach() for value in _candidate_log_probs(args, batch, logits, policy_loss=True)["selected"]
    ]
    return result


def policy_centering_token_loss(
    args: Namespace, batch: RolloutBatch, logits: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Build an additive token correction; no sampled TIS or reduction is repeated."""
    old_head = batch.get("old_policy_topk_log_probs")
    if old_head is None:
        raise ValueError("Policy centering requires candidate scores captured by the frozen old-trainer forward")
    current = torch.cat(_candidate_log_probs(args, batch, logits, policy_loss=True)["selected"])[:, 1:]
    active = torch.cat(
        get_local_response_loss_masks(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
            args.qkv_format,
            batch.get("max_seq_lens"),
        )
    ).bool()
    advantages = torch.where(active, torch.cat(batch["advantages"]).detach(), 0.0)
    ids = _local_candidates(args, batch, "rollout_topk_token_ids", logits.device)
    rollout = _local_candidates(args, batch, "rollout_topk_log_probs", logits.device)
    loss, metrics = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=current,
            old_head_log_probs=torch.cat(old_head).to(logits.device).detach(),
            rollout_head_log_probs=rollout,
            head_mask=(ids >= 0) & active[:, None],
            advantages=advantages,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
            tis_clip=args.tis_clip,
        )
    )
    debug = None
    if args.dump_details is not None:
        debug = {
            "train_head_log_probs": current.detach(),
            "old_head_log_probs": torch.cat(old_head).detach(),
            "rollout_head_log_probs": rollout.detach(),
            "head_ids": ids.detach(),
            "centering_token_loss": loss.detach(),
        }
    return loss, metrics, debug


def checked_policy_frozen_log_probs(
    old: list[torch.Tensor] | None,
    rollout: list[torch.Tensor] | None,
    active: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Fail on corrupt trained rows; observation rows must not create 0*NaN in native TIS."""
    if old is None or rollout is None:
        raise ValueError("Policy centering requires sampled old-trainer and rollout log-probabilities")
    lengths = [value.numel() for value in old]
    old_flat, rollout_flat = torch.cat(old).detach(), torch.cat(rollout).detach()
    if old_flat.shape != active.shape or rollout_flat.shape != active.shape:
        raise ValueError("Policy-centering sampled probability/mask alignment mismatch")
    if not (torch.isfinite(old_flat[active]).all() and torch.isfinite(rollout_flat[active]).all()):
        raise ValueError("Policy centering requires finite sampled old-trainer and rollout log-probabilities")
    return (
        list(torch.where(active, old_flat, 0.0).split(lengths)),
        list(torch.where(active, rollout_flat, 0.0).split(lengths)),
    )
