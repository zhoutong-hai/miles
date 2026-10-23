"""Distributed update and exact-resume wiring for actor entropy feedback."""

from __future__ import annotations

import math
from argparse import Namespace

import torch

from miles.backends.megatron_utils.initialize import is_first_replica_megatron_main_rank
from miles.backends.training_utils.adaptive_entropy import next_entropy_coefficient, saved_entropy_coefficient
from miles.utils.distributed_utils import get_gloo_group


def advance_adaptive_entropy(args: Namespace, loss_dict: dict, optimizer_advanced: bool) -> dict[str, float]:
    """Use one globally reduced entropy observation for the next optimizer step.

    Only the first replica's last pipeline rank supplies the already aggregated
    token-mean metric. Every trainer rank participates so ``args.entropy_coef``
    remains identical in model forward/backward and checkpoint common state.
    """
    group = get_gloo_group()
    source = is_first_replica_megatron_main_rank()
    entropy = float(loss_dict.get("entropy_loss", float("nan"))) if source else 0.0
    invalid = source and (not math.isfinite(entropy) or entropy < 0)
    signal = torch.tensor(
        [entropy if not invalid else 0.0, float(source), float(invalid), float(optimizer_advanced)],
        dtype=torch.float64,
        device="cpu",
    )
    torch.distributed.all_reduce(signal, group=group)
    world_size = torch.distributed.get_world_size(group=group)
    advanced_count = int(signal[3].item())
    if advanced_count not in (0, world_size):
        raise RuntimeError("Adaptive entropy optimizer-step state disagrees across trainer ranks")
    applied = float(args.entropy_coef)
    if advanced_count:
        if signal[1].item() != 1 or signal[2].item() != 0:
            raise RuntimeError("Adaptive entropy requires one finite global token-mean observation")
        args.entropy_coef = next_entropy_coefficient(
            applied,
            float(signal[0].item()),
            args.adaptive_entropy_target,
            args.adaptive_entropy_rate,
            args.adaptive_entropy_max_coef,
        )
    return {
        "entropy_coef_applied": applied,
        "entropy_coef_next": float(args.entropy_coef),
        "entropy_target": float(args.adaptive_entropy_target),
        "entropy_controller_advanced": float(bool(advanced_count)),
    }


def restore_adaptive_entropy(args: Namespace, iteration: int) -> None:
    """Restore the next-step coefficient from the same full checkpoint as the model."""
    if getattr(args, "adaptive_entropy_target", None) is None or args.finetune:
        return
    if args.no_load_optim or args.no_load_rng:
        raise ValueError("Adaptive entropy resume requires full optimizer and RNG state")

    packet: list[tuple[float | None, str | None]] = [(None, None)]
    if torch.distributed.get_rank() == 0:
        try:
            from megatron.core import dist_checkpointing
            from megatron.training.checkpointing import get_checkpoint_name

            checkpoint_dir = get_checkpoint_name(args.load, iteration, return_base_dir=True)
            common_state = dist_checkpointing.load_common_state_dict(checkpoint_dir)
            packet[0] = (saved_entropy_coefficient(common_state, args, iteration), None)
        except Exception as exc:
            packet[0] = (None, f"{type(exc).__name__}: {exc}")
    torch.distributed.broadcast_object_list(packet, src=0, group=get_gloo_group())
    coefficient, error = packet[0]
    if error is not None:
        raise RuntimeError(f"Unable to restore adaptive entropy state: {error}")
    args.entropy_coef = coefficient
