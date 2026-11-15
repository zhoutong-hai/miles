"""Equal prompt-presentation PG means over ORIGINAL response-token masks."""

from collections import Counter, defaultdict

import torch


def validate_prompt_mean_args(args):
    if args.policy_loss_aggregation != "prompt-mean":
        return
    required = (
        args.train_backend == "megatron"
        and args.loss_type == "policy_loss"
        and args.calculate_per_token_loss
        and args.skip_actor_forward_only
        and not args.use_rollout_logprobs
        and args.use_tis
        and args.num_steps_per_rollout == 1
        and args.global_batch_size == args.rollout_batch_size * args.n_samples_per_prompt
        and args.advantage_estimator == "grpo"
        and args.rewards_normalization
        and not args.grpo_std_normalization
        and not args.normalize_advantages
        and args.entropy_coef == args.kl_coef == 0
        and not args.use_kl_loss
        and not args.use_opd
        and not args.use_opsm
        and not args.use_dynamic_global_batch_size
        and not args.multi_lora
        and args.custom_pg_loss_reducer_function_path is None
        and args.custom_convert_samples_to_train_data_path is None
        and args.custom_reward_post_process_path is None
        and args.custom_tis_function_path is None
        and args.rollout_data_postprocess_path is None
        and args.length_penalty_coeff == 0
    )
    if not required:
        raise ValueError(
            "prompt-mean requires one complete Megatron GRPO score-reuse update, original-token reduction, and no custom shaping/reducers"
        )


def prompt_mean_scales(group_ids, loss_masks, *, expected_groups, expected_fanout):
    """Compute global factors before DP splitting, balancing or microbatching.

    N_g includes every original valid token in all siblings, including A==0.
    Empty groups contribute nothing and are excluded from G. Miles counts a
    fully masked response as one in its optimizer token normalizer; include
    that convention in D so D/(G*N_g) cancels it exactly after accumulation.
    For nonempty responses D=N and this is i047's N/(G*N_g).
    """
    if len(group_ids) != len(loss_masks) or any(type(g) is not int or g < 0 for g in group_ids):
        raise ValueError("prompt-mean requires explicit integer prompt-presentation identities")
    counts = Counter(group_ids)
    if len(counts) != expected_groups or set(counts.values()) != {expected_fanout}:
        raise ValueError("prompt-mean requires every sibling of the complete optimizer batch")
    totals = defaultdict(int)
    normalizer = 0
    responses = 0
    for group, mask in zip(group_ids, loss_masks, strict=True):
        if any(value not in (0, 1, False, True) for value in mask):
            raise ValueError("Original response masks must be binary")
        tokens = int(sum(mask))
        totals[group] += tokens
        normalizer += max(tokens, 1)
        responses += tokens > 0
    groups = sum(count > 0 for count in totals.values())
    if not groups:
        raise ValueError("prompt-mean requires original valid response tokens")
    return [normalizer / (groups * totals[g]) if totals[g] else 0.0 for g in group_ids], normalizer / responses


def scale_prompt_losses(losses, factors, local_lengths):
    """Scale PG only; advantages, TIS, masks and observational reducers stay original."""
    if factors is None or len(factors) != len(local_lengths) or sum(local_lengths) != losses.numel():
        raise ValueError("Missing or misaligned complete-batch prompt factors")
    values = torch.stack([torch.as_tensor(x, device=losses.device, dtype=losses.dtype) for x in factors]).detach()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid prompt factors")
    return losses * torch.repeat_interleave(values, torch.tensor(local_lengths, device=losses.device))
