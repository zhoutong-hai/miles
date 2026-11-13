"""Megatron CP1 prompt-mean REINFORCE with original-sampler fixed MIS.

Native scoring and the existing Megatron original-token normalizer own the
distributed graph/scaling. No PPO surrogate or probability-based PG scaling is used.
"""

import torch

from miles.backends.training_utils.loss.hub.response_layout import response_layout
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.reinforce import fixed_mis_loss


def reinforce_loss_function(args, batch, logits, sum_of_sample_mean):
    state = get_parallel_state()
    if args.train_backend != "megatron" or state.cp.size != 1:
        raise ValueError("Prompt-mean REINFORCE currently supports Megatron CP1")
    if not args.backend.calculate_per_token_loss or not args.skip_actor_forward_only:
        raise ValueError("REINFORCE requires original-token reduction and training-forward score reuse")
    if args.entropy_coef or args.kl_coef or args.use_kl_loss or args.normalize_advantages:
        raise ValueError("REINFORCE does not compose with another objective or advantage whitening")
    if args.policy_loss_aggregation != "prompt-mean":
        raise ValueError("REINFORCE requires explicit prompt-mean aggregation")
    layout = response_layout(args, batch, logits, sum_of_sample_mean)
    logp, entropy = layout.score(logits, with_entropy=True, entropy_requires_grad=False)
    masks = layout.loss_masks()
    q = layout.rows(batch["rollout_log_probs"])
    advantage = layout.rows(batch["advantages"])
    scales = batch["prompt_loss_scales"]
    if scales is None:
        raise ValueError("Missing complete-batch prompt normalizers")
    factors = layout.rows([torch.full_like(x, float(s)) for x, s in zip(batch["advantages"], scales, strict=True)])
    bounds = (args.mis_positive_low, args.mis_positive_high, args.mis_negative_low, args.mis_negative_high)
    pg, weight, ratio = fixed_mis_loss(logp, q, advantage, masks, factors, bounds)
    original_reduce = layout.sum_of_sample_mean

    def reduce(x):
        return original_reduce(torch.where(masks.bool(), x, 0.0))

    loss = reduce(pg)
    metrics = {
        "loss": loss.detach(),
        "pg_loss": loss.detach(),
        "sfc/pg_prompt_mean": loss.detach(),
        "entropy_loss": reduce(entropy.detach()),
        "train_rollout_logprob_abs_diff": reduce((logp.detach().float() - q.float()).abs()),
        "train_rollout_kl": reduce(torch.exp((logp.detach().float() - q.float()).clamp(-10, 10)) - 1 - (logp.detach().float() - q.float()).clamp(-10, 10)),
        "sfc/applied_weight": reduce(weight),
        "sfc/raw_ratio": reduce(ratio),
        "sfc/zero_advantage_token_fraction": reduce((advantage == 0).float()),
        "sfc/MIS_rejected_fraction": reduce((weight == 0).float()),
        "sfc/positive_abs_coefficient_mass": reduce(weight * advantage.clamp_min(0) * factors),
        "sfc/negative_abs_coefficient_mass": reduce(weight * (-advantage).clamp_min(0) * factors),
    }
    if args.reinforce_diagnostics_dir:
        from miles.utils.reinforce_observations import capture

        capture(args, batch, layout, logits, logp, q, advantage, masks, factors, weight, ratio, entropy)
    return loss, metrics
