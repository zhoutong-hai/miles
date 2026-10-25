"""Configuration contracts for score centering (arXiv:2609.20807)."""

import math
import os
from argparse import Namespace

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sampling


def validate_score_centering_args(args: Namespace) -> None:
    if args.policy_score_centering:
        _validate_policy_score_centering_args(args)
        return
    if args.loss_type != "score_centering":
        return
    if args.rollout_top_logprobs_num <= 0:
        raise ValueError("Score centering requires a positive --rollout-top-logprobs-num")
    if not math.isfinite(args.score_centering_tis_clip) or args.score_centering_tis_clip <= 0:
        raise ValueError("--score-centering-tis-clip must be finite and positive")
    low, high = args.score_centering_mis_low, args.score_centering_mis_high
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("Score-centering MIS bounds must be finite with 0 < low <= high")
    validate_rollout_topk_logprobs_sampling(
        {"top_k": args.rollout_top_k},
        temperature=args.rollout_temperature,
        candidate_count=args.rollout_top_logprobs_num,
    )
    if args.advantage_estimator != "grpo":
        raise ValueError("Score centering currently supports --advantage-estimator grpo (group-centered rewards)")
    incompatible = {
        "use_tis": "use --score-centering-is instead",
        "custom_tis_function_path": "only the built-in score-centering TIS/MIS weights are supported",
        "use_opsm": "sequence masking changes the score-centering estimator",
        "true_on_policy_mode": "score centering uses float32 probability arithmetic",
        "recompute_logprobs_via_prefill": "sampler probabilities must be recorded at generation time",
        "sglang_speculative_algorithm": "speculative candidate-logprob semantics are not verified",
        "custom_pg_loss_reducer_function_path": "use the standard token/sample reducer",
        "multi_lora": "per-sample Tinker losses bypass the score-centering loss",
        "use_opd": "distillation composition is not supported",
        "custom_convert_samples_to_train_data_path": "custom converters skip rollout top-k logprobs validation",
    }
    for option, reason in incompatible.items():
        if getattr(
            args, option
        ):  # config-access-exempt: option is selected from the required compatibility field names
            raise ValueError(f"Score centering is incompatible with --{option.replace('_', '-')}: {reason}")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("Score centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")


def _validate_policy_score_centering_args(args: Namespace) -> None:
    """Keep the experimental three-distribution path narrow and fail before model load."""
    required = {
        "loss_type": "policy_loss",
        "advantage_estimator": "grpo",
        "train_backend": "megatron",
        "context_parallel_size": 1,
        "log_probs_backend": "torch",
        "tis_clip_low": 0.0,
        "score_centering_is": "none",
        "rollout_top_p": 1.0,
        "rollout_top_k": -1,
    }
    for key, expected in required.items():
        if getattr(args, key) != expected:  # config-access-exempt: required launch fields selected by name
            raise ValueError(f"--policy-score-centering requires --{key.replace('_', '-')}={expected}")
    if not args.use_tis or args.rollout_top_logprobs_num <= 0:
        raise ValueError("--policy-score-centering requires --use-tis and positive --rollout-top-logprobs-num")
    if args.eps_clip_c is not None:
        raise ValueError("--policy-score-centering is incompatible with --eps-clip-c")
    if not (
        math.isfinite(args.tis_clip)
        and args.tis_clip > 0
        and math.isfinite(args.eps_clip)
        and 0 <= args.eps_clip < 1
        and math.isfinite(args.eps_clip_high)
        and args.eps_clip_high >= 0
    ):
        raise ValueError("Policy centering requires finite positive TIS and valid PPO clipping bounds")
    incompatible = (
        "use_rollout_logprobs",
        "skip_actor_forward_only",
        "custom_tis_function_path",
        "custom_pg_loss_reducer_function_path",
        "use_opsm",
        "true_on_policy_mode",
        "use_sampling_support_replay",
        "recompute_logprobs_via_prefill",
        "sglang_speculative_algorithm",
        "multi_lora",
        "use_opd",
        "custom_convert_samples_to_train_data_path",
    )
    for key in incompatible:
        if getattr(args, key):  # config-access-exempt: incompatible launch fields selected by name
            raise ValueError(f"--policy-score-centering is incompatible with --{key.replace('_', '-')}")
    validate_rollout_topk_logprobs_sampling(
        {"top_k": args.rollout_top_k},
        temperature=args.rollout_temperature,
        candidate_count=args.rollout_top_logprobs_num,
    )
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("Policy centering requires SGLANG_RETURN_ORIGINAL_LOGPROB=0 on rollout servers")
