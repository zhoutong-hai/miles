"""Narrow native-generation contract for the retained GLM SFC/KL backport."""

import math
import os


def validate_candidate_sampling(sampling, *, temperature):
    """Candidates must describe the actual unfiltered generation distribution."""
    if not math.isfinite(temperature) or temperature <= 0 or sampling.get("temperature", temperature) != temperature:
        raise ValueError("SFC requires the same positive temperature on every training generation")
    if sampling.get("top_p", 1.0) != 1.0 or sampling.get("top_k", -1) != -1 or sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("The GLM SFC backport requires unfiltered top_p=1, top_k=-1, min_p=0 sampling")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"The GLM SFC backport does not support constrained sampling ({key})")


def validate_score_centering_args(args):
    coefficient = getattr(args, "sampler_kl_coef", 0.0)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("--sampler-kl-coef must be finite and non-negative")
    if getattr(args, "loss_type", None) != "score_centering":
        if coefficient or getattr(args, "rollout_top_logprobs_num", 0):
            raise ValueError("The GLM sampler-KL/candidate backport requires --loss-type score_centering")
        return
    if args.rollout_top_logprobs_num <= 0:
        raise ValueError("Score centering requires positive --rollout-top-logprobs-num")
    if args.score_centering_is not in ("none", "tis"):
        raise ValueError("This backport selects native importance mode none or tis")
    if not math.isfinite(args.score_centering_tis_clip) or args.score_centering_tis_clip <= 0:
        raise ValueError("--score-centering-tis-clip must be finite and positive")
    validate_candidate_sampling(
        {"top_p": args.rollout_top_p, "top_k": args.rollout_top_k}, temperature=args.rollout_temperature
    )
    if args.advantage_estimator != "grpo":
        raise ValueError("Score centering requires group-centered GRPO advantages")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("The retained GLM candidate/CP backport currently supports qkv_format=thd")
    incompatible = (
        "use_tis", "custom_tis_function_path", "use_opsm", "true_on_policy_mode",
        "recompute_logprobs_via_prefill", "sglang_speculative_algorithm",
        "custom_pg_loss_reducer_function_path", "multi_lora", "use_opd",
        "custom_convert_samples_to_train_data_path", "use_sampling_support_replay",
        "fully_async", "partial_rollout",
    )
    for option in incompatible:
        if getattr(args, option, None):
            raise ValueError(f"The GLM SFC backport is incompatible with --{option.replace('_', '-')}")
    if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
        raise ValueError("SFC requires actual sampling logprobs: SGLANG_RETURN_ORIGINAL_LOGPROB=0")
