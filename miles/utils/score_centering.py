"""Narrow native-generation contract for the retained GLM SFC/KL backport."""

import math
import os


def observation_updates(args, *, probe=False):
    name = "sfc_fixed_probe_updates" if probe else "sfc_logit_audit_updates"
    default = (0, 8, 16, 32, 48, 64) if probe else (8, 32, 64)
    values = tuple(getattr(args, name, default))
    if not values or any(type(value) is not int or value < (0 if probe else 1) for value in values):
        raise ValueError(f"{name} requires nonnegative probe or positive audit update positions")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} positions must be unique and increasing")
    return values


def observation_importance_mode(args):
    if getattr(args, "loss_type", "policy_loss") == "score_centering":
        return args.score_centering_is
    correction = getattr(args, "custom_tis_function_path",
                         "miles.backends.training_utils.loss_hub.corrections.icepop_function")
    if correction in (None, "miles.backends.training_utils.loss_hub.corrections.vanilla_tis_function"):
        return "policy_tis"
    return "mis"


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
    mis_bounds = tuple(getattr(args, name, None) for name in (
        "mis_positive_low", "mis_positive_high", "mis_negative_low", "mis_negative_high"))
    if any(value is not None for value in mis_bounds):
        if (any(value is None for value in mis_bounds)
            or not all(math.isfinite(value) for value in mis_bounds)
            or not (0 < mis_bounds[0] <= mis_bounds[1] and 0 < mis_bounds[2] <= mis_bounds[3])):
            raise ValueError("Specify four finite positive ordered MIS advantage bounds")
        policy_mis = (getattr(args, "loss_type", None) in ("policy_loss", "reinforce") and getattr(args, "use_tis", False)
                      and getattr(args, "custom_tis_function_path", None)
                      == "miles.backends.training_utils.loss_hub.corrections.icepop_function")
        sfc_mis = (getattr(args, "loss_type", None) == "score_centering"
                   and getattr(args, "score_centering_is", None) == "mis")
        if not (policy_mis or sfc_mis):
            raise ValueError("Fixed advantage bounds require native policy MIS or SFC MIS")
    coefficient = getattr(args, "sampler_kl_coef", 0.0)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("--sampler-kl-coef must be finite and non-negative")
    diagnostics = getattr(args, "sfc_diagnostics_dir", None)
    if getattr(args, "loss_type", None) == "reinforce":
        if (not getattr(args, "skip_actor_forward_only", False)
            or getattr(args, "use_rollout_logprobs", False)
            or args.advantage_estimator != "grpo" or getattr(args, "use_opsm", False)):
            raise ValueError("Direct REINFORCE requires single-step training-score reuse and GRPO advantages")
    if diagnostics:
        observation_updates(args, probe=True)
        observation_updates(args)
        if (getattr(args, "normalize_advantages", False) or not getattr(args, "calculate_per_token_loss", False)
            or getattr(args, "kl_coef", 0) or getattr(args, "use_kl_loss", False)
            or args.score_centering_tis_clip != 2):
            raise ValueError("Selected SFC observations bind token mean, raw centered rewards, cap2 and no reference/reward KL")
    if getattr(args, "sfc_fixed_probe", None) and (not diagnostics or not getattr(args, "sfc_fixed_probe_sha256", None)):
        raise ValueError("SFC fixed probe needs a diagnostics directory and source SHA256")
    if getattr(args, "loss_type", None) != "score_centering":
        if coefficient or getattr(args, "rollout_top_logprobs_num", 0):
            raise ValueError("The GLM sampler-KL/candidate backport requires --loss-type score_centering")
        if diagnostics:
            if args.loss_type == "policy_loss" and getattr(args, "skip_actor_forward_only", False):
                raise ValueError("Historical PPO observations retain their separate scoring path; select reinforce for reuse")
            if (args.loss_type not in ("policy_loss", "reinforce") or not getattr(args, "use_tis", False)
                or getattr(args, "custom_tis_function_path", None) not in (
                    None, "miles.backends.training_utils.loss_hub.corrections.vanilla_tis_function",
                    "miles.backends.training_utils.loss_hub.corrections.icepop_function")
                or args.tis_clip_low != .5 or args.tis_clip != 2):
                raise ValueError("Policy observations bind native TIS or MIS [0.5,2] only")
            unsupported = ("use_opsm", "use_rollout_logprobs", "true_on_policy_mode",
                           "custom_pg_loss_reducer_function_path", "partial_rollout", "fully_async")
            if any(getattr(args, key, False) for key in unsupported) or args.advantage_estimator != "grpo":
                raise ValueError("Policy observations require native GRPO reduction and aligned trainer scores")
            if args.entropy_coef != 0 or getattr(args, "qkv_format", "thd") != "thd":
                raise ValueError("MIS observation binding requires entropy0 and THD")
            validate_candidate_sampling({"top_p": args.rollout_top_p, "top_k": args.rollout_top_k},
                                        temperature=args.rollout_temperature)
            if os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB", "").lower() in ("1", "true"):
                raise ValueError("MIS observations require actual sampling logprobs")
        return
    if args.rollout_top_logprobs_num <= 0:
        raise ValueError("Score centering requires positive --rollout-top-logprobs-num")
    if args.score_centering_is not in ("none", "tis", "mis"):
        raise ValueError("This backport selects native importance mode none, tis or mis")
    if args.score_centering_is == "mis" and any(value is None for value in mis_bounds):
        raise ValueError("SFC MIS requires all four explicit advantage-sign bounds")
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
