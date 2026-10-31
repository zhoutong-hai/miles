"""Validation for the selected observational REINFORCE/TIS path."""

import math


def validate_policy_diagnostics_args(args):
    required = {
        "loss_type": "reinforce",
        "advantage_estimator": "grpo",
        "skip_actor_forward_only": True,
        "use_rollout_logprobs": False,
        "use_tis": True,
        "tis_clip_low": 0.5,
        "tis_clip": 2.0,
        "entropy_coef": 0.0,
        "kl_coef": 0.0,
        "use_kl_loss": False,
    }
    if any(getattr(args, key, None) != value for key, value in required.items()):
        raise ValueError("Selected REINFORCE path requires GRPO, detached training-score reuse and native TIS[.5,2]")
    if getattr(args, "custom_tis_function_path", None) is not None:
        raise ValueError("Selected REINFORCE observations bind native TIS")
    for key in (
        "normalize_advantages",
        "use_opsm",
        "use_sampling_support_replay",
        "true_on_policy_mode",
        "fully_async",
        "partial_rollout",
    ):
        if getattr(args, key, False):
            raise ValueError("Unqualified policy diagnostic treatment: " + key)
    if getattr(args, "custom_pg_loss_reducer_function_path", None):
        raise ValueError("Policy diagnostics require the native original-token reducer")
    if not args.policy_diagnostics_dir:
        if getattr(args, "save_full_lora_state", False):
            raise ValueError("Selected full-state saving requires paired diagnostic bindings")
        return
    if not args.calculate_per_token_loss or args.qkv_format != "thd":
        raise ValueError("Selected K3 diagnostics require THD original-token reduction")
    if (
        args.rollout_top_p != 1.0
        or args.rollout_top_k != -1
        or not math.isfinite(args.rollout_temperature)
        or args.rollout_temperature <= 0
    ):
        raise ValueError("Selected diagnostics require unfiltered, positive-temperature generation")
    for name in ("policy_audit_updates", "policy_probe_updates"):
        values = getattr(args, name)
        if (
            not values
            or sorted(set(values)) != values
            or any(type(v) is not int or v < (0 if "probe" in name else 1) for v in values)
        ):
            raise ValueError("Invalid diagnostic update schedule: " + name)
    if not args.policy_fixed_probe or not args.policy_fixed_probe_sha256:
        raise ValueError("Policy diagnostics require a sealed fixed-prefix probe")
    if getattr(args, "save_full_lora_state", False):
        if (
            args.num_rollout != 40
            or args.save_interval not in (10, 20)
            or not args.save
            or args.no_save_optim
            or args.no_save_rng
            or args.async_save
        ):
            raise ValueError("Selected full-state run requires40 updates, sync ten/twenty-update saves, optimizer and RNG")
