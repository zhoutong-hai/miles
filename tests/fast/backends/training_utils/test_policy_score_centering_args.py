"""Fail closed before initialization when the experimental policy path is unsupported."""

from argparse import Namespace

import pytest

from miles.utils.score_centering import validate_score_centering_args


def _args(**updates):
    values = dict(
        policy_score_centering=True,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        train_backend="megatron",
        context_parallel_size=1,
        log_probs_backend="torch",
        tis_clip_low=0.0,
        tis_clip=2.0,
        use_tis=True,
        score_centering_is="none",
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        rollout_top_logprobs_num=128,
        eps_clip=0.2,
        eps_clip_high=0.2,
        use_rollout_logprobs=False,
        skip_actor_forward_only=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        use_opsm=False,
        true_on_policy_mode=False,
        eps_clip_c=None,
        use_sampling_support_replay=False,
        recompute_logprobs_via_prefill=False,
        sglang_speculative_algorithm=None,
        multi_lora=False,
        use_opd=False,
        custom_convert_samples_to_train_data_path=None,
    )
    return Namespace(**(values | updates))


def test_selected_configuration_is_accepted(monkeypatch):
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    validate_score_centering_args(_args())


@pytest.mark.parametrize(
    "update",
    [
        dict(loss_type="score_centering"),
        dict(advantage_estimator="ppo"),
        dict(advantage_estimator="gspo"),
        dict(train_backend="fsdp"),
        dict(context_parallel_size=2),
        dict(log_probs_backend="fused"),
        dict(use_tis=False),
        dict(tis_clip_low=0.1),
        dict(tis_clip=float("inf")),
        dict(use_rollout_logprobs=True),
        dict(skip_actor_forward_only=True),
        dict(eps_clip_c=3),
        dict(eps_clip_c=0),
        dict(rollout_top_logprobs_num=0),
        dict(rollout_top_p=0.9),
        dict(rollout_top_k=128),
        dict(score_centering_is="tis"),
        dict(custom_tis_function_path="custom"),
        dict(custom_pg_loss_reducer_function_path="custom"),
        dict(use_opsm=True),
        dict(true_on_policy_mode=True),
        dict(use_sampling_support_replay=True),
        dict(sglang_speculative_algorithm="DRAFT"),
        dict(multi_lora=True),
        dict(use_opd=True),
        dict(custom_convert_samples_to_train_data_path="custom"),
        dict(recompute_logprobs_via_prefill=True),
        dict(eps_clip=1),
        dict(eps_clip_high=-0.1),
        dict(rollout_temperature=0),
    ],
)
def test_unsupported_combinations_are_rejected_before_model_load(update, monkeypatch):
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    with pytest.raises(ValueError):
        validate_score_centering_args(_args(**update))


def test_original_sampler_logprobs_are_rejected(monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB"):
        validate_score_centering_args(_args())


def test_disabled_feature_leaves_native_policy_configuration_unchanged():
    args = _args(policy_score_centering=False, advantage_estimator="gspo", use_tis=False)
    before = vars(args).copy()
    validate_score_centering_args(args)
    assert vars(args) == before
