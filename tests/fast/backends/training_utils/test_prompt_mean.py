"""CPU algebra/integration checks; no claim of a Megatron or model-scale run.

The source loader executes the real native loss functions without importing
CUDA/Ray. Only the row scorer/layout is replaced by a CPU softmax layout.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss.hub.corrections import vanilla_tis_function
from miles.backends.training_utils.loss.hub.sign_metrics import advantage_sign_metrics
from miles.utils.prompt_mean import prompt_mean_scales, scale_prompt_losses, validate_prompt_mean_args

ROOT = Path(__file__).resolve().parents[4]
HUB = ROOT / "miles/backends/training_utils/loss/hub"


def _source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class _Layout:
    def __init__(self, batch, logits, reduce):
        self.batch, self.logits = batch, logits
        self.lengths = [len(x) for x in batch["advantages"]]
        self.sum_of_sample_mean = reduce

    def rows(self, values):
        return torch.cat(values)

    def score(self, logits, **kwargs):
        logp = logits.log_softmax(-1)
        return logp[:, 0], -(logp.exp() * logp).sum(-1).detach()

    def loss_masks(self):
        return torch.cat(self.batch.get("local_masks", self.batch["loss_masks"]))

    def per_sample(self, rows):
        return rows.split(self.lengths)

    def reducer(self, masks):
        return self.sum_of_sample_mean

    def ess_ratio(self, ppo_kl):
        return ppo_kl.new_zeros(())  # suppressed on the score-reuse path


def _loss_function():
    namespace = dict(
        torch=torch,
        _LOG_RATIO_EXP_CLAMP=20.0,
        response_layout=lambda args, batch, logits, reduce: _Layout(batch, logits, reduce),
        get_parallel_state=lambda: SimpleNamespace(),
        vanilla_tis_function=vanilla_tis_function,
        scale_prompt_losses=scale_prompt_losses,
        advantage_sign_metrics=advantage_sign_metrics,
    )
    _source_functions(
        HUB / "math_utils.py",
        ["_safe_clamp_log_ratio", "_safe_exp_neg_ppo_kl", "compute_policy_loss", "compute_approx_kl"],
        namespace,
    )
    return _source_functions(HUB / "losses.py", ["policy_loss_function"], namespace)["policy_loss_function"]


def _args(**updates):
    values = dict(
        policy_loss_aggregation="prompt-mean",
        train_backend="megatron",
        loss_type="policy_loss",
        calculate_per_token_loss=True,
        skip_actor_forward_only=True,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        use_rollout_logprobs=False,
        num_steps_per_rollout=1,
        global_batch_size=6,
        rollout_batch_size=3,
        n_samples_per_prompt=2,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        normalize_advantages=False,
        entropy_coef=0.0,
        kl_coef=0.0,
        use_kl_loss=False,
        use_opd=False,
        use_opsm=False,
        use_dynamic_global_batch_size=False,
        multi_lora=False,
        custom_pg_loss_reducer_function_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_reward_post_process_path=None,
        custom_tis_function_path=None,
        rollout_data_postprocess_path=None,
        length_penalty_coeff=0,
        observe_advantage_sign_metrics=True,
        observe_training_entropy=True,
        use_tis=True,
        get_mismatch_metrics=False,
        tis_clip_low=0.5,
        tis_clip=2.0,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        dump_details=None,
    )
    return SimpleNamespace(**(values | updates))


@pytest.mark.parametrize("split", ["whole", "unequal_microbatches", "dp_cp_slices"])
@pytest.mark.parametrize("empty_response", [False, True])
@pytest.mark.parametrize("routing_forward", [False, True])
def test_native_score_reuse_gradient_matches_direct_prompt_reinforce(split, empty_response, routing_forward):
    torch.manual_seed(42)
    masks = [torch.ones(n) for n in (1, 3, 2, 2, 1, 5)]
    if empty_response:
        masks[0].zero_()
    groups = [10, 10, 20, 20, 30, 30]
    factors, response_scale = prompt_mean_scales(groups, masks, expected_groups=3, expected_fanout=2)
    normalizer = sum(max(float(mask.sum()), 1) for mask in masks)
    logits = torch.randn(14, 5, requires_grad=True)
    original_q = torch.linspace(-5, -0.1, 14, requires_grad=True)
    advantages = [torch.full_like(mask, a) for mask, a in zip(masks, [1, -1, 0, 0, 2, -2], strict=True)]
    offsets = [0, 1, 4, 6, 8, 9]
    partitions = [list(range(6))] if split == "whole" else [[5, 0], [3], [1, 4, 2]]
    loss = logits.new_zeros(())
    metrics = {}
    for rows in partitions:
        cp_parts = [0, 1] if split == "dp_cp_slices" else [None]
        for cp in cp_parts:
            selections = [
                torch.arange(len(masks[i])) if cp is None else torch.arange(cp, len(masks[i]), 2) for i in rows
            ]
            token_ids = torch.cat([positions + offsets[i] for i, positions in zip(rows, selections, strict=True)])
            local_masks = [masks[i][positions] for i, positions in zip(rows, selections, strict=True)]
            batch = dict(
                advantages=[advantages[i][positions] for i, positions in zip(rows, selections, strict=True)],
                rollout_log_probs=list(original_q[token_ids].split([len(x) for x in local_masks])),
                loss_masks=[masks[i] for i in rows],
                local_masks=local_masks,
                response_lengths=[len(masks[i]) for i in rows],
                total_lengths=[len(masks[i]) + 1 for i in rows],
                prompt_loss_scale=[factors[i] for i in rows],
                entropy_response_scale=[response_scale] * len(rows),
            )
            if routing_forward:
                # The standalone forward exists to record R3 routes. Deliberately
                # perturb its scores: PG/TIS must still use the training forward.
                batch["log_probs"] = list((logits[token_ids].detach().log_softmax(-1)[:, 0] - 3).split(
                    [len(x) for x in local_masks]
                ))

            def reduce(values, masks=local_masks):
                return (values * torch.cat(masks)).sum()

            part, frame = _loss_function()(
                _args(skip_actor_forward_only=not routing_forward, use_routing_replay=routing_forward),
                batch, logits[token_ids], reduce,
            )
            loss = loss + part / normalizer
            for name, value in frame.items():
                metrics[name] = metrics.get(name, 0) + value / normalizer
    loss.backward()
    actual = logits.grad.clone()
    assert original_q.grad is None
    reference_logits = logits.detach().clone().requires_grad_()
    logp = reference_logits.log_softmax(-1)[:, 0]
    weights = (logp.detach().float() - original_q.detach().float()).exp().clamp(0.5, 2)
    token_loss = -torch.cat(advantages) * weights * logp
    expected = (
        sum(
            sum((token_loss[offsets[i] : offsets[i] + len(masks[i])] * masks[i]).sum() for i in [2 * g, 2 * g + 1])
            / sum(masks[i].sum() for i in [2 * g, 2 * g + 1])
            for g in range(3)
        )
        / 3
    )
    expected.backward()
    torch.testing.assert_close(actual, reference_logits.grad, rtol=2e-6, atol=1e-7)
    entropy = -(logits.detach().log_softmax(-1).exp() * logits.detach().log_softmax(-1)).sum(-1)
    response_mean = sum(
        (entropy[start : start + len(mask)] * mask).sum() / mask.sum()
        for start, mask in zip(offsets, masks, strict=True)
        if mask.sum() > 0
    ) / sum(mask.sum() > 0 for mask in masks)
    torch.testing.assert_close(metrics["entropy/equal_nonempty_response_mean"], response_mean)
    for name in ["positive", "negative", "zero"]:
        prefix = f"sign/{name}"
        histogram = sum(
            metrics[f"{prefix}/tis_{bin}_fraction"]
            for bin in ["below_floor", "floor_to_one", "one_to_cap", "above_cap"]
        )
        torch.testing.assert_close(histogram, metrics[f"{prefix}/token_fraction"])
    torch.testing.assert_close(
        sum(metrics[f"sign/{name}/token_fraction"] for name in ["positive", "negative", "zero"]),
        metrics["sign/original_valid_fraction"],
    )
    assert not any(key in metrics for key in ["pg_clipfrac", "ppo_kl", "ess_ratio", "ois"])


@pytest.mark.parametrize(
    "updates",
    [
        dict(num_steps_per_rollout=2),
        dict(global_batch_size=2),
        dict(normalize_advantages=True),
        dict(grpo_std_normalization=True),
        dict(entropy_coef=0.1),
        dict(use_rollout_logprobs=True),
        dict(custom_tis_function_path="other"),
    ],
)
def test_refuse_unsupported_objective_compositions(updates):
    validate_prompt_mean_args(_args())
    with pytest.raises(ValueError):
        validate_prompt_mean_args(_args(**updates))


def test_r3_scoring_forward_is_admitted_without_rollout_replay():
    validate_prompt_mean_args(_args(skip_actor_forward_only=False, use_routing_replay=True))
    with pytest.raises(ValueError):
        validate_prompt_mean_args(_args(skip_actor_forward_only=False))
    with pytest.raises(ValueError):
        validate_prompt_mean_args(_args(skip_actor_forward_only=False, use_routing_replay=True,
                                        use_rollout_routing_replay=True))


def test_composed_algorithm_cli_registers_prompt_mean_once():
    import argparse
    from miles.utils.args.configs.algo import AlgoConfig, AlgoRolloutOnlyConfig

    parser = argparse.ArgumentParser()
    AlgoConfig.add_arguments(parser)
    AlgoRolloutOnlyConfig.add_arguments(parser)
    args = parser.parse_args(["--policy-loss-aggregation", "prompt-mean", "--disable-grpo-std-normalization"])
    assert args.policy_loss_aggregation == "prompt-mean"
    assert args.rewards_normalization and not args.grpo_std_normalization


def test_empty_group_and_original_denominators():
    factors, _ = prompt_mean_scales([0, 0, 1, 1], [[0], [], [1], [1, 1, 1]], expected_groups=2, expected_fanout=2)
    assert factors == [0, 0, 1.5, 1.5]  # native D=6, one nonempty group N_g=4
    for ids, masks in [([0, 0, 1], [[1]] * 3), ([0, 0, None, None], [[1]] * 4), ([0, 0, 1, 1], [[0]] * 4)]:
        with pytest.raises(ValueError):
            prompt_mean_scales(ids, masks, expected_groups=2, expected_fanout=2)


@pytest.mark.parametrize("value", [0.1, 0.3, 0.7])
def test_native_uniform_reward_group_has_exact_zero_advantage(value):
    namespace = dict(
        torch=torch,
        reward_group_segments=lambda args, samples, sizes: [list(range(len(samples)))],
        rollout_segments=lambda samples, segments: [[row] for row in segments],
    )
    path = ROOT / "miles/ray/rollout/train_data_conversion.py"
    normalize = _source_functions(path, ["_normalize_rewards_by_rollout"], namespace)["_normalize_rewards_by_rollout"]
    assert normalize(_args(), [object()] * 16, [value] * 16, None) == [0.0] * 16
    torch.testing.assert_close(
        torch.tensor(normalize(_args(), [object()] * 2, [0.25, 0.75], None)), torch.tensor([-0.25, 0.25])
    )


def test_complete_batch_scales_follow_actual_native_dp_packaging():
    path = ROOT / "miles/ray/rollout/train_data_conversion.py"
    package = _source_functions(path, ["_package_shards"], {})["_package_shards"]
    factors, response_scale = prompt_mean_scales(
        [0, 0, 1, 1], [[1], [1], [1, 1], [1, 1]], expected_groups=2, expected_fanout=2
    )
    data = dict(prompt_loss_scale=factors, entropy_response_scale=[response_scale] * 4, total_lengths=[2, 2, 3, 3])
    shards = package(_args(), data, [[3, 0], [2, 1]])
    assert [shard["prompt_loss_scale"] for shard in shards] == [[0.75, 1.5], [0.75, 1.5]]
    assert all(shard["entropy_response_scale"] == [1.5, 1.5] for shard in shards)


def test_sign_statistics_preserve_finite_signed_entropy_and_zero_bucket():
    result = advantage_sign_metrics(
        torch.tensor([1.0, -1.0, 0.0]),
        torch.tensor([True] * 3),
        torch.tensor([0.2, -1e-8, 0.1]),
        torch.ones(3),
        torch.ones(3),
        low=0.5,
        high=2.0,
        reduce=torch.sum,
    )
    assert result["sign/negative/entropy_mass"] < 0
    assert result["sign/zero/token_fraction"] == 1
    with pytest.raises(ValueError):
        advantage_sign_metrics(
            torch.tensor([float("nan")]),
            torch.tensor([True]),
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1),
            low=0.5,
            high=2.0,
            reduce=torch.sum,
        )
