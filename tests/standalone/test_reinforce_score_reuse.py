"""Direct REINFORCE and SFC reuse: native losses, derivatives and private records."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.loss_hub import policy_observations, sfc_observations
from miles.backends.training_utils.loss_hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.loss_hub.sfc_summary import read_frames
from miles.utils.score_centering import validate_score_centering_args
from test_entropy_screens import _native_policy_function, arguments, local_parallel


@pytest.mark.parametrize("advantage", [-.5, 0., .5])
@pytest.mark.parametrize("objective", ["reinforce", "score_centering"])
def test_native_reused_scores_match_dense_objective_and_observations(tmp_path, monkeypatch, advantage, objective):
    monkeypatch.setenv("MILES_SOURCE_COMMIT", "cpu-reinforce-reuse")
    generator = torch.Generator().manual_seed(73)
    initial = torch.randn(1, 5, 5, generator=generator)
    tokens = torch.tensor([0, 1, 2, 3, 4, 0])
    mask = torch.tensor([1., 1., 0., 1., 1.])
    old = initial.log_softmax(-1)[0].gather(-1, tokens[1:, None]).squeeze(-1)
    q = old-torch.tensor([.1, .75, 1., 1.7, 3.]).log()
    head = torch.randn(5, 5, generator=generator).log_softmax(-1).topk(3, dim=-1)
    batch = dict(advantages=[torch.full((5,), advantage)], rollout_log_probs=[q],
        loss_masks=[mask], total_lengths=[6], response_lengths=[5], unconcat_tokens=[tokens],
        rollout_mask_sums=[mask.sum()], sample_indices=[10], weight_versions=[["v1"]],
        sfc_generation_log_probs=[q], rollout_topk_token_ids=[head.indices], rollout_topk_log_probs=[head.values],
        sfc_sample_metadata=[dict(valid_tokens=4, first_valid_positions=[0, 1, 3, 4],
                                  theme="a", scenario="b", advantage=advantage)])
    args = arguments(loss_type=objective, skip_actor_forward_only=True, custom_tis_function_path=None,
        use_tis=objective == "reinforce", score_centering_is="none", rollout_top_logprobs_num=3,
        score_centering_mis_low=.5, score_centering_mis_high=5., log_probs_chunk_size=2)
    if objective == "reinforce":
        args.rollout_top_logprobs_num = 0
    validate_score_centering_args(args)
    reducer = cp_utils.get_sum_of_sample_mean([6], [5], [mask], calculate_per_token_loss=True)
    function = _native_policy_function() if objective == "reinforce" else score_centering_loss_function
    logits = initial.clone().requires_grad_(True)
    plain, metrics = function(args, batch, logits, reducer)
    gradient, = torch.autograd.grad(plain, logits)
    dense = initial.clone().requires_grad_(True)
    logp = dense[0].log_softmax(-1)
    selected = logp.gather(-1, tokens[1:, None]).squeeze(-1)
    if objective == "reinforce":
        weight = (selected.detach()-q).exp().clamp(.5, 2.)
        expected = -advantage*weight*selected
        assert not {"ppo_kl", "pg_clipfrac", "ess_ratio", "ois"} & metrics.keys()
    else:
        expected, _ = score_centering_loss(ScoreCenteringInputs(selected, logp.gather(-1, head.indices),
            q, head.values, torch.ones_like(head.indices, dtype=torch.bool), torch.full_like(q, advantage), mode="none"))
    expected = (expected*mask).sum()
    dense_gradient, = torch.autograd.grad(expected, dense)
    torch.testing.assert_close(plain, expected)
    torch.testing.assert_close(gradient, dense_gradient, atol=2e-7, rtol=2e-5)
    args.sfc_diagnostics_dir = str(tmp_path)
    sfc_observations.configure_rollout(args, 0, "v1", {**batch, "rewards": [advantage]})
    observed_logits = initial.clone().requires_grad_(True)
    observed, _ = function(args, batch, observed_logits, reducer)
    observed_gradient, = torch.autograd.grad(observed, observed_logits)
    torch.testing.assert_close(observed, plain, atol=0, rtol=0)
    torch.testing.assert_close(observed_gradient, gradient, atol=0, rtol=0)
    summary = sfc_observations.finish(args, "NORMAL")
    frame, = read_frames(tmp_path/"paired_scores/update-000001/cp-0.bin")
    assert frame["loss_type"] == objective and frame["trainer_score_source"] == "training_forward"
    assert frame["standalone_actor_forward"] is False
    assert "cached_pre_update_logp" not in frame["data"]
    assert summary["sfc/original_valid_tokens"] == 4
    assert summary["sfc/pg_token_mean"] == pytest.approx(float(plain.detach()/4))
    if advantage == 0:
        assert observed_gradient.count_nonzero() == 0


def test_reinforce_tis_audit_matches_dense_gradient_without_ppo_anchor(local_parallel):
    args = arguments(loss_type="reinforce", skip_actor_forward_only=True, custom_tis_function_path=None)
    logits = torch.randn(8, 7, generator=torch.Generator().manual_seed(55), requires_grad=True)
    samples = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    logp = logits[:, :5].log_softmax(-1)
    selected = logp.gather(-1, samples[:, None]).squeeze(-1)
    old = selected.detach()
    q = old-torch.tensor([.1, .3, .75, 1., 1.7, 3., 5., 1.]).log()
    advantage = torch.tensor([1., 1., -1., 0., -2., -.5, .8, 1.])
    weights = (old-q).exp().clamp(.5, 2.)
    expected = (-advantage*weights*selected).sum()/19
    gradient, = torch.autograd.grad(expected, logits)
    actual = policy_observations.component_gradients(logits.detach(), samples, q, old, advantage,
                                                    args=args, denominator=19)
    torch.testing.assert_close(actual["actual_tis_pg"], gradient, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("objective", ["reinforce", "score_centering"])
def test_single_step_reuse_guard_and_rollout_replay(objective):
    root = Path(__file__).resolve().parents[2]
    namespace = dict(SimpleNamespace=SimpleNamespace)
    for path, name in [("miles/utils/arguments.py", "validate_skip_actor_forward_only"),
                       ("tests/fast/utils/test_arguments.py", "_make_skip_actor_forward_only_args")]:
        tree = ast.parse((root/path).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    args = namespace["_make_skip_actor_forward_only_args"](loss_type=objective,
        use_routing_replay=True, use_rollout_routing_replay=True, num_steps_per_rollout=1)
    validate = namespace["validate_skip_actor_forward_only"]
    validate(args)
    args.num_steps_per_rollout = 2
    with pytest.raises(AssertionError, match="exactly one optimizer step"):
        validate(args)
    args.num_steps_per_rollout = 1
    args.use_rollout_routing_replay = False
    with pytest.raises(AssertionError, match="routing-replay"):
        validate(args)


@pytest.mark.parametrize("changes", [dict(skip_actor_forward_only=False), dict(use_rollout_logprobs=True),
                                    dict(advantage_estimator="reinforce_plus_plus")])
def test_direct_reinforce_rejects_changed_score_or_advantage_semantics(changes):
    args = arguments(**(dict(loss_type="reinforce", skip_actor_forward_only=True) | changes))
    with pytest.raises(ValueError, match="Direct REINFORCE"):
        validate_score_centering_args(args)
