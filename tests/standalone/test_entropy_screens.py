"""Changed-path CPU checks for the i020-based MIS and SFC entropy screens."""

import ast
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils, parallel
from miles.backends.training_utils.loss_hub import corrections, math_utils, policy_observations, sfc_observations
from miles.backends.training_utils.loss_hub.sfc_summary import read_frames
from miles.utils.score_centering import observation_updates, validate_score_centering_args


@pytest.fixture(autouse=True)
def local_parallel():
    previous = parallel._parallel_state
    parallel.set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=1, rank=0, group=None),
        tp=SimpleNamespace(size=1, rank=0, group=None), is_pp_last_stage=True,
        effective_dp=SimpleNamespace(size=1)))
    yield
    parallel.set_parallel_state(previous)
    sfc_observations._CONTEXT = None


def arguments(**changes):
    return Namespace(**(dict(loss_type="policy_loss", use_tis=True, sfc_diagnostics_dir=None,
        sfc_fixed_probe_updates=[0, 5, 10, 15, 20], sfc_logit_audit_updates=[10, 20],
        score_centering_tis_clip=2., sampler_kl_coef=0., rollout_top_logprobs_num=0,
        custom_tis_function_path="miles.backends.training_utils.loss_hub.corrections.icepop_function",
        tis_clip_low=.5, tis_clip=2., eps_clip=.2, eps_clip_high=.28, eps_clip_c=10.,
        rollout_temperature=1., rollout_top_p=1., rollout_top_k=-1, vocab_size=5,
        entropy_coef=0., calculate_per_token_loss=True, normalize_advantages=False,
        observe_training_entropy=True, qkv_format="thd", allgather_cp=False,
        true_on_policy_mode=False, skip_actor_forward_only=False, use_rollout_logprobs=False,
        advantage_estimator="grpo", use_opsm=False, get_mismatch_metrics=False,
        custom_pg_loss_reducer_function_path=None, use_kl_loss=False, wandb_run_id="screen") | changes))


@pytest.mark.parametrize("temperature", [.7, 1.])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_native_ppo_mis_logit_components_match_dense_autograd(temperature, asymmetric):
    bounds = dict(mis_positive_low=.25, mis_positive_high=4., mis_negative_low=.5, mis_negative_high=2.) if asymmetric else {}
    args = arguments(rollout_temperature=temperature, **bounds)
    logits = torch.randn(8, 7, generator=torch.Generator().manual_seed(55), requires_grad=True)
    samples = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    logp = (logits[:, :5]/temperature).log_softmax(-1)
    selected = logp.gather(-1, samples[:, None]).squeeze(-1)
    # Include native PPO clipping, dual clipping and MIS rejection independently.
    old = selected.detach()-torch.tensor([.1, .5, -1., 0., 3., .3, -.4, -30.])
    q = old-torch.tensor([.3, 3., 7.4, .67, 1.22, .37, 1.1, 1.]).log()
    advantage = torch.tensor([1., 1., -1., 0., -2., -.5, .8, 1.])
    token_loss, _ = math_utils.compute_policy_loss(old-selected, advantage, .2, .28, 10.)
    ratio = (old-q).exp()
    tis = ratio.clamp(.5, 2.)
    keep = torch.where(advantage > 0, (ratio >= .25) & (ratio <= 4.),
                       (ratio >= .5) & (ratio <= 2.)) if asymmetric else (ratio >= .5) & (ratio <= 2.)
    mis = torch.where(keep, ratio, 0.)
    entropy = -(logp.exp()*logp).sum(-1)
    expected = {"counterfactual_tis_pg": (tis*token_loss).sum()/19,
                "mis_minus_tis": ((mis-tis)*token_loss).sum()/19, "entropy": entropy.sum()}
    if asymmetric:
        symmetric = torch.where((ratio >= .5) & (ratio <= 2.), ratio, 0.)
        expected["asymmetric_minus_symmetric_mis"] = ((mis-symmetric)*token_loss).sum()/19
    actual = policy_observations.component_gradients(logits.detach()/temperature, samples, q, old,
                                                    advantage, args=args, denominator=19)
    for name, loss in expected.items():
        grad, = torch.autograd.grad(loss, logits, retain_graph=True)
        torch.testing.assert_close(actual[name], grad, atol=2e-7, rtol=2e-5)
        assert not actual[name].requires_grad


def _native_policy_function():
    # Exercise the real policy function without importing the optional Ray runtime.
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root/"miles/backends/training_utils/loss_hub/losses.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "policy_loss_function")
    def dense_probabilities(logits, **kwargs):
        logp = logits.log_softmax(-1)
        return {"log_probs": [logp[0].gather(-1, kwargs["unconcat_tokens"][0][1:, None]).squeeze(-1)],
                "entropy": [-(logp[0].exp()*logp[0]).sum(-1)]}
    namespace = {**vars(math_utils), **vars(cp_utils), "Namespace": Namespace, "Callable": Callable,
                 "RolloutBatch": dict, "policy_observations": policy_observations,
                 "vanilla_tis_function": corrections.vanilla_tis_function,
                 "get_log_probs_and_entropy": dense_probabilities,
                 "load_function": lambda name: corrections.icepop_function}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "native_policy", "exec"), namespace)
    return namespace["policy_loss_function"]


@pytest.mark.parametrize("all_rejected", [False, True])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("advantage", [-.5, 0., .5])
def test_real_policy_observations_preserve_loss_gradient_and_original_denominator(tmp_path, monkeypatch, all_rejected, asymmetric, advantage):
    monkeypatch.setenv("MILES_SOURCE_COMMIT", "cpu-test")
    logits = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(12), requires_grad=True)
    tokens = torch.tensor([0, 1, 2, 3, 4, 0])
    old = logits.detach().log_softmax(-1)[0].gather(-1, tokens[1:, None]).squeeze(-1)
    ratio = torch.tensor([.1, .5, 1., 2., 3.]) if not all_rejected else torch.full((5,), .1)
    q = old-ratio.log()
    mask = torch.tensor([1., 1., 0., 1., 1.])
    batch = dict(advantages=[torch.full((5,), advantage)], log_probs=[old], rollout_log_probs=[q],
        loss_masks=[mask], total_lengths=[6], response_lengths=[5], unconcat_tokens=[tokens],
        rollout_mask_sums=[mask.sum()], sample_indices=[10], weight_versions=[["version1"]],
        sfc_generation_log_probs=[q], sfc_sample_metadata=[dict(valid_tokens=4,
            first_valid_positions=[0, 1, 3, 4], theme="a", scenario="b", advantage=advantage)])
    function = _native_policy_function()
    reducer = cp_utils.get_sum_of_sample_mean([6], [5], [mask], calculate_per_token_loss=True)
    args = arguments(**(dict(mis_positive_low=.25, mis_positive_high=4.,
                            mis_negative_low=.5, mis_negative_high=2.) if asymmetric else {}))
    plain, _ = function(args, batch, logits, reducer)
    plain_grad, = torch.autograd.grad(plain, logits, retain_graph=True)
    args.sfc_diagnostics_dir = str(tmp_path)
    sfc_observations.configure_rollout(args, 0, "version1", {**batch, "rewards": [advantage]})
    observed, _ = function(args, batch, logits, reducer)
    observed_grad, = torch.autograd.grad(observed, logits)
    torch.testing.assert_close(plain, observed, rtol=0, atol=0)
    torch.testing.assert_close(plain_grad, observed_grad, rtol=0, atol=0)
    result = sfc_observations.finish(args, "NORMAL")
    frame, = read_frames(tmp_path/"paired_scores/update-000001/cp-0.bin")
    assert not frame["sfc_enabled"] and "correction" not in frame["data"]
    assert not any("centering_correction" in name or "head_mass" in name for name in result)
    assert result["sfc/original_valid_tokens"] == 4
    assert result["sfc/mis_retained_fraction"] == (0 if all_rejected else .75 if asymmetric and advantage > 0 else .5)
    assert result["sfc/applied_ess"] == 0 if all_rejected else result["sfc/applied_ess"] > 0
    assert result["sfc/pg_token_mean"] == pytest.approx(float(observed.detach()/4))


def test_asymmetric_mis_inclusive_boundaries_zero_advantages_and_detached_weights():
    args = arguments(mis_positive_low=.25, mis_positive_high=4., mis_negative_low=.5, mis_negative_high=2.)
    values = torch.tensor([.25, 4., .5, 2.])
    ratio = torch.cat([torch.nextafter(values, torch.zeros_like(values)), values,
                       torch.nextafter(values, torch.full_like(values, float("inf")))]).requires_grad_()
    advantages = torch.tensor([1., 1., -1., -1.]*3)
    expected_keep = torch.tensor([False, True, False, True, True, True, True, True, True, False, True, False])
    actual = corrections.mis_weights(args, ratio, advantages)
    torch.testing.assert_close(actual, torch.where(expected_keep, ratio.detach(), 0.))
    assert not actual.requires_grad
    # Zero advantages have no direct gradient, but retain the original diagnostic band.
    torch.testing.assert_close(corrections.mis_weights(args, values, torch.zeros_like(values)),
                               torch.tensor([0., 0., .5, 2.]))
    with pytest.raises(ValueError, match="aligned token advantages"):
        corrections.mis_weights(args, ratio)


@pytest.mark.parametrize("changes", [dict(mis_positive_low=.25),
    dict(mis_positive_low=float("nan"), mis_positive_high=4., mis_negative_low=.5, mis_negative_high=2.),
    dict(mis_positive_low=4., mis_positive_high=.25, mis_negative_low=.5, mis_negative_high=2.)])
def test_asymmetric_mis_invalid_bounds_fail_before_model_initialization(changes):
    with pytest.raises(ValueError, match="MIS advantage bounds"):
        validate_score_centering_args(arguments(**changes))


def test_screen_cadences_and_mis_validation():
    args = arguments(sfc_diagnostics_dir="/private/numeric")
    validate_score_centering_args(args)
    assert observation_updates(args, probe=True) == (0, 5, 10, 15, 20)
    assert observation_updates(args) == (10, 20)
    assert observation_updates(Namespace()) == (8, 32, 64)
    for changes in ({"tis_clip_low": 0.}, {"use_rollout_logprobs": True}, {"skip_actor_forward_only": True},
                    {"custom_tis_function_path": "other"}, {"entropy_coef": .01},
                    {"sfc_logit_audit_updates": [20, 10]}, {"sfc_fixed_probe_updates": [0, 0]}):
        with pytest.raises(ValueError):
            validate_score_centering_args(arguments(sfc_diagnostics_dir="/private/numeric", **changes))
