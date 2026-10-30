"""Native TIS observation neutrality and independent derivative oracles."""

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.loss_hub import math_utils, policy_observations, sfc_observations
from miles.backends.training_utils.loss_hub.sfc_summary import read_frames
from miles.utils.score_centering import observation_importance_mode, validate_score_centering_args
from test_entropy_screens import _native_policy_function, arguments, local_parallel


@pytest.mark.parametrize("advantage", [-.5, 0., .5])
def test_native_tis_floor_cap_and_original_denominator_are_observational(tmp_path, monkeypatch, advantage):
    monkeypatch.setenv("MILES_SOURCE_COMMIT", "cpu-paired-reruns")
    logits = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(73), requires_grad=True)
    tokens = torch.tensor([0, 1, 2, 3, 4, 0])
    old = logits.detach().log_softmax(-1)[0].gather(-1, tokens[1:, None]).squeeze(-1)
    # Stay away from exponentiation boundary rounding in this neutrality check.
    q = old-torch.tensor([.1, .75, 1., 1.7, 3.]).log()
    mask = torch.tensor([1., 1., 0., 1., 1.])
    batch = dict(advantages=[torch.full((5,), advantage)], log_probs=[old], rollout_log_probs=[q],
        loss_masks=[mask], total_lengths=[6], response_lengths=[5], unconcat_tokens=[tokens],
        rollout_mask_sums=[mask.sum()], sample_indices=[10], weight_versions=[["version1"]],
        sfc_generation_log_probs=[q], sfc_sample_metadata=[dict(valid_tokens=4,
            first_valid_positions=[0, 1, 3, 4], theme="a", scenario="b", advantage=advantage)])
    args = arguments(custom_tis_function_path=None)
    function = _native_policy_function()
    reducer = cp_utils.get_sum_of_sample_mean([6], [5], [mask], calculate_per_token_loss=True)
    plain, _ = function(args, batch, logits, reducer)
    before, = torch.autograd.grad(plain, logits, retain_graph=True)
    args.sfc_diagnostics_dir = str(tmp_path)
    validate_score_centering_args(args)
    sfc_observations.configure_rollout(args, 0, "version1", {**batch, "rewards": [advantage]})
    observed, _ = function(args, batch, logits, reducer)
    after, = torch.autograd.grad(observed, logits)
    torch.testing.assert_close(plain, observed, atol=0, rtol=0)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    result = sfc_observations.finish(args, "NORMAL")
    frame, = read_frames(tmp_path/"paired_scores/update-000001/cp-0.bin")
    assert frame['importance_mode'] == 'policy_tis' and not frame['sfc_enabled']
    expected = (old-q).exp().clamp(.5, 2.)
    torch.testing.assert_close(frame['data']['weight'], expected)
    assert result['sfc/original_valid_tokens'] == 4
    assert result['sfc/applied_floor_fraction'] == .25
    assert result['sfc/applied_cap_fraction'] == .25
    assert result['sfc/tis_total_clipped_adv_all_fraction'] == .5
    assert result['sfc/tis_total_clipped_adv_all_tokens'] == 2
    assert result['sfc/pg_token_mean'] == pytest.approx(float(observed.detach()/4))
    assert 'sfc/mis_retained_fraction' not in result
    assert 'sfc/sampler_kl_raw_token_mean' not in result
    if advantage == 0:
        assert after.eq(0).all()
        assert result['sfc/zero_advantage_token_fraction'] == 1


@pytest.mark.parametrize('temperature', [.7, 1.])
def test_native_tis_audit_matches_dense_ppo_gradients(temperature):
    args = arguments(custom_tis_function_path=None, rollout_temperature=temperature)
    logits = torch.randn(8, 7, generator=torch.Generator().manual_seed(55), requires_grad=True)
    samples = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    logp = (logits[:, :5]/temperature).log_softmax(-1)
    selected = logp.gather(-1, samples[:, None]).squeeze(-1)
    old = selected.detach()-torch.tensor([.1, .5, -1., 0., 3., .3, -.4, -30.])
    q = old-torch.tensor([.3, 3., 7.4, .67, 1.22, .37, 1.1, 1.]).log()
    advantage = torch.tensor([1., 1., -1., 0., -2., -.5, .8, 1.])
    loss, _ = math_utils.compute_policy_loss(old-selected, advantage, .2, .28, 10.)
    ratio = (old-q).exp();tis = ratio.clamp(.5, 2.);mis = torch.where((ratio >= .5) & (ratio <= 2.), ratio, 0.)
    expected = {'unweighted_policy_pg': loss.sum()/19, 'actual_tis_pg': (tis*loss).sum()/19,
                'tis_change_from_unweighted': ((tis-1)*loss).sum()/19,
                'counterfactual_mis_minus_tis': ((mis-tis)*loss).sum()/19,
                'entropy': -(logp.exp()*logp).sum()}
    actual = policy_observations.component_gradients(logits.detach()/temperature, samples, q, old,
                                                    advantage, args=args, denominator=19)
    assert set(actual) == set(expected)
    for name, value in expected.items():
        grad, = torch.autograd.grad(value, logits, retain_graph=True)
        torch.testing.assert_close(actual[name], grad, atol=2e-7, rtol=2e-5)
        assert not actual[name].requires_grad


@pytest.mark.parametrize('loss,correction', [('policy_loss', None), ('score_centering', None)])
def test_pair_selects_nonzero_signs_instead_of_zero_only_prefix(tmp_path, monkeypatch, loss, correction):
    monkeypatch.setenv('MILES_SOURCE_COMMIT', 'cpu-paired-reruns')
    args = arguments(loss_type=loss, custom_tis_function_path=correction, score_centering_is='none',
                     sfc_audit_sign_aware=True, sfc_diagnostics_dir=str(tmp_path))
    metadata = [dict(valid_tokens=32, first_valid_positions=list(range(32))) for _ in range(3)]
    sfc_observations.configure_rollout(args, 9, '10', dict(sample_indices=[0,1,2],
        sfc_sample_metadata=metadata, rewards=[0.,1.,-1.]))
    context = sfc_observations._CONTEXT
    assert context['audit_selector'] == 'advantage_sign_v1'
    assert sum(i==1 for i,j in context['audit_positions']) == 16
    assert sum(i==2 for i,j in context['audit_positions']) == 16
    assert not any(i==0 for i,j in context['audit_positions'])
    assert observation_importance_mode(args) == ('policy_tis' if loss == 'policy_loss' else 'none')
