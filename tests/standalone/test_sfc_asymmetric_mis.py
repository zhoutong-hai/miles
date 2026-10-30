"""Independent dense oracles for sign-specific SFC MIS and sparse diagnostics."""
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss_hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.loss_hub.score_centering.importance_sampling import importance_sampling
from miles.backends.training_utils.loss_hub.sfc_audit_positions import select_positions
from miles.backends.training_utils.loss_hub.sfc_logit_audit import component_gradients

INTERVALS = (.25, 4., .5, 2.)


def oracle(logp, logq, ids, samples, advantage, intervals=INTERVALS):
    p, q = logp.exp(), logq.exp()
    positive = advantage > 0
    low = torch.where(positive, intervals[0], intervals[2]).to(logp)
    high = torch.where(positive, intervals[1], intervals[3]).to(logp)
    ph, qh = p.gather(-1, ids), q.gather(-1, ids)
    rh = ph / qh
    sampled_logp = logp.gather(-1, samples[:, None]).squeeze(-1)
    ratio = (sampled_logp-logq.gather(-1, samples[:, None]).squeeze(-1)).exp()
    weight = torch.where((ratio >= low) & (ratio <= high), ratio, 0).detach()
    rho = (1-qh.sum(-1)).clamp_min(1e-6)/(1-ph.sum(-1)).clamp_min(1e-6)
    alpha = ((rho >= 1/high) & (rho <= 1/low)).to(logp)
    residual = (torch.where((rh >= low[:, None]) & (rh <= high[:, None]), ph, 0)-alpha[:, None]*ph).detach()
    correction = (residual*logp.gather(-1, ids)).sum(-1)
    return -advantage*(weight*sampled_logp-correction), weight, correction


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('head_size', [3, 7])
def test_dense_loss_gradient_and_logit_components(dtype, head_size):
    gen = torch.Generator().manual_seed(238)
    logits = torch.randn(6, 7, generator=gen, dtype=dtype, requires_grad=True)
    logp = logits.log_softmax(-1)
    logq = torch.randn(6, 7, generator=gen, dtype=dtype).log_softmax(-1)
    ids = logq.topk(head_size, dim=-1).indices
    samples = torch.tensor([0, 2, 4, 6, 3, 1])
    advantage = torch.tensor([1., -1., 0., .3, -.7, 0.], dtype=dtype)
    common = dict(train_log_probs=logp.gather(-1, samples[:, None]).squeeze(-1),
                  train_head_log_probs=logp.gather(-1, ids),
                  rollout_log_probs=logq.gather(-1, samples[:, None]).squeeze(-1),
                  rollout_head_log_probs=logq.gather(-1, ids), head_mask=torch.ones_like(ids, dtype=torch.bool),
                  advantages=advantage, mode='mis', mis_low=.5, mis_high=2., mis_intervals=INTERVALS)
    actual, metrics = score_centering_loss(ScoreCenteringInputs(**common))
    expected, weights, _ = oracle(logp, logq, ids, samples, advantage)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(metrics.importance_weight, weights, atol=2e-6, rtol=1e-5)
    grad = torch.autograd.grad(actual.sum()/37, logits, retain_graph=True)[0]
    reference = torch.autograd.grad(expected.sum()/37, logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, reference, atol=2e-7, rtol=1e-5)
    assert not metrics.importance_weight.requires_grad and not metrics.correction.requires_grad
    components = component_gradients(logits, ids, logq.gather(-1, ids), samples,
                                    logq.gather(-1, samples[:, None]).squeeze(-1), advantage,
                                    mode='mis', beta=0., denominator=37, mis_intervals=INTERVALS)
    torch.testing.assert_close(components['unweighted_sfc_pg']+components['importance_change'], reference,
                               atol=2e-7, rtol=1e-5)
    symmetric, _, _ = oracle(logp, logq, ids, samples, advantage, (.5, 2., .5, 2.))
    symmetric_grad = torch.autograd.grad(symmetric.sum()/37, logits, retain_graph=True)[0]
    torch.testing.assert_close(components['asymmetric_minus_symmetric_mis'], reference-symmetric_grad,
                               atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(grad[advantage == 0], torch.zeros_like(grad[advantage == 0]))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_inclusive_sample_head_and_tail_boundaries(dtype):
    advantage = torch.tensor([1., 1., -1., -1., 0., 0.], dtype=dtype)
    sampling = importance_sampling('mis', tis_clip=2., mis_low=.5, mis_high=2.,
                                   mis_intervals=INTERVALS, advantages=advantage)
    ratio = torch.tensor([.25, 4., .5, 2., .5, 2.], dtype=dtype)
    actual_ratio = ratio.log().exp()
    low = torch.tensor([.25,.25,.5,.5,.5,.5],dtype=dtype)
    high = torch.tensor([4.,4.,2.,2.,2.,2.],dtype=dtype)
    torch.testing.assert_close(sampling.sample_weight(ratio.log()), torch.where((actual_ratio>=low)&(actual_ratio<=high),actual_ratio,0.))
    rho = ratio.reciprocal()
    torch.testing.assert_close(sampling.tail_scale(rho,dtype), torch.ones_like(rho))
    assert sampling.sample_weight(torch.full_like(ratio,1000.)).eq(0).all()


def test_rejected_sample_can_have_nonzero_centering_gradient():
    logits = torch.tensor([[-.2,-.7,1.2]],dtype=torch.float64,requires_grad=True)
    logp = logits.log_softmax(-1)
    # Rejected sample0, but supported head1 creates a retained correction.
    logq = torch.tensor([[.99,.005,.005]],dtype=torch.float64).log()
    ids = torch.tensor([[1]])
    loss, metrics = score_centering_loss(ScoreCenteringInputs(logp[:,0],logp[:,1:2],logq[:,0],logq[:,1:2],
        torch.ones_like(ids,dtype=torch.bool),torch.ones(1,dtype=torch.float64),mode='mis',mis_intervals=INTERVALS))
    assert metrics.importance_weight.item()==0 and abs(metrics.correction.item())>0
    assert torch.autograd.grad(loss.sum(),logits)[0].abs().sum()>0


def test_sign_aware_audit_excludes_zero_only_prefix_when_nonzero_exists():
    metadata=[{'advantage':0.,'first_valid_positions':list(range(32))},
              {'advantage':1.,'first_valid_positions':list(range(32))},
              {'advantage':-1.,'first_valid_positions':list(range(32))}]
    selected=select_positions(metadata,[0,1,2],sign_aware=True)
    assert len(set(selected))==32 and sum(i==1 for i,_ in selected)==sum(i==2 for i,_ in selected)==16
    assert select_positions(metadata,[0,1,2])==[(0,i) for i in range(32)]
