"""Independent dense-vocabulary oracles for three-distribution PPO+TIS centering."""

import math
from dataclasses import replace

import pytest
import torch

from miles.backends.training_utils.loss.hub.score_centering.policy import (
    PolicyScoreCenteringInputs,
    _active_policy_branch,
    policy_score_centering_correction,
)


def _effective_weight(p, old, q, advantage):
    ratio = p / old
    positive = advantage[..., None] >= 0
    gate = torch.where(positive, ratio <= 1.2, ratio >= 0.8)
    return (old / q).clamp_max(2) * ratio * gate


@pytest.mark.parametrize("k", [2, 7])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_partial_and_full_head_gradient_matches_dense_tail_oracle(k, sign):
    gen = torch.Generator().manual_seed(91)
    logits = torch.randn(3, 7, generator=gen, dtype=torch.float64, requires_grad=True)
    old = torch.randn(3, 7, generator=gen, dtype=torch.float64).softmax(-1)
    q = torch.randn(3, 7, generator=gen, dtype=torch.float64).softmax(-1)
    q_head, ids = q.topk(k, dim=-1)
    advantages = torch.tensor([sign, -sign, 0.0], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    loss, _ = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=logp.gather(-1, ids),
            old_head_log_probs=old.gather(-1, ids).log(),
            rollout_head_log_probs=q_head.log(),
            head_mask=torch.ones_like(ids, dtype=torch.bool),
            advantages=advantages,
        )
    )
    actual = torch.autograd.grad(loss.sum(), logits)[0]

    # Reconstruct both complete distributions independently. Their omitted tails
    # have the trainer's shape, with each distribution's own remaining mass.
    reference_logits = logits.detach().clone().requires_grad_()
    reference_logp = reference_logits.log_softmax(-1)
    p = reference_logp.detach().exp()
    p_tail = (1 - p.gather(-1, ids).sum(-1)).clamp_min(1e-6)
    old_hat = p * ((1 - old.gather(-1, ids).sum(-1)).clamp_min(1e-6) / p_tail)[:, None]
    q_hat = p * ((1 - q_head.sum(-1)).clamp_min(1e-6) / p_tail)[:, None]
    old_hat.scatter_(-1, ids, old.gather(-1, ids))
    q_hat.scatter_(-1, ids, q_head)
    mass = q_hat * _effective_weight(p, old_hat, q_hat, advantages)
    expected_loss = advantages * (mass.detach() * reference_logp).sum(-1)
    expected = torch.autograd.grad(expected_loss.sum(), reference_logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("advantage", [-1.7, 2.3])
def test_full_distribution_cancels_constant_advantage_drift(advantage):
    logits = torch.tensor([0.7, -1.2, 2.3, 0.0], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([0.45, 0.05, 0.2, 0.3], dtype=torch.float64)
    q = torch.tensor([0.1, 0.5, 0.15, 0.25], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    ratio = (logp - old.log()).exp()
    # The sampled term is the actual clipped surrogate, not the centering formula.
    native = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage) * (old / q).clamp_max(2)
    correction, _ = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=logp.expand(4, 4),
            old_head_log_probs=old.log().expand(4, 4),
            rollout_head_log_probs=q.log().expand(4, 4),
            head_mask=torch.ones(4, 4, dtype=torch.bool),
            advantages=torch.full((4,), advantage, dtype=torch.float64),
        )
    )
    gradient = torch.autograd.grad((q * (native + correction)).sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


@pytest.mark.parametrize("positive", [False, True])
def test_signed_gate_matches_clipped_surrogate_autograd_at_boundaries(positive):
    values = torch.tensor([-21.0, -20.0, math.log(0.8), 0.0, math.log(1.2), 20.0, 21.0], requires_grad=True)
    advantage = 1.0 if positive else -1.0
    ratio = values.clamp(-20, 20).exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    gradient = torch.autograd.grad(loss.sum(), values)[0]
    gate = _active_policy_branch(values.detach(), torch.tensor(positive), 0.2, 0.2)
    torch.testing.assert_close(gradient, -advantage * ratio.detach() * gate)


def test_matching_policies_have_zero_correction():
    logp = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64).log().requires_grad_()
    loss, metrics = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=logp[:, :2],
            old_head_log_probs=logp[:, :2].detach(),
            rollout_head_log_probs=logp[:, :2].detach(),
            head_mask=torch.ones(1, 2, dtype=torch.bool),
            advantages=torch.ones(1),
        )
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(metrics["sc_correction"], torch.zeros(1, dtype=torch.float64))


def test_old_policy_sampler_advantage_and_metrics_are_detached():
    train = torch.tensor([[0.6, 0.2]], requires_grad=True).log()
    old = torch.tensor([[0.2, 0.3]], requires_grad=True).log()
    q = torch.tensor([[0.1, 0.5]], requires_grad=True).log()
    advantage = torch.ones(1, requires_grad=True)
    loss, metrics = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=train,
            old_head_log_probs=old,
            rollout_head_log_probs=q,
            head_mask=torch.ones(1, 2, dtype=torch.bool),
            advantages=advantage,
        )
    )
    gradients = torch.autograd.grad(loss.sum(), (train, old, q, advantage), allow_unused=True)
    assert gradients[0] is not None and all(x is None for x in gradients[1:])
    assert all(not value.requires_grad and torch.isfinite(value).all() for value in metrics.values())


def test_padding_zero_advantage_and_tiny_tails_are_finite():
    train = torch.tensor([[0.0, -torch.inf], [torch.nan, torch.nan]], requires_grad=True)
    old = torch.tensor([[0.0, -torch.inf], [torch.nan, torch.nan]])
    q = torch.tensor([[-0.1, -torch.inf], [torch.nan, torch.nan]])
    loss, metrics = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=train,
            old_head_log_probs=old,
            rollout_head_log_probs=q,
            head_mask=torch.tensor([[True, False], [True, True]]),
            advantages=torch.tensor([1.0, 0.0]),
        )
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(train.grad).all()
    assert all(torch.isfinite(value).all() for value in metrics.values())
    torch.testing.assert_close(train.grad[1], torch.zeros(2))


def test_clipped_sample_can_still_have_centering_gradient():
    logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
    logp = logits.log_softmax(-1)
    old = torch.tensor([[0.1, 0.5, 0.4]])
    q = torch.tensor([[0.2, 0.4, 0.4]])
    ratio = logp[0, 0].exp() / old[0, 0]
    native = -torch.minimum(ratio, ratio.clamp(0.8, 1.2))
    native_gradient = torch.autograd.grad(native, logits, retain_graph=True)[0]
    correction, _ = policy_score_centering_correction(
        PolicyScoreCenteringInputs(
            train_head_log_probs=logp,
            old_head_log_probs=old.log(),
            rollout_head_log_probs=q.log(),
            head_mask=torch.ones(1, 3, dtype=torch.bool),
            advantages=torch.ones(1),
        )
    )
    gradient = torch.autograd.grad(native + correction.sum(), logits)[0]
    assert torch.count_nonzero(native_gradient) == 0
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    "update",
    [
        dict(tis_clip=0),
        dict(eps_clip=1),
        dict(eps_clip_high=-1),
        dict(eps=float("nan")),
        dict(old_head_log_probs=torch.ones(1, 1)),
    ],
)
def test_invalid_inputs_fail(update):
    args = PolicyScoreCenteringInputs(
        torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.ones(1)
    )
    with pytest.raises(ValueError):
        replace(args, **update)
