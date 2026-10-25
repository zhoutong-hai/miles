"""Regression coverage for reference KL and finite score-centering gradients."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss.hub import score_centering_loss as loss_module
from miles.backends.training_utils.loss.hub.score_centering_loss import _regularization
from miles.backends.training_utils.loss.hub.score_centering.sampler_kl import sampler_kl_loss


def test_candidate_collection_retains_full_vocabulary_kl_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    logits = torch.tensor([[0.2, -0.3, 0.8, 0.4, 10.0]], requires_grad=True)
    args = Namespace(
        use_kl_loss=True,
        entropy_coef=0,
        observe_training_entropy=False,
        use_sampling_support_replay=True,
        vocab_size=4,
        rollout_temperature=0.7,
        log_probs_chunk_size=2,
        allgather_cp=False,
    )
    parallel = SimpleNamespace(tp=SimpleNamespace(size=1, group=None), cp=SimpleNamespace(size=1))
    monkeypatch.setattr(loss_module, "get_parallel_state", lambda: parallel)
    monkeypatch.setattr(
        loss_module,
        "_iter_response_chunks",
        lambda *unused, **kwargs: iter([(logits, torch.tensor([2]), [0])]),
    )
    batch = {
        "rollout_topk_token_ids": [torch.tensor([[2, 3, -1]])],
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
    }
    probabilities = loss_module._candidate_log_probs(args, batch, logits)
    full = (logits[:, :4] / 0.7).log_softmax(-1)[:, 2]
    support = (logits[:, [2, 3]] / 0.7).log_softmax(-1)[:, 0]
    torch.testing.assert_close(probabilities["kl_log_probs"][0], full)
    torch.testing.assert_close(probabilities["selected"][0][:, 0], support)
    actual = torch.autograd.grad(probabilities["kl_log_probs"][0].sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(actual, torch.autograd.grad(full.sum(), logits)[0])


@pytest.mark.parametrize("kind", ["k1", "k2", "k3", "low_var_kl"])
@pytest.mark.parametrize("unbiased", [False, True])
def test_reference_kl_matches_full_vocabulary_gradient(kind: str, unbiased: bool) -> None:
    logits = torch.tensor([0.2, -0.3, 0.8], requires_grad=True)
    full = logits.log_softmax(-1)[0:1]
    support = logits[:2].log_softmax(-1)[0:1]
    rollout = torch.tensor([-0.5])
    reference = torch.tensor([-1.3])
    args = Namespace(use_kl_loss=True, use_unbiased_kl=unbiased, kl_loss_type=kind, kl_loss_coef=0.1)
    loss, _ = _regularization(args, {"ref_log_probs": [reference]}, None, support, rollout, torch.tensor([True]), torch.mean, full)
    difference = full - reference
    if kind == "k1":
        expected = difference
    elif kind == "k2":
        expected = difference.square() / 2
    else:
        expected = (-difference).exp() - 1 + difference
    if unbiased:
        expected = expected * (support - rollout).exp()
    if kind == "low_var_kl":
        expected = expected.clamp(-10, 10)
    expected = 0.1 * expected.mean()
    torch.testing.assert_close(loss, expected)
    actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(actual_gradient, torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("coefficient", [0.0, 0.1])
@pytest.mark.parametrize("unbiased", [False, True])
def test_extreme_reference_kl_has_finite_loss_and_gradients(coefficient: float, unbiased: bool) -> None:
    log_probs = torch.tensor([-100.0, -1.0, -torch.inf, -0.5], requires_grad=True)
    reference = torch.tensor([0.0, -0.8, -torch.inf, -torch.inf])
    rollout = torch.tensor([-200.0, -1.1, -torch.inf, -torch.inf])
    args = Namespace(use_kl_loss=True, use_unbiased_kl=unbiased, kl_loss_type="k3", kl_loss_coef=coefficient)
    loss, metrics = _regularization(
        args,
        {"ref_log_probs": [reference]},
        None,
        log_probs,
        rollout,
        torch.tensor([True, True, False, True]),
        torch.mean,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])
    if coefficient:
        loss.backward()
        assert torch.isfinite(log_probs.grad).all()
        assert log_probs.grad[0] == log_probs.grad[2] == log_probs.grad[3] == 0
        assert log_probs.grad[1] != 0
    else:
        assert loss == 0


def test_identical_full_vocabulary_policies_have_zero_kl_under_replay() -> None:
    logits = torch.tensor([0.2, -0.3, 0.8], requires_grad=True)
    full = logits.log_softmax(-1)[0:1]
    support = logits[:2].log_softmax(-1)[0:1]
    args = Namespace(use_kl_loss=True, use_unbiased_kl=False, kl_loss_type="k2", kl_loss_coef=0.1)
    loss, metrics = _regularization(
        args,
        {"ref_log_probs": [full.detach()]},
        None,
        support,
        support.detach(),
        torch.tensor([True]),
        torch.mean,
        full,
    )
    assert metrics["kl_loss"] == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("seed", [0, 19, 81])
def test_sampler_kl_full_head_matches_true_dense_kl(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 9, dtype=torch.float64, generator=generator, requires_grad=True)
    sampler_logits = torch.randn(4, 9, dtype=torch.float64, generator=generator, requires_grad=True)
    logp, logq = logits.log_softmax(-1), sampler_logits.log_softmax(-1)
    active = torch.tensor([True, False, True, True])
    actual = sampler_kl_loss(logp, logq, torch.ones_like(logp, dtype=torch.bool), active)
    q = logq.detach().exp()
    expected = torch.where(active, (q * (logq.detach() - logp)).sum(-1), 0.0)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.sum(), (logits, sampler_logits), allow_unused=True, retain_graph=True)
    torch.testing.assert_close(grad[0], torch.autograd.grad(expected.sum(), logits)[0])
    assert grad[1] is None


def test_sampler_kl_modeled_tail_and_true_tail_error_are_distinct() -> None:
    logits = torch.tensor([[0.4, -0.3, 0.7, -0.8, 0.1]], dtype=torch.float64, requires_grad=True)
    p = logits.softmax(-1)
    head = torch.tensor([0, 2])
    q = p.detach().clone()
    q[:, head] = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    tail = torch.tensor([1, 3, 4])
    q[:, tail] *= 0.4 / q[:, tail].sum(-1, keepdim=True)
    actual = sampler_kl_loss(logits.log_softmax(-1)[:, head], q.log()[:, head], torch.ones(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    dense = (q * (q.log() - logits.log_softmax(-1))).sum()
    torch.testing.assert_close(actual.sum(), dense)
    grad = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(dense, logits, retain_graph=True)[0])
    # A different true tail with the same transported head is indistinguishable
    # to the approximation. Measure its error against a genuinely dense oracle.
    different_q = q.clone()
    different_q[:, tail] = torch.tensor([[0.05, 0.3, 0.05]], dtype=torch.float64)
    true_kl = (different_q * (different_q.log() - logits.log_softmax(-1))).sum()
    true_gradient = torch.autograd.grad(true_kl, logits)[0]
    assert (grad - true_gradient).abs().max() > 0.1


def test_sampler_kl_masks_padding_and_equal_policies() -> None:
    logits = torch.tensor([[0.4, -0.3, 0.7], [0.2, 0.1, -0.4]], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    logq = logp.detach().clone()
    head_p = torch.cat((logp, torch.full((2, 1), float("nan"))), -1)
    head_q = torch.cat((logq, torch.full((2, 1), float("nan"))), -1)
    head_p = torch.where(torch.tensor([[True], [False]]), head_p, float("nan"))
    head_q[1] = float("nan")
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    value = sampler_kl_loss(head_p, head_q, mask, torch.tensor([True, False]))
    torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-14, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(value.sum(), logits)[0], torch.zeros_like(logits), atol=1e-14, rtol=0)


@pytest.mark.parametrize("coefficient", [0.0, 0.03])
def test_sampler_kl_native_consumer_with_zero_advantages(monkeypatch: pytest.MonkeyPatch, coefficient: float) -> None:
    logits = torch.tensor([[0.4, -0.3, 0.7], [0.2, 0.1, -0.4]], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    logq = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype=torch.float64).log()
    ids = torch.arange(3).expand(2, -1)
    active = torch.tensor([True, False])
    monkeypatch.setattr(loss_module, "_candidate_log_probs", lambda *unused: {"selected": [torch.cat((logp[:, :1], logp), -1)]})
    monkeypatch.setattr(loss_module, "get_local_response_loss_masks", lambda *unused: [active])
    monkeypatch.setattr(loss_module, "_local_candidates", lambda args, batch, key, device: ids if key.endswith("ids") else logq)
    args = Namespace(sampler_kl_coef=coefficient, use_kl_loss=False, entropy_coef=0, score_centering_is="none", score_centering_tis_clip=2, score_centering_mis_low=0.5, score_centering_mis_high=5, qkv_format="thd")
    batch = {"total_lengths": [2], "response_lengths": [2], "loss_masks": [active], "rollout_log_probs": [logq[:, 0]], "advantages": [torch.zeros(2, dtype=torch.float64)]}
    loss, metrics = loss_module.score_centering_loss_function(args, batch, logits, lambda t: t[active].mean())
    expected = coefficient * (logq[0].exp() * (logq[0] - logp[0])).sum()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0])
    assert ("sampler_kl_loss" in metrics) == (coefficient != 0)


@pytest.mark.parametrize("coefficient", [-0.1, float("inf"), float("nan")])
def test_invalid_sampler_kl_coefficient_is_rejected(coefficient: float) -> None:
    from miles.utils.score_centering import validate_score_centering_args

    with pytest.raises(ValueError, match="sampler-kl-coef"):
        validate_score_centering_args(Namespace(loss_type="policy_loss", sampler_kl_coef=coefficient))


def test_sampler_kl_cannot_silently_enter_another_loss() -> None:
    from miles.utils.score_centering import validate_score_centering_args

    with pytest.raises(ValueError, match="requires.*score_centering"):
        validate_score_centering_args(Namespace(loss_type="policy_loss", sampler_kl_coef=0.03))
