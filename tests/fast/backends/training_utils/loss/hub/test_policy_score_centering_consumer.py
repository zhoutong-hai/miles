"""Exercise the real native PPO/TIS consumer and old-forward capture on CPU.

Only the GPU-only selected-token cross-entropy kernel is replaced by dense
torch log_softmax. PPO, TIS, candidate scoring, masks, diagnostics and reduction
are the production functions.
"""

from functools import partial

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import logit_processors
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.hub.policy_score_centering_loss import old_policy_log_probs_with_candidates


@pytest.fixture
def dense_kernel(native_single_rank, monkeypatch):
    def calculate(logits, tokens, process_group, **kwargs):
        lp = (logits / kwargs["temperature"]).log_softmax(-1)
        entropy = -(lp.exp() * lp).sum(-1) if kwargs["with_entropy"] else None
        if entropy is not None and not kwargs["entropy_requires_grad"]:
            entropy = entropy.detach()
        return lp.gather(-1, tokens[:, None]), entropy

    monkeypatch.setattr(logit_processors, "calculate_log_probs_and_entropy", calculate)


def _fixture(tmp_path):
    args = make_args(
        policy_score_centering=True,
        true_on_policy_mode=False,
        use_tis=True,
        tis_clip=2.0,
        tis_clip_low=0.0,
        kl_coef=0.0,
        entropy_coef=0.0,
        observe_training_entropy=True,
        vocab_size=5,
        dump_details=tmp_path,
    )
    batch = dict(
        unconcat_tokens=[torch.tensor([0, 1, 2])],
        total_lengths=[3],
        response_lengths=[2],
        loss_masks=[torch.tensor([1.0, 1.0])],
        advantages=[torch.tensor([1.0, -1.0])],
        rollout_log_probs=[torch.tensor([0.2, 0.3]).log()],
        rollout_topk_token_ids=[torch.tensor([[0, 3], [1, 4]], dtype=torch.int32)],
        rollout_topk_log_probs=[torch.tensor([[0.4, 0.1], [0.2, 0.1]]).log()],
        sample_indices=[19],
    )
    old_logits = torch.tensor([[[0.2, -0.1, 0.8, 0.4, -0.6], [0.1, 0.7, -0.3, 0.5, -0.2], [0.0, 0.0, 0.0, 0.0, 0.0]]])
    callback = partial(
        logit_processors.get_log_probs_and_entropy,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=[3],
        response_lengths=[2],
    )
    with torch.no_grad():
        frozen = old_policy_log_probs_with_candidates(old_logits, args=args, batch=batch, callback=callback)
    batch.update(frozen)
    logits = torch.tensor(
        [[[0.8, -0.2, 0.1, 0.6, -0.4], [0.2, -0.3, 0.7, 0.1, -0.8], [0.0, 0.0, 0.0, 0.0, 0.0]]], requires_grad=True
    )
    return args, batch, old_logits, logits


def test_native_loss_plus_dense_tail_correction_and_private_diagnostics(dense_kernel, tmp_path):
    args, batch, old_logits, logits = _fixture(tmp_path)
    old_lp = old_logits[0, :2].log_softmax(-1)
    ids = batch["rollout_topk_token_ids"][0].long()
    torch.testing.assert_close(batch["old_policy_topk_log_probs"][0], old_lp.gather(-1, ids))
    assert not batch["old_policy_topk_log_probs"][0].requires_grad
    reduce = get_sum_of_sample_mean([3], [2], batch["loss_masks"])
    actual, metrics = policy_loss_function(args, batch, logits, reduce)
    actual_grad = torch.autograd.grad(actual, logits)[0]

    args.policy_score_centering = False
    reference_logits = logits.detach().clone().requires_grad_()
    baseline, _ = policy_loss_function(args, batch, reference_logits, reduce)
    lp = reference_logits[0, :2].log_softmax(-1)
    p = lp.detach().exp()
    old_head = old_lp.exp().gather(-1, ids)
    q_head = batch["rollout_topk_log_probs"][0].exp()
    p_tail = (1 - p.gather(-1, ids).sum(-1)).clamp_min(1e-6)
    old_hat = p * ((1 - old_head.sum(-1)) / p_tail)[:, None]
    q_hat = p * ((1 - q_head.sum(-1)) / p_tail)[:, None]
    old_hat.scatter_(-1, ids, old_head)
    q_hat.scatter_(-1, ids, q_head)
    ratio = p / old_hat
    adv = batch["advantages"][0]
    gate = torch.where(adv[:, None] >= 0, ratio <= 1.2, ratio >= 0.8)
    mass = q_hat * (old_hat / q_hat).clamp_max(2) * ratio * gate
    expected = baseline + reduce(adv * (mass.detach() * lp).sum(-1))
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, reference_logits)[0], atol=2e-6, rtol=2e-6)
    assert torch.isfinite(actual) and all(torch.isfinite(x) for x in metrics.values())
    assert metrics["entropy_loss"] > 0 and not metrics["entropy_loss"].requires_grad
    assert "sc_old_head_mass" in metrics
    dumps = [torch.load(p, weights_only=False) for p in (tmp_path / "policy_loss_debug").glob("*.pt")]
    centered = next(d for d in dumps if "score_centering" in d)
    torch.testing.assert_close(centered["score_centering"]["old_head_log_probs"], old_lp.gather(-1, ids))
    assert centered["sample_indices"] == [19]


def test_masked_observation_nan_does_not_enter_native_tis(dense_kernel, tmp_path):
    args, batch, _, logits = _fixture(tmp_path)
    batch["loss_masks"][0][1] = 0
    batch["log_probs"][0][1] = torch.nan
    batch["rollout_log_probs"][0][1] = torch.nan
    batch["rollout_topk_token_ids"][0][1] = -1
    batch["rollout_topk_log_probs"][0][1] = torch.nan
    reduce = get_sum_of_sample_mean([3], [2], batch["loss_masks"])
    loss, metrics = policy_loss_function(args, batch, logits, reduce)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros(5))
    assert all(torch.isfinite(x) for x in metrics.values())


def test_missing_old_candidate_capture_is_an_explicit_error(dense_kernel, tmp_path):
    args, batch, _, logits = _fixture(tmp_path)
    del batch["old_policy_topk_log_probs"]
    with pytest.raises(ValueError, match="frozen old-trainer forward"):
        policy_loss_function(args, batch, logits, torch.mean)
