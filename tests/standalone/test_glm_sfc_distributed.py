"""Real four-process CPU/Gloo TP4 and allgather-CP4 candidate/gradient checks."""

from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.score_centering import (
    ScoreCenteringInputs,
    score_centering_loss,
    selected_log_probs_and_entropy,
)
from miles.backends.training_utils.loss_hub.score_centering.sampler_kl import sampler_kl_loss
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import set_parallel_state


def _tp4(rank):
    generator = torch.Generator().manual_seed(10)
    full = torch.randn(3, 20, generator=generator, requires_grad=True)
    local = full.detach()[:, rank * 5:(rank + 1) * 5].clone().requires_grad_()
    ids = torch.tensor([[0, 6, 16, -1], [1, 12, 12, 5], [3, 9, 14, 16]])
    logp, entropy = selected_log_probs_and_entropy(local, ids, group=dist.group.WORLD, vocab_size=17,
                                                 temperature=0.7, chunk_size=2, with_entropy=True)
    reference = (full[:, :17] / 0.7).log_softmax(-1)
    chosen = reference.gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
    expected_entropy = -(reference.exp() * reference).sum(-1)
    weight = torch.randn(logp.shape, generator=generator)
    (logp.mul(weight).sum() - 0.2 * entropy.sum()).backward()
    (chosen.mul(weight).sum() - 0.2 * expected_entropy.sum()).backward()
    torch.testing.assert_close(logp, chosen, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(local.grad, full.grad[:, rank * 5:(rank + 1) * 5], atol=1e-6, rtol=1e-6)


def _cp4(rank, mode, coefficient):
    set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=4, rank=rank, group=dist.group.WORLD),
                                      tp=SimpleNamespace(size=1, group=None)))
    args = Namespace(qkv_format="thd", allgather_cp=True, true_on_policy_mode=False, rollout_temperature=1.0,
                     vocab_size=7, log_probs_chunk_size=2, entropy_coef=0.0, observe_training_entropy=True,
                     use_kl_loss=False, sampler_kl_coef=coefficient, score_centering_is=mode,
                     score_centering_tis_clip=2.0, score_centering_mis_low=0.5, score_centering_mis_high=5.0)
    generator = torch.Generator().manual_seed(27)
    full = torch.randn(32, 8, generator=generator, requires_grad=True)
    local = full.detach()[rank * 8:(rank + 1) * 8].clone().unsqueeze(0).requires_grad_()
    totals, responses = [5, 17, 6], [3, 4, 0]
    tokens = [torch.randint(7, (n,), generator=generator) for n in totals]
    masks = [torch.tensor([1., 0, 1]), torch.tensor([1., 1, 0, 1]), torch.empty(0)]
    advantages = [torch.tensor([1., 1, 0]), torch.tensor([-0.5, 0, 1., 0.7]), torch.empty(0)]
    qs = [torch.randn(n, 7, generator=generator).softmax(-1) for n in responses]
    heads = [q.topk(3, dim=-1) for q in qs]
    sampled = [q.log().gather(-1, t[-n:, None]).squeeze(-1) if n else torch.empty(0)
               for q, t, n in zip(qs, tokens, responses)]
    batch = dict(unconcat_tokens=tokens, total_lengths=totals, response_lengths=responses, loss_masks=masks,
                 advantages=[slice_log_prob_with_cp(a, t, n) for a, t, n in zip(advantages, totals, responses)],
                 rollout_log_probs=[slice_log_prob_with_cp(q, t, n) for q, t, n in zip(sampled, totals, responses)],
                 rollout_topk_token_ids=[h.indices for h in heads], rollout_topk_log_probs=[h.values.log() for h in heads])
    reduce = get_sum_of_sample_mean(totals, responses, masks, calculate_per_token_loss=True)
    count = sum(m.sum() for m in masks)
    actual, log = score_centering_loss_function(args, batch, local, reduce)
    (actual / count).backward()
    expected = full.sum() * 0
    start = 0
    for total, n, token, mask, advantage, q_sample, head in zip(totals, responses, tokens, masks, advantages, sampled, heads):
        if n:
            logp = full[start + total - n - 1:start + total - 1, :7].log_softmax(-1)
            selected = logp.gather(-1, token[-n:, None]).squeeze(-1)
            candidates = logp.gather(-1, head.indices)
            losses, _ = score_centering_loss(ScoreCenteringInputs(
                train_log_probs=selected, train_head_log_probs=candidates, rollout_log_probs=q_sample,
                rollout_head_log_probs=head.values.log(), head_mask=torch.ones_like(candidates, dtype=torch.bool),
                advantages=advantage, mode=mode,
            ))
            kl = sampler_kl_loss(candidates, head.values.log(), torch.ones_like(candidates, dtype=torch.bool), mask.bool())
            expected = expected + ((losses + coefficient * kl) * mask).sum() / count
        start += total
    expected.backward()
    torch.testing.assert_close(local.grad.squeeze(0), full.grad[rank * 8:(rank + 1) * 8], atol=2e-6, rtol=2e-5)
    total = actual.detach().clone()
    dist.all_reduce(total)
    torch.testing.assert_close(total / count, expected.detach(), atol=1e-6, rtol=1e-6)
    assert all(torch.isfinite(value) for value in log.values())


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        _tp4(rank)
        for mode in ("none", "tis"):
            for coefficient in (0.0, 0.03):
                _cp4(rank, mode, coefficient)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU/Gloo unavailable")
def test_tp4_and_allgather_cp4_native_sfc_kl(tmp_path):
    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=4, join=True)
