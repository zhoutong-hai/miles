"""Finite model-free CUDA TP4 x CP4 parity, invoked by the E010 operator."""

from argparse import Namespace
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.loss_hub.score_centering.sampler_kl import sampler_kl_loss
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.loss_hub.sfc_logit_audit import component_gradients
from miles.backends.training_utils.loss_hub.sfc_probe import distribution_statistics
from miles.backends.training_utils.parallel import set_parallel_state


def _groups(rank):
    tp_group = cp_group = None
    for cp in range(4):
        group = dist.new_group([cp*4+i for i in range(4)])
        if rank//4 == cp:
            tp_group = group
    for tp in range(4):
        group = dist.new_group([cp*4+tp for cp in range(4)])
        if rank % 4 == tp:
            cp_group = group
    return tp_group, cp_group


def _case(rank, tp_group, cp_group, mode, beta):
    cp, tp = rank//4, rank % 4
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cuda").manual_seed(31)
    full = torch.randn(32, 20, generator=generator, device=device, requires_grad=True)
    local = full.detach()[cp*8:(cp+1)*8, tp*5:(tp+1)*5].clone().unsqueeze(0).requires_grad_()
    totals, responses = [5, 17, 6], [3, 4, 0]
    tokens = [torch.randint(17, (n,), generator=generator, device=device) for n in totals]
    masks = [torch.tensor(x, device=device) for x in ([1., 0., 1.], [1., 1., 0., 1.], [])]
    advantages = [torch.tensor(x, device=device) for x in ([1., 1., 0.], [-.5, 0., 1., .7], [])]
    qs = [torch.randn(n, 17, generator=generator, device=device).softmax(-1) for n in responses]
    heads = [q.topk(3, dim=-1) for q in qs]
    sampled = [q.log().gather(-1, t[-n:, None]).squeeze(-1) if n else q.new_empty(0)
               for q, t, n in zip(qs, tokens, responses, strict=True)]
    args = Namespace(loss_type="score_centering", qkv_format="thd", allgather_cp=True,
                     true_on_policy_mode=False, rollout_temperature=1., vocab_size=17, log_probs_chunk_size=2,
                     entropy_coef=0., observe_training_entropy=True, use_kl_loss=False,
                     sampler_kl_coef=beta, score_centering_is=mode, score_centering_tis_clip=2.,
                     score_centering_mis_low=.5, score_centering_mis_high=5.)
    batch = dict(unconcat_tokens=tokens, total_lengths=totals, response_lengths=responses, loss_masks=masks,
                 advantages=[slice_log_prob_with_cp(a, t, n) for a, t, n in zip(advantages, totals, responses)],
                 rollout_log_probs=[slice_log_prob_with_cp(q, t, n) for q, t, n in zip(sampled, totals, responses)],
                 rollout_topk_token_ids=[h.indices for h in heads], rollout_topk_log_probs=[h.values.log() for h in heads])
    reducer = get_sum_of_sample_mean(totals, responses, masks, calculate_per_token_loss=True)
    count = sum(m.sum() for m in masks)
    actual, metrics = score_centering_loss_function(args, batch, local, reducer)
    (actual/count).backward()
    expected = full.sum()*0
    start = 0
    for total, n, token, mask, advantage, q_sample, head in zip(totals, responses, tokens, masks, advantages, sampled, heads):
        if n:
            logp = full[start+total-n-1:start+total-1, :17].log_softmax(-1)
            chosen = logp.gather(-1, token[-n:, None]).squeeze(-1)
            candidates = logp.gather(-1, head.indices)
            losses, _ = score_centering_loss(ScoreCenteringInputs(chosen, candidates, q_sample, head.values.log(),
                                           torch.ones_like(candidates, dtype=torch.bool), advantage, mode=mode))
            kl = sampler_kl_loss(candidates, head.values.log(), torch.ones_like(candidates, dtype=torch.bool), mask.bool())
            expected = expected+((losses+beta*kl)*mask).sum()/count
        start += total
    expected.backward()
    torch.testing.assert_close(local.grad.squeeze(0), full.grad[cp*8:(cp+1)*8, tp*5:(tp+1)*5], atol=3e-6, rtol=3e-5)
    value = actual.detach().clone()
    dist.all_reduce(value, group=cp_group)
    torch.testing.assert_close(value/count, expected.detach(), atol=2e-6, rtol=2e-5)
    assert all(torch.isfinite(v) for v in metrics.values())


def _audit_and_probe(rank, tp_group):
    tp = rank % 4
    generator = torch.Generator(device="cuda").manual_seed(73)
    full = torch.randn(4, 20, generator=generator, device="cuda", requires_grad=True)
    local = full.detach()[:, tp*5:(tp+1)*5]
    logp = full[:, :17].log_softmax(-1)
    logq = torch.randn(4, 17, generator=generator, device="cuda").log_softmax(-1)
    qhead, ids = logq.topk(3)
    samples = torch.tensor([1, 4, 16, 7], device="cuda")
    qsample = logq.gather(-1, samples[:, None]).squeeze(-1)
    advantages = torch.tensor([1., -.7, 0., .2], device="cuda")
    entropy, mass = distribution_statistics(local, vocab_size=17, group=tp_group, topk=3)
    torch.testing.assert_close(entropy, -(logp.exp()*logp).sum(-1), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(mass, logp.exp().topk(3).values.sum(-1), atol=2e-6, rtol=2e-5)
    for mode in ("none", "tis"):
        actual = component_gradients(local, ids, qhead, samples, qsample, advantages,
                                     mode=mode, beta=.01, vocab_size=17, group=tp_group, denominator=21)
        head_p = logp.gather(-1, ids)
        selected = logp.gather(-1, samples[:, None]).squeeze(-1)
        common = (selected, head_p, qsample, qhead, torch.ones_like(ids, dtype=torch.bool), advantages)
        pg_none, _ = score_centering_loss(ScoreCenteringInputs(*common, mode="none"))
        pg, _ = score_centering_loss(ScoreCenteringInputs(*common, mode=mode))
        kl = .01*sampler_kl_loss(head_p, qhead, torch.ones_like(ids, dtype=torch.bool), torch.ones(4, dtype=torch.bool, device="cuda"))
        expected = {"unweighted_sfc_pg": pg_none.sum()/21, "importance_change": (pg-pg_none).sum()/21,
                    "weighted_sampler_kl": kl.sum()/21}
        for name, objective in expected.items():
            gradient = torch.autograd.grad(objective, full, retain_graph=True)[0]
            torch.testing.assert_close(actual[name], gradient[:, tp*5:(tp+1)*5], atol=2e-6, rtol=3e-5)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    rank = dist.get_rank()
    if dist.get_world_size() != 16:
        raise ValueError("Selected qualification binds16 GPUs: TP4 x CP4")
    tp_group, cp_group = _groups(rank)
    set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=4, rank=rank//4, group=cp_group),
                                      tp=SimpleNamespace(size=4, rank=rank % 4, group=tp_group)))
    for mode in ("none", "tis"):
        for beta in (0., .01):
            _case(rank, tp_group, cp_group, mode, beta)
    _audit_and_probe(rank, tp_group)
    torch.cuda.synchronize()
    path = Path(os.environ["E010_SFC_QUALIFICATION_OUTPUT"])
    path.mkdir(parents=True, exist_ok=True)
    receipt = {"state": "MODEL_FREE_CUDA_TP4_CP4_PARITY_PASS", "rank": rank,
               "source": os.environ["MILES_SOURCE_COMMIT"], "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(), "modes": ["none", "tis"], "beta": [0., .01],
               "compilation_disabled": os.environ.get("TORCH_COMPILE_DISABLE", "0"),
               "scope": "selected candidate/loss/gradient/probe/audit mechanics, not full-model dispatch or speed"}
    with (path/f"rank-{rank}.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
