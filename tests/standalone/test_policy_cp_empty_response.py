"""Native policy scoring must retain CP backward collectives on empty owners."""
from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.multiprocessing as mp

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils.loss_hub import math_utils
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import set_parallel_state


@pytest.mark.parametrize("true_on_policy", [False, True])
@pytest.mark.parametrize("entropy_requires_grad", [False, True])
def test_empty_native_scores_preserve_graph_and_entropy_observation_mode(true_on_policy, entropy_requires_grad):
    logits = torch.empty(0, 7, requires_grad=True)
    logp, entropy = math_utils.calculate_log_probs_and_entropy(
        logits, torch.empty(0, dtype=torch.long), None, with_entropy=True,
        entropy_requires_grad=entropy_requires_grad, true_on_policy=true_on_policy)
    assert logp.shape == entropy.shape == (0,)
    assert logp.requires_grad
    assert entropy.requires_grad == entropy_requires_grad
    logp.sum().backward()
    assert logits.grad.shape == logits.shape


def run_cp_backward(rank, group, device):
    """Production response iterator/redistribution, with3 empty contiguous owners.

    The CE implementation alone is dense here; CUDA qualification uses native
    Megatron fused TP CE. This fixture targets the shared empty-score graph edge.
    """
    singleton = None
    for member in range(4):
        candidate = dist.new_group([member])
        if rank == member:
            singleton = candidate
    set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=4, rank=rank, group=group),
        tp=SimpleNamespace(size=1, rank=0, group=singleton)))
    original = math_utils.compute_log_probs
    math_utils.compute_log_probs = lambda logits, tokens, process_group: logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
    try:
        full = torch.randn(16, 7, generator=torch.Generator(device=device).manual_seed(93), device=device)
        local = full[rank*4:(rank+1)*4].clone().requires_grad_()
        tokens = torch.arange(16, device=device) % 7
        args = Namespace(qkv_format="thd", allgather_cp=True, true_on_policy_mode=False,
            rollout_temperature=1., loss_type="policy_loss", log_probs_chunk_size=-1, vocab_size=7)
        result = get_log_probs_and_entropy(local[None], args=args, unconcat_tokens=[tokens],
            total_lengths=[16], response_lengths=[3], with_entropy=True, entropy_requires_grad=False)
        flags = torch.tensor([result['log_probs'][0].requires_grad], device=device)
        gathered = [torch.empty_like(flags) for _ in range(4)]
        dist.all_gather(gathered, flags, group=group)
        assert all(bool(flag.item()) for flag in gathered), "Empty native scores dropped the CP backward collective"
        selected = full[12:15].log_softmax(-1).gather(-1, tokens[-3:, None]).squeeze(-1)
        old = selected.detach()
        q = old-torch.tensor([.1, 1., 3.], device=device).log()
        ratio = (old-q).exp()
        mis = torch.where((ratio >= .5) & (ratio <= 2.), ratio, 0.)
        weight = cp_utils.slice_log_prob_with_cp(mis,16,3)
        score = result['log_probs'][0]
        pg, _ = math_utils.compute_policy_loss(cp_utils.slice_log_prob_with_cp(old,16,3)-score,
            torch.ones_like(score), .2, .28, 10.)
        # Same original-token denominator, including MIS-rejected tokens.
        ((pg*weight).sum()/3 + local.sum()*0).backward()
        reference = full.clone().requires_grad_()
        reference_score = reference[12:15].log_softmax(-1).gather(-1,tokens[-3:,None]).squeeze(-1)
        reference_pg, _ = math_utils.compute_policy_loss(old-reference_score,torch.ones(3,device=device),.2,.28,10.)
        (reference_pg.mul(mis).sum()/3).backward()
        torch.testing.assert_close(local.grad, reference.grad[rank*4:(rank+1)*4], atol=2e-7, rtol=2e-5)
        # A follow-on collective catches a surviving ordering skew immediately.
        sentinel = torch.ones(1,device=device)
        dist.all_reduce(sentinel,group=group)
        assert sentinel.item() == 4
    finally:
        math_utils.compute_log_probs = original


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=4,timeout=timedelta(seconds=30))
    try:
        run_cp_backward(rank,dist.group.WORLD,torch.device('cpu'))
    finally:
        dist.destroy_process_group()


def test_native_cp4_empty_owners_backward_matches_dense_mis(tmp_path):
    mp.spawn(worker,args=((tmp_path/'gloo').as_uri(),),nprocs=4,join=True)
