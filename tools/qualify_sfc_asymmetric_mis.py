"""Finite credential-free TP4 x CP4 SFC/MIS math and observation qualification."""
from argparse import Namespace
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub import sfc_observations
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import set_parallel_state
from tools.qualify_policy_cp_empty import groups


def dense_oracle(logp, q, head_ids, sampled_ids, advantage):
    """Independent full-vocabulary autograd reference for the modeled tail."""
    p = logp.exp()
    ph, qh = p.gather(-1, head_ids), q.gather(-1, head_ids)
    low, high = (.25, 4.) if advantage > 0 else (.5, 2.)
    sampled = logp.gather(-1, sampled_ids[:, None]).squeeze(-1)
    qs = q.gather(-1, sampled_ids[:, None]).squeeze(-1)
    ratio = (sampled.detach()-qs.log()).exp()
    weight = torch.where((ratio >= low) & (ratio <= high), ratio, 0.)
    head_ratio = (logp.gather(-1, head_ids).detach()-qh.log()).exp()
    rho = (1-qh.sum(-1)).clamp_min(1e-6)/(1-ph.detach().sum(-1)).clamp_min(1e-6)
    alpha = ((rho >= 1/high) & (rho <= 1/low)).to(logp)
    residual = torch.where((head_ratio >= low) & (head_ratio <= high), ph.detach(), 0.)-alpha[:, None]*ph.detach()
    correction = (residual*logp.gather(-1, head_ids)).sum(-1)
    return -advantage*(weight*sampled-correction)


def make_case(device, total, response, dtype, advantage):
    generator = torch.Generator(device=device).manual_seed(128)
    full = torch.randn(total, 256, generator=generator, device=device).to(dtype).float()
    tokens = torch.arange(total, device=device) % 256
    logp = full[-response-1:-1].log_softmax(-1)
    p = logp.exp()
    sampled_ids = tokens[-response:]
    sampled_p = p.gather(-1, sampled_ids[:, None]).squeeze(-1)
    ratios = torch.tensor([.125, .25, .5, 1., 2., 4., 8.], device=device).repeat((response+6)//7)[:response]
    sampled_q = sampled_p/ratios
    q = p*((1-sampled_q)/(1-sampled_p))[:, None]
    q.scatter_(-1, sampled_ids[:, None], sampled_q[:, None])
    head_q, head_ids = q.topk(128, dim=-1)
    return full, tokens, q, head_ids, head_q.log(), logp.gather(-1, sampled_ids[:, None]).squeeze(-1)


def case(rank, tp_group, cp_group, device, total, response, dtype, advantage, ordinal):
    cp, tp = rank//4, rank%4
    set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=4, rank=cp, group=cp_group),
        tp=SimpleNamespace(size=4, rank=tp, group=tp_group), effective_dp=SimpleNamespace(size=1),
        is_pp_last_stage=True))
    full, tokens, q, ids, head_q, cached = make_case(device, total, response, dtype, advantage)
    width = total//4
    local = full[cp*width:(cp+1)*width, tp*64:(tp+1)*64].clone().to(dtype).requires_grad_()
    root = Path(os.environ['E010_MIS_QUALIFICATION_OUTPUT'])/f'case-{ordinal:02d}'
    args = Namespace(qkv_format='thd', allgather_cp=True, true_on_policy_mode=False,
        rollout_temperature=1., loss_type='score_centering', score_centering_is='mis',
        score_centering_tis_clip=2., score_centering_mis_low=.5, score_centering_mis_high=2.,
        mis_positive_low=.25, mis_positive_high=4., mis_negative_low=.5, mis_negative_high=2.,
        tis_clip_low=0., tis_clip=2., log_probs_chunk_size=-1, vocab_size=256,
        entropy_coef=0., observe_training_entropy=True, use_kl_loss=False, sampler_kl_coef=0.,
        sfc_diagnostics_dir=str(root), sfc_logit_audit_updates=[10], sfc_fixed_probe_updates=[0,5,10,15,20],
        wandb_run_id='i028-cuda-qualification')
    q_sample = q.gather(-1, tokens[-response:, None]).squeeze(-1).log()
    def owned(value):
        return slice_log_prob_with_cp(value, total, response)
    metadata = [dict(valid_tokens=response, first_valid_positions=list(range(min(response,32))),
                     advantage=advantage, theme='qualification', scenario='synthetic', group_index=0)]
    batch = dict(unconcat_tokens=[tokens], total_lengths=[total], response_lengths=[response],
        loss_masks=[torch.ones(response, device=device)], advantages=[owned(torch.full((response,),advantage,device=device))],
        rollout_log_probs=[owned(q_sample)], rollout_topk_token_ids=[ids], rollout_topk_log_probs=[head_q],
        log_probs=[owned(cached)], sample_indices=[0], weight_versions=[['10' if response>=32 else '1']],
        rewards=[advantage], sfc_sample_metadata=metadata, sfc_generation_log_probs=[q_sample])
    completed = 10 if response>=32 else 1
    sfc_observations.configure_rollout(args,completed-1,completed,batch)
    reducer = get_sum_of_sample_mean([total],[response],batch['loss_masks'],calculate_per_token_loss=True)
    actual, _ = score_centering_loss_function(args,batch,local[None],reducer)
    (actual/response+local.sum()*0).backward()
    reference = full.clone().requires_grad_()
    expected = dense_oracle(reference[-response-1:-1].log_softmax(-1),q,ids,tokens[-response:],advantage).sum()/response
    expected.backward()
    target = reference.grad[cp*width:(cp+1)*width,tp*64:(tp+1)*64].to(dtype)
    tolerance = 2e-5 if dtype==torch.bfloat16 else 3e-6
    torch.testing.assert_close(local.grad,target,atol=tolerance,rtol=.02 if dtype==torch.bfloat16 else 3e-5)
    assert torch.isfinite(local.grad).all()
    summary = sfc_observations.finish(args,'NORMAL')
    if tp==0 and cp==0:
        assert summary['sfc/original_valid_tokens']==response and summary['sfc/candidate_count_mean']==128
        assert summary['sfc/mis_positive_low_bound']==.25 and summary['sfc/mis_negative_high_bound']==2.
    sentinel=torch.ones(1,device=device)
    dist.all_reduce(sentinel,group=cp_group);assert sentinel.item()==4
    dist.all_reduce(sentinel,group=tp_group);assert sentinel.item()==16
    return {'total':total,'response':response,'dtype':str(dtype),'advantage':advantage,
            'max_abs_gradient_error':float((local.grad-target).abs().max()),'original_denominator':response,
            'follow_on_collectives':True,'K128_transport_observer':True,'audit32':response>=32}


def main():
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device);dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    tp_group,cp_group=groups(rank)
    cases=[]
    for dtype in (torch.float32,torch.bfloat16):
        for total,response in ((16,3),(1024,155)):
            for advantage in (1.,-1.,0.):
                cases.append(case(rank,tp_group,cp_group,device,total,response,dtype,advantage,len(cases)))
    output=Path(os.environ['E010_MIS_QUALIFICATION_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
    with (output/f'rank-{rank}.json').open('x') as stream:
        json.dump({'state':'CUDA_TP4_CP4_SFC_ASYMMETRIC_MIS_PASS_NOT_MODEL_ACCEPTANCE','rank':rank,
            'source_commit':os.environ['MILES_SOURCE_COMMIT'],'model_work':False,'cases':cases},stream,allow_nan=False)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    main()
