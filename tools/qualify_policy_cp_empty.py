"""Model-free native TP4 x CP4 CUDA backward after the empty-score repair."""
from argparse import Namespace
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.nn

from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.math_utils import compute_policy_loss
from miles.backends.training_utils.loss_hub.corrections import icepop_function
from miles.backends.training_utils.parallel import set_parallel_state


def groups(rank):
    tp_group = cp_group = None
    for cp in range(4):
        group = dist.new_group([cp*4+i for i in range(4)])
        if rank//4 == cp: tp_group = group
    for tp in range(4):
        group = dist.new_group([cp*4+tp for cp in range(4)])
        if rank%4 == tp: cp_group = group
    return tp_group, cp_group


def case(rank, tp_group, cp_group, device, all_rejected, total, response, dtype):
    cp, tp = rank//4, rank%4
    set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=4,rank=cp,group=cp_group),
        tp=SimpleNamespace(size=4,rank=tp,group=tp_group)))
    width=total//4
    full=torch.randn(total,32,generator=torch.Generator(device=device).manual_seed(93),device=device).to(dtype).float()
    local=full[cp*width:(cp+1)*width,tp*8:(tp+1)*8].clone().to(dtype).requires_grad_()
    tokens=torch.arange(total,device=device)%32
    args=Namespace(qkv_format='thd',allgather_cp=True,true_on_policy_mode=False,
        rollout_temperature=1.,loss_type='policy_loss',log_probs_chunk_size=-1,vocab_size=32,
        tis_clip_low=.5,tis_clip=2.,mis_positive_low=.25,mis_positive_high=4.,
        mis_negative_low=.5,mis_negative_high=2.)
    result=get_log_probs_and_entropy(local[None],args=args,unconcat_tokens=[tokens],total_lengths=[total],
        response_lengths=[response],with_entropy=True,entropy_requires_grad=False)
    flag=torch.tensor([result['log_probs'][0].requires_grad],device=device)
    flags=[torch.empty_like(flag) for _ in range(4)]
    dist.all_gather(flags,flag,group=cp_group)
    assert all(bool(f.item()) for f in flags), 'CP owner lost differentiable redistribution'
    assert not result['entropy'][0].requires_grad
    old=full[total-response-1:total-1].log_softmax(-1).gather(-1,tokens[-response:,None]).squeeze(-1)
    values=[.1,8.,.1] if all_rejected else [.25,3.,.3,.5,4.,2.,.1,1.,5.]
    ratio=torch.tensor(values*((response+len(values)-1)//len(values)),device=device)[:response]
    q=old-ratio.log();ratio=(old-q).exp()
    advantages=torch.tensor([1.,-1.,0.]*((response+2)//3),device=device)[:response]
    keep=torch.where(advantages>0,(ratio>=.25)&(ratio<=4.),(ratio>=.5)&(ratio<=2.))
    mis=torch.where(keep,ratio,0.)
    score=result['log_probs'][0]
    local_advantages=slice_log_prob_with_cp(advantages,total,response)
    pg,_=compute_policy_loss(slice_log_prob_with_cp(old,total,response)-score,local_advantages,.2,.28,10.)
    weighted,masks,_=icepop_function(args,pg_loss=pg,train_log_probs=[slice_log_prob_with_cp(old,total,response)],
        rollout_log_probs=[slice_log_prob_with_cp(q,total,response)],loss_masks=[torch.ones_like(score)],
        advantages=local_advantages)
    assert masks[0].sum()==score.numel()
    (weighted.sum()/response+local.sum()*0).backward()
    reference=full.clone().requires_grad_()
    chosen=reference[total-response-1:total-1].log_softmax(-1).gather(-1,tokens[-response:,None]).squeeze(-1)
    native,_=compute_policy_loss(old-chosen,advantages,.2,.28,10.)
    (native.mul(mis).sum()/response).backward()
    # This pinned native fused CE deliberately returns BF16 gradients, even
    # for FP32 inputs. Mirror that unchanged backward boundary in the oracle.
    expected=reference.grad[cp*width:(cp+1)*width,tp*8:(tp+1)*8].to(torch.bfloat16).to(dtype)
    torch.testing.assert_close(local.grad,expected,atol=2e-6,rtol=.01 if dtype==torch.bfloat16 else 1e-4)
    assert torch.isfinite(local.grad).all()
    sentinel=torch.ones(1,device=device)
    dist.all_reduce(sentinel,group=cp_group);assert sentinel.item()==4
    dist.all_reduce(sentinel,group=tp_group);assert sentinel.item()==16
    return {'all_rejected':all_rejected,'input_dtype':str(dtype),'max_abs_gradient_error':float((local.grad-expected).abs().max()),
        'three_empty_contiguous_cp_owners':True,'native_tp_fused_ce':True,
        'original_token_denominator':response,'follow_on_collectives_passed':True}


def main():
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    try:
        assert dist.get_world_size()==16
        tp_group,cp_group=groups(rank)
        cases=[case(rank,tp_group,cp_group,device,v,total,response,dtype)
               for dtype in (torch.float32,torch.bfloat16)
               for total,response in ((16,3),(1024,155)) for v in (False,True)]
        output=Path(os.environ['E010_MIS_QUALIFICATION_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
        packet={'state':'CUDA_TP4_CP4_EMPTY_OWNER_NATIVE_BACKWARD_PASS_NOT_MODEL_ACCEPTANCE',
            'rank':rank,'source_commit':os.environ['MILES_SOURCE_COMMIT'],'device':'CUDA','backend':'NCCL',
            'model_work':False,'cases':cases}
        with (output/f'rank-{rank}.json').open('x') as stream:json.dump(packet,stream,indent=2,allow_nan=False)
        dist.barrier()
        if rank==0: print(json.dumps({'marker':'NATIVE_EMPTY_CP_BACKWARD_PASS','ranks':16,'cases_per_rank':8}))
    except BaseException:
        # A failing rank must surface its assertion before cleanup can wait
        # on peers already inside their next collective.
        import traceback
        traceback.print_exc()
        raise
    else:
        dist.destroy_process_group()


if __name__=='__main__':main()
