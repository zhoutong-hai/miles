"""Sealed fixed-prefix distribution probes, with no sampling or optimizer step."""

import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.parallel import get_parallel_state


def distribution_statistics(logits, *, vocab_size, group=None, topk=128):
    """Exact full-vocabulary entropy and top-k mass from vocabulary TP shards."""
    rank = dist.get_rank(group) if group is not None else 0
    work = logits.float()
    valid = torch.arange(work.size(-1), device=work.device) + rank*work.size(-1) < vocab_size
    work = work.masked_fill(~valid, -torch.inf)
    maximum = work.amax(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    exps = (work-maximum).exp()
    total = exps.sum(-1, keepdim=True)
    weighted_logits = (exps*torch.where(valid, work-maximum, 0)).sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(total, group=group)
        dist.all_reduce(weighted_logits, group=group)
    entropy = (total.log()-weighted_logits/total).squeeze(-1)
    local_top = exps.topk(min(topk, work.size(-1)), dim=-1).values
    if group is not None:
        parts = [torch.empty_like(local_top) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, local_top, group=group)
        local_top = torch.cat(parts, dim=-1)
    mass = local_top.topk(min(topk, vocab_size), dim=-1).values.sum(-1)/total.squeeze(-1)
    return entropy, mass


def _probe_callback(logits, *, args, unconcat_tokens, total_lengths, response_lengths,
                    with_entropy=False, max_seq_lens=None, non_loss_data=True):
    parallel = get_parallel_state()
    group = parallel.tp.group if parallel.tp.size > 1 else None
    result = {"probe_entropy": [], "probe_head_mass": [], "probe_count": []}
    chunks = _iter_response_chunks(logits, args=args, unconcat_tokens=unconcat_tokens,
                                  total_lengths=total_lengths, response_lengths=response_lengths,
                                  max_seq_lens=max_seq_lens, include_response_indices=False)
    for chunk, _, _ in chunks:
        if len(chunk):
            entropy, mass = distribution_statistics(chunk, vocab_size=args.vocab_size, group=group)
        else:
            entropy = mass = chunk.new_empty(0)
        # A response consists of one sealed target; only its owning CP rank
        # contributes a value. Forward aggregation restores original row order.
        result["probe_entropy"].append(entropy.detach().sum().reshape(1))
        result["probe_head_mass"].append(mass.detach().sum().reshape(1))
        result["probe_count"].append(chunk.new_tensor([len(chunk)]))
    return result


def _persist_probe(args, result, packet, source_hash, completed, elapsed):
    parallel = get_parallel_state()
    if not parallel.is_pp_last_stage or parallel.tp.rank != 0:
        return
    values = torch.stack([torch.cat(result[key]) for key in ("probe_entropy", "probe_head_mass", "probe_count")])
    if parallel.cp.size > 1:
        dist.all_reduce(values, group=parallel.cp.group)
    if parallel.cp.rank != 0:
        return
    if values.shape != (3, 64) or not torch.isfinite(values).all() or not (values[2] == 1).all():
        raise ValueError("Fixed probe must contain exactly64 unique finite original positions")
    directory = Path(args.sfc_diagnostics_dir)/"fixed_probe"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f"completed-{completed:06d}.json"
    output = {"schema_version": 1, "completed_update": completed, "probe_sha256": source_hash,
              "identities": packet["identities"], "entropy": values[0].cpu().tolist(),
              "top128_mass": values[1].cpu().tolist(), "forward_seconds": elapsed,
              "scope": "trainer full-distribution statistics on fixed sealed prefixes; no generation/update"}
    with path.open("x") as stream:
        json.dump(output, stream, allow_nan=False)


def run_fixed_probe(args, model, completed):
    if not getattr(args, "sfc_fixed_probe", None) or completed not in (0, 8, 16, 32, 48, 64):
        return
    # Megatron is optional in CPU tests and this module is imported only by the
    # owning Megatron actor; defer these framework imports to the actual probe.
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from megatron.core.utils import get_model_config
    from miles.backends.megatron_utils.model import forward_only
    from miles.backends.training_utils.data import get_data_iterator

    source = Path(args.sfc_fixed_probe)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != args.sfc_fixed_probe_sha256:
        raise ValueError("Fixed probe differs from the frozen execution source seal")
    packet = torch.load(source, map_location="cpu", weights_only=False)
    if len(packet["tokens"]) != 64 or len(set(map(tuple, packet["identities"]))) != 64:
        raise ValueError("Invalid sealed64-prefix probe")
    configuration = copy.copy(args)
    configuration.global_batch_size = 64
    data = {"tokens": [t.to(device=torch.cuda.current_device(), dtype=torch.long) for t in packet["tokens"]],
            "total_lengths": [len(t) for t in packet["tokens"]], "response_lengths": [1]*64,
            "loss_masks": [torch.ones(1, dtype=torch.int, device=torch.cuda.current_device()) for _ in range(64)]}
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    tracker = get_cuda_rng_tracker()
    rng_megatron = {k: v.clone() for k, v in tracker.get_states().items()}
    modes = [m.training for m in model]
    config = get_model_config(model[0])
    timers = config.timers
    start = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            iterator, microbatches = get_data_iterator(configuration, model, data)
            result = forward_only(_probe_callback, configuration, model, iterator, microbatches,
                                  rollout_id=completed, store_prefix="")
            torch.cuda.synchronize()
            _persist_probe(args, result, packet, source_hash, completed, time.perf_counter()-start)
    finally:
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
        tracker.set_states(rng_megatron)
        config.timers = timers
        for module, mode in zip(model, modes, strict=True):
            module.train(mode)
