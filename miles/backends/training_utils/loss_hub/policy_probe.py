"""Sealed fixed-prefix distribution probes, with no sampling or optimizer step."""

import copy
import hashlib
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.loss_hub.policy_metrics import publish_metrics, write_json


from miles.backends.training_utils.loss_hub.policy_audit import distribution_statistics

_COMPLETED = set()


def _probe_callback(
    logits,
    *,
    args,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    with_entropy=False,
    max_seq_lens=None,
    non_loss_data=True,
):
    parallel = get_parallel_state()
    group = parallel.tp.group if parallel.tp.size > 1 else None
    result = {"probe_entropy": [], "probe_head_mass": [], "probe_count": []}
    chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=False,
    )
    for chunk, _, _ in chunks:
        if len(chunk):
            entropy, mass, _, _ = distribution_statistics(
                chunk, vocab_size=args.vocab_size, group=group, temperature=args.rollout_temperature
            )
        else:
            entropy = mass = chunk.new_empty(0)
        # A response consists of one sealed target; only its owning CP rank
        # contributes a value. Forward aggregation restores original row order.
        result["probe_entropy"].append(entropy.detach().sum().reshape(1))
        result["probe_head_mass"].append(mass.detach().sum().reshape(1))
        result["probe_count"].append(chunk.new_tensor([len(chunk)]))
    return result


def _persist_probe(args, result, packet, source_hash, completed, elapsed, selected):
    parallel = get_parallel_state()
    if not parallel.is_pp_last_stage or parallel.tp.rank != 0:
        return
    values = torch.stack([torch.cat(result[key]) for key in ("probe_entropy", "probe_head_mass", "probe_count")]).cpu()
    parts = [None] * parallel.effective_dp.size
    payload = {"positions": selected, "values": values, "seconds": elapsed}
    if parallel.effective_dp.size > 1:
        dist.all_gather_object(parts, payload, group=parallel.effective_dp.group)
    else:
        parts = [payload]
    if parallel.effective_dp.rank != 0:
        return
    all_positions = [p for part in parts for p in part["positions"]]
    values = torch.cat([part["values"] for part in parts], dim=1)
    if sorted(all_positions) != list(range(64)) or not torch.isfinite(values).all() or not (values[2] == 1).all():
        raise ValueError("Fixed probe must cover exactly64 unique global positions")
    order = torch.tensor(all_positions).argsort()
    values = values[:, order]
    directory = Path(args.policy_diagnostics_dir) / "fixed_probe"
    directory.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "completed_update": completed,
        "probe_sha256": source_hash,
        "identities": packet["identities"],
        "entropy": values[0].tolist(),
        "top128_mass": values[1].tolist(),
        "source": os.environ["MILES_SOURCE_COMMIT"],
        "forward_max_rank_seconds": max(part["seconds"] for part in parts),
    }
    write_json(directory / f"completed-{completed:06d}.json", output)
    metrics = {
        "policy/step": completed,
        "policy/probe/entropy": float(values[0].mean()),
        "policy/probe/top128_mass": float(values[1].mean()),
        "policy/probe/positions": 64,
        "policy/probe/seconds": output["forward_max_rank_seconds"],
    }
    write_json(directory / f"metrics-{completed:06d}.json", metrics)
    publish_metrics(args, metrics, directory / f"tracking-{completed:06d}.json")


def run_fixed_probe(args, model, completed):
    if not getattr(args, "policy_fixed_probe", None) or completed not in args.policy_probe_updates:
        return
    # Megatron is optional in CPU tests and this module is imported only by the
    # owning Megatron actor; defer these framework imports to the actual probe.
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from megatron.core.utils import get_model_config
    from miles.backends.megatron_utils.model import forward_only
    from miles.backends.training_utils.data import get_data_iterator

    parallel = get_parallel_state()
    if parallel.cp.size != 1 or 64 % parallel.effective_dp.size:
        raise ValueError("Selected fixed probe requires CP1 and an even DP partition of64")
    if completed in _COMPLETED:
        return
    source = Path(args.policy_fixed_probe)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != args.policy_fixed_probe_sha256:
        raise ValueError("Fixed probe differs from the frozen execution source seal")
    packet = torch.load(source, map_location="cpu", weights_only=False)
    if len(packet["tokens"]) != 64 or len(set(map(tuple, packet["identities"]))) != 64:
        raise ValueError("Invalid sealed64-prefix probe")
    selected = list(range(parallel.effective_dp.rank, 64, parallel.effective_dp.size))
    configuration = copy.copy(args)
    configuration.global_batch_size = 64
    configuration.num_steps_per_rollout = 1
    data = {
        "tokens": [
            t.to(device=torch.cuda.current_device(), dtype=torch.long) for i in selected for t in [packet["tokens"][i]]
        ],
        "total_lengths": [len(packet["tokens"][i]) for i in selected],
        "response_lengths": [1] * len(selected),
        "loss_masks": [torch.ones(1, dtype=torch.int, device=torch.cuda.current_device()) for _ in selected],
    }
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    tracker = get_cuda_rng_tracker()
    rng_megatron = {k: v.clone() for k, v in tracker.get_states().items()}
    from miles.backends.megatron_utils.actor import all_replay_managers

    replay_stages = [(manager, manager.stage) for manager in all_replay_managers if manager.enabled]
    for manager, _ in replay_stages:
        manager.stage = "fallthrough"
    modes = [m.training for m in model]
    config = get_model_config(model[0])
    timers = config.timers
    start = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            iterator, microbatches = get_data_iterator(configuration, model, data)
            result = forward_only(
                _probe_callback, configuration, model, iterator, microbatches, rollout_id=completed, store_prefix=""
            )
            torch.cuda.synchronize()
            _persist_probe(args, result, packet, source_hash, completed, time.perf_counter() - start, selected)
    finally:
        for manager, stage in replay_stages:
            manager.stage = stage
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
        tracker.set_states(rng_megatron)
        config.timers = timers
        for module, mode in zip(model, modes, strict=True):
            module.train(mode)
    _COMPLETED.add(completed)
