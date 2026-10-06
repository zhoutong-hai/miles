"""Opt-in aggregate receipts from the actor's actual dynamic packing plan.

This observer does not change iterators, create collectives, or persist samples.
One TP/CP/PP representative writes each effective-DP shard; consumers must
reconcile every shard rather than multiplying a representative's token count.
"""

import json
import math
import os
from pathlib import Path


def summarize_plan(*, lengths, response_lengths, partitions, microbatches, cp_size, tp_size, pad_multiplier):
    """Summarize the thd/allgather-CP plan without emitting sample indices."""
    if not lengths or any(type(value) is not int or value <= 0 for value in lengths):
        raise ValueError("Packing lengths must be positive integers")
    if len(response_lengths) != len(lengths) or any(type(response) is not int or not 0 <= response <= total for response, total in zip(response_lengths, lengths, strict=True)):
        raise ValueError("Response lengths do not match total lengths")
    if not microbatches or any(type(value) is not int or value <= 0 for value in microbatches):
        raise ValueError("Microbatch counts must be positive integers")
    if len(partitions) != sum(microbatches) or any(not indices for indices in partitions):
        raise ValueError("Packing plan does not match optimizer steps")
    indices = [index for partition in partitions for index in partition]
    if sorted(indices) != list(range(len(lengths))):
        raise ValueError("Packing plan must consume every sample exactly once")
    if any(type(value) is not int or value <= 0 for value in (cp_size, tp_size, pad_multiplier)):
        raise ValueError("Invalid packing parallelism or padding")
    multiple = cp_size * tp_size * pad_multiplier
    packed_lengths = [sum(lengths[index] for index in partition) for partition in partitions]
    padded_lengths = [math.ceil(length / multiple) * multiple for length in packed_lengths]
    return {
        "sample_count": len(lengths),
        "processed_tokens": sum(lengths),
        "response_tokens": sum(response_lengths),
        "microbatches_per_update": list(microbatches),
        "padded_tokens": sum(padded_lengths),
        "max_packed_tokens": max(packed_lengths),
        "max_padded_tokens_per_cp_rank": max(padded_lengths) // cp_size,
        "padding_multiple": multiple,
    }


def record_actor_packing(*, rollout_id, args, rollout_data, data_iterator, microbatches, parallel_state, timers):
    """Only the enabled canonical rank writes after actual actor training."""
    if os.environ.get("MILES_RECORD_THROUGHPUT", "0") != "1":
        return
    if any(group.rank != 0 for group in (parallel_state.tp, parallel_state.cp, parallel_state.pp)):
        return
    if not args.use_dynamic_batch_size or args.qkv_format != "thd" or not args.allgather_cp:
        raise ValueError("Throughput observer requires dynamic thd/allgather-CP packing")
    summary = summarize_plan(
        lengths=rollout_data["total_lengths"],
        response_lengths=rollout_data["response_lengths"],
        partitions=data_iterator[0].micro_batch_indices,
        microbatches=microbatches,
        cp_size=parallel_state.cp.size,
        tp_size=parallel_state.tp.size,
        pad_multiplier=args.data_pad_size_multiplier,
    )
    phase_seconds = {name: timers[name] for name in ("log_probs", "actor_train", "train")}
    if any(not math.isfinite(value) or value <= 0 for value in phase_seconds.values()):
        raise ValueError("Missing or invalid actual actor phase times")
    receipt = {
        "schema": 1,
        "rollout_id": rollout_id,
        "effective_dp_rank": parallel_state.effective_dp.rank,
        "effective_dp_size": parallel_state.effective_dp.size,
        "context_parallel_size": parallel_state.cp.size,
        "max_tokens_per_gpu": args.max_tokens_per_gpu,
        "unsliced_packing_target": args.max_tokens_per_gpu * parallel_state.cp.size,
        "phase_seconds": phase_seconds,
        "phase_timers_overlap": True,
        **summary,
    }
    root = Path(os.environ["MILES_RUN_DIR"]) / "performance" / "packing"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rollout-{rollout_id}-dp-{parallel_state.effective_dp.rank}.json"
    # A repeated attempt must be reconciled, never silently overwrite a result.
    with path.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, sort_keys=True, allow_nan=False)
        output.write("\n")
