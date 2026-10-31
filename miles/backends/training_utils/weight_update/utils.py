import hashlib
import time
from miles.backends.training_utils.k3_publication.shard_manifest import encode_sender_shards, is_routed_inner

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


def get_data_replica_rank_and_size(parallel_state: ParallelState, placement: WeightUpdatePlacement) -> tuple[int, int]:
    """(replica_rank, replica_size): this rank's index among the ranks that hold
    identical data after gathering per ``placement``, and their count. Collective."""
    if placement.gather_pp:
        return dist.get_rank(), dist.get_world_size()

    column_id = min(dist.get_process_group_ranks(parallel_state.pp.group))
    all_column_ids: list = [None] * dist.get_world_size()
    dist.all_gather_object(all_column_ids, column_id)
    return sorted(set(all_column_ids)).index(column_id), dist.get_world_size() // parallel_state.pp.size


def record_lora_checksums(bucket, checksums, *, mode: str, profile=None) -> None:
    if mode == "structural":
        from miles.backends.training_utils.k3_publication.publication import record_structural
        return record_structural(bucket, checksums, profile)
    if mode not in ("original", "direct"):
        raise ValueError("unknown diagnostic sender checksum mode")
    """Accumulate the sha256 checksums the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        t0 = time.perf_counter()
        if mode == "original":
            snapshot = tensor.detach().cpu().contiguous()
        else:
            # Keep this owned snapshot alive through synchronous hashing.
            # CPU-first contiguous conversion avoids a full extra HBM buffer.
            snapshot = tensor.detach().to(device="cpu", copy=True).contiguous()
        t1 = time.perf_counter()
        if mode == "original":
            wire = snapshot.flatten().view(torch.uint8).numpy().tobytes()
        else:
            wire = memoryview(snapshot.reshape(-1).view(torch.uint8).numpy())
        t2 = time.perf_counter()
        digest = hashlib.sha256(wire).hexdigest()
        if is_routed_inner(hf_key):
            digest = encode_sender_shards(hf_key, snapshot, digest)
        if profile is not None:
            profile["sender_copy_s"] += t1 - t0
            profile["sender_wire_materialize_s"] += t2 - t1
            profile["sender_sha_s"] += time.perf_counter() - t2
        checksums[lora_name][hf_key] = digest
