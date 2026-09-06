import hashlib
import json

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


def record_lora_checksums(bucket, checksums) -> None:
    """Accumulate the sha256 manifest the engines verify at end_weight_update."""
    for name, tensor in bucket:
        if ":" not in name:
            continue
        lora_name, hf_key = name.split(":", 1)
        digest = hashlib.sha256(
            tensor.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        checksums[lora_name][hf_key] = digest


def record_initial_lora_zero_b_evidence(bucket, inventory: dict[str, list]) -> None:
    """Record the exported A names and complete byte-level B evidence."""
    for name, tensor in bucket:
        if ".lora_A." in name:
            inventory.setdefault("a_names", []).append(name)
            continue
        if ".lora_B." not in name:
            continue
        value = tensor.detach()
        raw = value.cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
        inventory.setdefault("b_tensors", []).append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "numel": value.numel(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "finite": bool(torch.isfinite(value).all().item()),
                "zero": bool(torch.count_nonzero(value).item() == 0),
            }
        )


def validate_initial_lora_zero_b_evidence(inventory: dict[str, list]) -> dict:
    """Fail closed unless every exported LoRA A has one finite, zero B peer."""
    a_names = inventory.get("a_names", [])
    b_tensors = inventory.get("b_tensors", [])
    b_names = [entry["name"] for entry in b_tensors]
    if not b_names:
        raise RuntimeError("initial LoRA source gate found no exported lora_B tensors")
    if len(a_names) != len(set(a_names)) or len(b_names) != len(set(b_names)):
        raise RuntimeError("initial LoRA source gate found duplicate LoRA tensor names")

    expected_b_names = {name.replace(".lora_A.", ".lora_B.", 1) for name in a_names}
    observed_b_names = set(b_names)
    if expected_b_names != observed_b_names:
        raise RuntimeError(
            "initial LoRA source gate A/B coverage mismatch: "
            f"missing_b={sorted(expected_b_names - observed_b_names)} "
            f"unexpected_b={sorted(observed_b_names - expected_b_names)}"
        )

    nonfinite = sorted(entry["name"] for entry in b_tensors if not entry["finite"])
    nonzero = sorted(entry["name"] for entry in b_tensors if not entry["zero"])
    if nonfinite:
        raise RuntimeError(f"initial LoRA source gate found non-finite lora_B tensors: {nonfinite}")
    if nonzero:
        raise RuntimeError(f"initial LoRA source gate found nonzero lora_B tensors: {nonzero}")

    tensor_evidence = sorted(b_tensors, key=lambda entry: entry["name"])
    encoded = json.dumps(tensor_evidence, sort_keys=True, separators=(",", ":")).encode()
    return {
        "tensor_count": len(tensor_evidence),
        "total_numel": sum(entry["numel"] for entry in tensor_evidence),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "tensors": tensor_evidence,
    }
