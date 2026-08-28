#!/usr/bin/env python3
"""Audit an exact Kimi-K3 DCP census and TP32/EP128 dry-load receipt.

The target receipt must be emitted by the exact 128-rank dry model/load job. It
is intentionally not synthesized from the 2.8T headline parameter count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


EXPECTED_TOPOLOGY = {
    "world_size": 128,
    "tensor_parallel": 32,
    "pipeline_parallel": 1,
    "context_parallel": 1,
    "expert_parallel": 128,
    "expert_tensor_parallel": 1,
    "attention_data_parallel": 4,
    "expert_data_parallel": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def logical_census(checkpoint: Path) -> tuple[dict[str, dict], object]:
    metadata = FileSystemReader(checkpoint).read_metadata()
    census = {}
    for name, item in metadata.state_dict_metadata.items():
        if not isinstance(item, TensorStorageMetadata):
            continue
        shape = list(item.size)
        count = math.prod(shape)
        census[name] = {
            "shape": shape,
            "dtype": str(item.properties.dtype),
            "bytes": count * dtype_bytes(item.properties.dtype),
        }
    if not census:
        raise RuntimeError("DCP metadata contains no tensor entries")
    return census, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-receipt", type=Path, required=True)
    parser.add_argument("--hbm-gib", type=float, default=141.0)
    parser.add_argument("--headroom-gib", type=float, default=15.0)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    receipt_path = args.target_receipt.resolve()
    if not checkpoint.is_dir() or not receipt_path.is_file():
        raise RuntimeError("checkpoint directory and target receipt must exist")

    census, metadata = logical_census(checkpoint)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("topology") != EXPECTED_TOPOLOGY:
        raise RuntimeError("target receipt topology does not equal TP32/PP1/CP1/EP128/ETP1")
    dry_load = receipt.get("dry_load", {})
    if dry_load.get("status") != "PASS" or dry_load.get("ranks_completed") != 128:
        raise RuntimeError("exact 128-rank shared-checkpoint dry load did not pass")
    if dry_load.get("tensor_inventory_equal") is not True or dry_load.get("finite") is not True:
        raise RuntimeError("dry-load tensor inventory/value checks are incomplete")

    recorded = receipt.get("logical_tensors")
    if recorded != census:
        missing = sorted(census.keys() - (recorded or {}).keys())[:5] if isinstance(recorded, dict) else []
        unexpected = sorted((recorded or {}).keys() - census.keys())[:5] if isinstance(recorded, dict) else []
        raise RuntimeError(f"receipt DCP census differs from checkpoint metadata: missing={missing} unexpected={unexpected}")

    rank_dcp = receipt.get("rank_resident_dcp_bytes")
    rank_extra = receipt.get("rank_runtime_extra_peak_bytes")
    if not isinstance(rank_dcp, list) or not isinstance(rank_extra, list) or len(rank_dcp) != 128 or len(rank_extra) != 128:
        raise RuntimeError("receipt must contain 128 exact DCP and runtime-extra byte counts")
    if any(not isinstance(value, int) or value <= 0 for value in rank_dcp):
        raise RuntimeError("per-rank DCP byte census is missing or non-positive")
    if any(not isinstance(value, int) or value < 0 for value in rank_extra):
        raise RuntimeError("per-rank runtime-extra peak census is missing")

    hbm_bytes = int(args.hbm_gib * 1024**3)
    required_free = int(args.headroom_gib * 1024**3)
    predicted_peak = [base + extra for base, extra in zip(rank_dcp, rank_extra, strict=True)]
    headroom = [hbm_bytes - peak for peak in predicted_peak]
    if min(headroom) < required_free:
        raise RuntimeError(
            f"predicted minimum HBM headroom {min(headroom) / 1024**3:.3f} GiB is below {args.headroom_gib:.3f} GiB"
        )

    metadata_file = checkpoint / ".metadata"
    physical_bytes = None
    storage_data = getattr(metadata, "storage_data", None)
    if isinstance(storage_data, dict):
        physical_bytes = sum(int(getattr(info, "length", 0)) for info in storage_data.values())

    result = {
        "checkpoint": str(checkpoint),
        "metadata_sha256": sha256(metadata_file) if metadata_file.is_file() else None,
        "receipt_sha256": sha256(receipt_path),
        "logical_tensor_count": len(census),
        "logical_tensor_bytes": sum(item["bytes"] for item in census.values()),
        "physical_chunk_bytes": physical_bytes,
        "max_rank_dcp_bytes": max(rank_dcp),
        "max_rank_predicted_peak_bytes": max(predicted_peak),
        "min_rank_headroom_gib": min(headroom) / 1024**3,
        "status": "KIMI_K3_DCP_TP32_EP128_CAPACITY_OK",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
