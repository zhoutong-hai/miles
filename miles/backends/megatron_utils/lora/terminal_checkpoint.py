"""Selected local terminal full-state save; completion is written after cursor validation."""

import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.policy_metrics import sha256, write_json


def admit_terminal_save(path, *, expected_payload_bytes=1_511_634_979_713, reserve_bytes=100_000_000_000):
    """One total generation slot, including inflight. No deletion or R2 fallback."""
    root = Path(path)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < expected_payload_bytes + reserve_bytes:
        raise ValueError("Insufficient current free space for selected terminal full state plus reserve")
    return {
        "free_bytes": free,
        "expected_payload_bytes": expected_payload_bytes,
        "reserve_bytes": reserve_bytes,
        "total_slots_including_inflight": 1,
    }


def seal_file(path):
    path = Path(path)
    start = time.perf_counter()
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    identity = lambda stat: [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    if identity(before) != identity(after):
        raise ValueError("Checkpoint changed during validation")
    return {
        "filename": path.name,
        "sha256": digest,
        "bytes": after.st_size,
        "file_identity": identity(after),
        "seal_seconds": time.perf_counter() - start,
    }


def save_full_state(args, path, *, iteration, model, optimizer, scheduler):
    from miles.backends.megatron_utils.lora.resumable_optimizer import save_resumable_optimizer_state
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    if iteration != 9 or args.num_rollout != 10:
        raise ValueError("Selected E012 full-state save is terminal update10 only")
    admit_terminal_save(args.save)
    source = os.environ["MILES_SOURCE_COMMIT"]
    run_id = os.environ["MILES_EXPERIMENT_RUN_ID"]
    return {
        "full_state_format": "e012-lora-full-v1",
        "optimizer_full_state": save_resumable_optimizer_state(
            path, iteration=iteration, model=model, optimizer=optimizer, scheduler=scheduler
        ),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all(),
            "megatron": get_cuda_rng_tracker().get_states(),
        },
        "provenance": {
            "run_id": run_id,
            "source": source,
            "completed_update": 10,
            "next_rollout_id": 10,
            "topology": {
                k: getattr(args, k, None)
                for k in (
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "context_parallel_size",
                    "expert_model_parallel_size",
                    "actor_num_nodes",
                    "actor_num_gpus_per_node",
                )
            },
            "configuration_sha256": os.environ["MILES_RESOLVED_CONFIG_SHA256"],
        },
    }


def prepare_rank_receipt(path, *, rank, state):
    if state.get("iteration") != 9 or not state.get("opt_param_scheduler") or not state.get("rng_state"):
        raise ValueError("Terminal training state is incomplete")
    files = [
        seal_file(Path(path) / name)
        for name in (
            f"adapter_megatron_rank{rank}.pt",
            f"training_state_rank{rank}.pt",
            f"optimizer_full_state_rank{rank}.pt",
        )
    ]
    write_json(
        Path(path) / f"full-state-rank-{rank}.json", {"rank": rank, "provenance": state["provenance"], "files": files}
    )


def seal_terminal_checkpoint(args, rollout_id):
    if rollout_id != 9 or args.num_rollout != 10 or args.save_interval != 10:
        raise ValueError("Unexpected full-state checkpoint boundary")
    root = Path(args.save)
    path = root / f"iter_{rollout_id:07d}" / "adapter"
    count = args.actor_num_nodes * args.actor_num_gpus_per_node
    receipts = [json.loads((path / f"full-state-rank-{rank}.json").read_text()) for rank in range(count)]
    reference = receipts[0]["provenance"]
    if any(receipt["rank"] != rank or receipt["provenance"] != reference for rank, receipt in enumerate(receipts)):
        raise ValueError("Full-state rank/provenance coverage differs")
    for receipt in receipts:
        for saved in receipt["files"]:
            stat = (path / saved["filename"]).stat()
            identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
            if identity != saved["file_identity"] or stat.st_size != saved["bytes"]:
                raise ValueError("Full-state file changed after rank validation")
    cursor = root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
    state = torch.load(cursor, map_location="cpu", weights_only=False)
    if not {"sample_offset", "epoch_id", "sample_group_index", "sample_index", "metadata"} <= state.keys():
        raise ValueError("Native dataset cursor is incomplete")
    diagnostics = Path(args.policy_diagnostics_dir) / "paired_scores" / "update-000010" / "complete.json"
    if json.loads(diagnostics.read_text())["binding"]["completed_update"] != 10:
        raise ValueError("Terminal ordinary update diagnostics are incomplete")
    marker = {
        "format": "e012-lora-full-v1",
        "state": "VALIDATED_LOCAL_SAVE_RESTORE_NOT_YET_PROVEN",
        "provenance": reference,
        "rank_count": count,
        "rank_receipts": receipts,
        "cursor": seal_file(cursor),
        "adapter_config": seal_file(path / "adapter_config.json"),
        "diagnostics": seal_file(diagnostics),
        "restore_and_advance": "PENDING_SEPARATE_AUTHORITY",
    }
    write_json(path / "FULL_STATE_COMPLETE.json", marker)


def restore_full_state(path, state, *, model, optimizer, scheduler):
    from miles.backends.megatron_utils.lora.resumable_optimizer import restore_resumable_optimizer_state
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from megatron.training.global_vars import get_args

    marker = json.loads((Path(path) / "FULL_STATE_COMPLETE.json").read_text())
    if marker["provenance"] != state["provenance"]:
        raise ValueError("Checkpoint provenance differs from completion seal")
    args = get_args()
    topology = {key: getattr(args, key, None) for key in state["provenance"]["topology"]}
    if (
        topology != state["provenance"]["topology"]
        or os.environ["MILES_SOURCE_COMMIT"] != state["provenance"]["source"]
    ):
        raise ValueError("Full-state restore requires the same source and topology")
    rank = dist.get_rank() if dist.is_initialized() else 0
    for saved in marker["rank_receipts"][rank]["files"]:
        if sha256(Path(path) / saved["filename"]) != saved["sha256"]:
            raise ValueError("Restore payload differs from sealed terminal generation")
    restore_resumable_optimizer_state(path, state, model=model, optimizer=optimizer, scheduler=scheduler)
    rng = state["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    get_cuda_rng_tracker().set_states(rng["megatron"])
