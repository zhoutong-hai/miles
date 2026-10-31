"""Selected periodic/terminal local full-state saves; seal after cursor validation."""

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


def admit_terminal_save(
    path, *, expected_payload_bytes=1_511_634_979_713, reserve_bytes=100_000_000_000,
    remaining_generations=4, total_slots=4
):
    """Reserve only future writes; retain every selected generation. No deletion."""
    if total_slots not in (2, 4) or not 1 <= remaining_generations <= total_slots:
        raise ValueError("Selected saving run has two or four total generation slots")
    root = Path(path)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < remaining_generations * expected_payload_bytes + reserve_bytes:
        raise ValueError("Insufficient current free space for remaining full states plus reserve")
    return {
        "free_bytes": free,
        "expected_payload_bytes": expected_payload_bytes,
        "reserve_bytes": reserve_bytes,
        "total_slots_including_inflight": total_slots,
        "remaining_generations": remaining_generations,
    }


def _selected_boundary(args, rollout_id):
    completed = int(rollout_id) + 1
    if (args.num_rollout != 40 or args.save_interval not in (10, 20)
            or completed not in range(args.save_interval, 41, args.save_interval)):
        raise ValueError("Selected E012 full-state save is outside its declared ten/twenty-update cadence")
    return completed


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

    completed = _selected_boundary(args, iteration)
    expected_payload = 1_511_634_979_713
    boundaries = tuple(range(args.save_interval, 41, args.save_interval))
    for prior_update in (update for update in boundaries if update < completed):
        prior_path = Path(args.save) / f"iter_{prior_update - 1:07d}" / "adapter" / "FULL_STATE_COMPLETE.json"
        prior = json.loads(prior_path.read_text())
        provenance = prior["provenance"]
        if (
            provenance["completed_update"] != prior_update
            or provenance["run_id"] != os.environ["MILES_EXPERIMENT_RUN_ID"]
            or provenance["source"] != os.environ["MILES_SOURCE_COMMIT"]
        ):
            raise ValueError("Every retained full state must belong to this same run and source")
        expected_payload = max(expected_payload, prior["payload_bytes"])
    admit_terminal_save(
        args.save, expected_payload_bytes=expected_payload,
        remaining_generations=sum(update >= completed for update in boundaries), total_slots=len(boundaries)
    )
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
            "completed_update": completed,
            "next_rollout_id": completed,
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
    if (
        state.get("iteration") not in (9, 19, 29, 39)
        or not state.get("opt_param_scheduler")
        or not state.get("rng_state")
        or state["provenance"]["completed_update"] != state["iteration"] + 1
    ):
        raise ValueError("Selected full training state is incomplete or has the wrong update")
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
    completed = _selected_boundary(args, rollout_id)
    root = Path(args.save)
    path = root / f"iter_{rollout_id:07d}" / "adapter"
    count = args.actor_num_nodes * args.actor_num_gpus_per_node
    receipts = [json.loads((path / f"full-state-rank-{rank}.json").read_text()) for rank in range(count)]
    reference = receipts[0]["provenance"]
    if reference["completed_update"] != completed or reference["next_rollout_id"] != completed:
        raise ValueError("Checkpoint state/provenance does not match the actual optimizer boundary")
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
    diagnostics = Path(args.policy_diagnostics_dir) / "paired_scores" / f"update-{completed:06d}" / "complete.json"
    if json.loads(diagnostics.read_text())["binding"]["completed_update"] != completed:
        raise ValueError("Save-boundary ordinary update diagnostics are incomplete")
    marker = {
        "format": "e012-lora-full-v1",
        "state": "VALIDATED_LOCAL_SAVE_RESTORE_NOT_YET_PROVEN",
        "provenance": reference,
        "rank_count": count,
        "rank_receipts": receipts,
        "checkpoint_role": "TERMINAL" if completed == args.num_rollout else "PERIODIC",
        "payload_bytes": sum(saved["bytes"] for receipt in receipts for saved in receipt["files"]),
        "cursor": seal_file(cursor),
        "adapter_config": seal_file(path / "adapter_config.json"),
        "diagnostics": seal_file(diagnostics),
        "restore_and_advance": "UNPROVEN_NO_EXTENSION_BEYOND40",
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
            raise ValueError("Restore payload differs from sealed full-state generation")
    restore_resumable_optimizer_state(path, state, model=model, optimizer=optimizer, scheduler=scheduler)
    rng = state["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    get_cuda_rng_tracker().set_states(rng["megatron"])
