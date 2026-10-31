"""Terminal checkpoint seals require every payload and native cursor."""

import json
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.terminal_checkpoint import (
    admit_terminal_save,
    prepare_rank_receipt,
    seal_terminal_checkpoint,
)
from miles.backends.training_utils.loss_hub.policy_metrics import write_json


def fixture_checkpoint(root):
    path = root / "save/iter_0000009/adapter"
    path.mkdir(parents=True)
    state = dict(
        iteration=9,
        opt_param_scheduler={"num_steps": 10},
        rng_state={"fixture": True},
        provenance=dict(run_id="i001", completed_update=10, next_rollout_id=10),
    )
    for rank in range(2):
        for filename in (f"adapter_megatron_rank{rank}.pt", f"optimizer_full_state_rank{rank}.pt"):
            torch.save({"finite_fixture": torch.ones(2)}, path / filename)
        torch.save(state, path / f"training_state_rank{rank}.pt")
        prepare_rank_receipt(path, rank=rank, state=state)
    (path / "adapter_config.json").write_text('{"format":"megatron_rank_sharded"}')
    cursor = root / "save/rollout/global_dataset_state_dict_9.pt"
    cursor.parent.mkdir()
    torch.save(dict(sample_offset=240, epoch_id=0, sample_group_index=240, sample_index=3840, metadata={}), cursor)
    diagnostics = root / "diagnostics/paired_scores/update-000010/complete.json"
    diagnostics.parent.mkdir(parents=True)
    write_json(diagnostics, {"binding": {"completed_update": 10}})
    args = SimpleNamespace(
        save=str(root / "save"),
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        num_rollout=10,
        save_interval=10,
        policy_diagnostics_dir=str(root / "diagnostics"),
    )
    return args, path, cursor


def test_complete_written_last_and_immutable(tmp_path):
    args, path, _ = fixture_checkpoint(tmp_path)
    assert not (path / "FULL_STATE_COMPLETE.json").exists()
    seal_terminal_checkpoint(args, 9)
    marker = json.loads((path / "FULL_STATE_COMPLETE.json").read_text())
    assert marker["rank_count"] == 2
    assert marker["state"] == "VALIDATED_LOCAL_SAVE_RESTORE_NOT_YET_PROVEN"
    with pytest.raises(FileExistsError):
        seal_terminal_checkpoint(args, 9)


def test_capacity_includes_the_one_inflight_generation(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=99))
    with pytest.raises(ValueError, match="Insufficient"):
        admit_terminal_save(tmp_path, expected_payload_bytes=80, reserve_bytes=20)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    assert (
        admit_terminal_save(tmp_path, expected_payload_bytes=80, reserve_bytes=20)["total_slots_including_inflight"]
        == 1
    )


@pytest.mark.parametrize("damage", ["optimizer", "adapter", "training_state", "cursor", "rank", "update"])
def test_unresolved_generation_never_gets_completion_marker(tmp_path, damage):
    args, path, cursor = fixture_checkpoint(tmp_path)
    if damage in ("optimizer", "adapter", "training_state"):
        filename = {
            "optimizer": "optimizer_full_state_rank1.pt",
            "adapter": "adapter_megatron_rank1.pt",
            "training_state": "training_state_rank1.pt",
        }[damage]
        (path / filename).write_bytes(b"changed")
    elif damage == "cursor":
        torch.save({"sample_offset": 240}, cursor)
    elif damage == "rank":
        (path / "full-state-rank-1.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        seal_terminal_checkpoint(args, 8 if damage == "update" else 9)
    assert not (path / "FULL_STATE_COMPLETE.json").exists()
