"""Periodic and terminal checkpoint seals require all payloads and native cursors."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.terminal_checkpoint import (
    admit_terminal_save,
    prepare_rank_receipt,
    seal_terminal_checkpoint,
    save_full_state,
)
from miles.backends.training_utils.loss_hub.policy_metrics import write_json


def fixture_checkpoint(root, completed=20):
    iteration = completed - 1
    path = root / f"save/iter_{iteration:07d}/adapter"
    path.mkdir(parents=True)
    state = dict(
        iteration=iteration,
        opt_param_scheduler={"num_steps": completed},
        rng_state={"fixture": True},
        provenance=dict(run_id="i001", completed_update=completed, next_rollout_id=completed),
    )
    for rank in range(2):
        for filename in (f"adapter_megatron_rank{rank}.pt", f"optimizer_full_state_rank{rank}.pt"):
            torch.save({"finite_fixture": torch.ones(2)}, path / filename)
        torch.save(state, path / f"training_state_rank{rank}.pt")
        prepare_rank_receipt(path, rank=rank, state=state)
    (path / "adapter_config.json").write_text('{"format":"megatron_rank_sharded"}')
    cursor = root / f"save/rollout/global_dataset_state_dict_{iteration}.pt"
    cursor.parent.mkdir(exist_ok=True)
    torch.save(dict(sample_offset=240, epoch_id=0, sample_group_index=240, sample_index=3840, metadata={}), cursor)
    diagnostics = root / f"diagnostics/paired_scores/update-{completed:06d}/complete.json"
    diagnostics.parent.mkdir(parents=True)
    write_json(diagnostics, {"binding": {"completed_update": completed}})
    args = SimpleNamespace(
        save=str(root / "save"),
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        num_rollout=40,
        save_interval=20,
        policy_diagnostics_dir=str(root / "diagnostics"),
    )
    return args, path, cursor


@pytest.mark.parametrize("completed", [20, 40])
def test_complete_written_last_and_immutable(tmp_path, completed):
    args, path, _ = fixture_checkpoint(tmp_path, completed)
    assert not (path / "FULL_STATE_COMPLETE.json").exists()
    seal_terminal_checkpoint(args, completed - 1)
    marker = json.loads((path / "FULL_STATE_COMPLETE.json").read_text())
    assert marker["rank_count"] == 2
    assert marker["state"] == "VALIDATED_LOCAL_SAVE_RESTORE_NOT_YET_PROVEN"
    with pytest.raises(FileExistsError):
        seal_terminal_checkpoint(args, completed - 1)


def test_capacity_includes_both_retained_and_inflight_generations(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=179))
    with pytest.raises(ValueError, match="Insufficient"):
        admit_terminal_save(tmp_path, expected_payload_bytes=80, reserve_bytes=20)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=180))
    assert (
        admit_terminal_save(tmp_path, expected_payload_bytes=80, reserve_bytes=20)["total_slots_including_inflight"]
        == 2
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
        seal_terminal_checkpoint(args, 18 if damage == "update" else 19)
    assert not (path / "FULL_STATE_COMPLETE.json").exists()


def test_remaining_save_capacity_does_not_double_count_retained_payload(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    assert (
        admit_terminal_save(tmp_path, expected_payload_bytes=80, reserve_bytes=20, remaining_generations=1)[
            "remaining_generations"
        ]
        == 1
    )
    with pytest.raises(ValueError):
        admit_terminal_save(tmp_path, expected_payload_bytes=80, reserve_bytes=20, remaining_generations=2)


def test_both_generations_stay_valid_after_terminal_seal(tmp_path):
    first_args, first, _ = fixture_checkpoint(tmp_path, 20)
    seal_terminal_checkpoint(first_args, 19)
    original_seal = (first / "FULL_STATE_COMPLETE.json").read_bytes()
    second_args, second, _ = fixture_checkpoint(tmp_path, 40)
    seal_terminal_checkpoint(second_args, 39)
    assert (first / "FULL_STATE_COMPLETE.json").read_bytes() == original_seal
    assert json.loads((second / "FULL_STATE_COMPLETE.json").read_text())["checkpoint_role"] == "TERMINAL"
    assert json.loads(original_seal)["checkpoint_role"] == "PERIODIC"


@pytest.mark.parametrize("completed", [10, 19, 21, 41])
def test_unselected_save_positions_are_rejected(tmp_path, completed):
    args = SimpleNamespace(num_rollout=40, save_interval=20)
    with pytest.raises(ValueError):
        seal_terminal_checkpoint(args, completed - 1)


@pytest.mark.parametrize("completed", [20, 40])
def test_full_optimizer_writer_admits_both_selected_positions(tmp_path, monkeypatch, completed):
    from miles.backends.megatron_utils.lora import resumable_optimizer

    class ManifestReached(Exception):
        pass

    def manifest(*args):
        raise ManifestReached

    monkeypatch.setattr(resumable_optimizer, "_optimizer_manifest", manifest)
    with pytest.raises(ManifestReached):
        resumable_optimizer.save_resumable_optimizer_state(
            tmp_path, iteration=completed - 1, model=[], optimizer=None, scheduler=None
        )
    with pytest.raises(RuntimeError, match="updates20 and40"):
        resumable_optimizer.save_resumable_optimizer_state(
            tmp_path, iteration=9, model=[], optimizer=None, scheduler=None
        )


@pytest.mark.parametrize("completed", [20, 40])
def test_full_save_provenance_and_remaining_capacity(tmp_path, monkeypatch, completed):
    from miles.backends.megatron_utils.lora import resumable_optimizer, terminal_checkpoint

    monkeypatch.setenv("MILES_SOURCE_COMMIT", "source")
    monkeypatch.setenv("MILES_EXPERIMENT_RUN_ID", "i001")
    monkeypatch.setenv("MILES_RESOLVED_CONFIG_SHA256", "config")
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.tensor_parallel.random",
        SimpleNamespace(get_cuda_rng_tracker=lambda: SimpleNamespace(get_states=lambda: {})),
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(resumable_optimizer, "save_resumable_optimizer_state", lambda *a, **k: {})
    admitted = []
    monkeypatch.setattr(terminal_checkpoint, "admit_terminal_save", lambda *a, **k: admitted.append(k))
    if completed == 40:
        prior = tmp_path / "save/iter_0000019/adapter/FULL_STATE_COMPLETE.json"
        prior.parent.mkdir(parents=True)
        write_json(
            prior,
            {
                "provenance": {"completed_update": 20, "run_id": "i001", "source": "source"},
                "payload_bytes": 3_000_000_000_000,
            },
        )
    state = save_full_state(
        SimpleNamespace(num_rollout=40, save_interval=20, save=str(tmp_path / "save")),
        tmp_path,
        iteration=completed - 1,
        model=[],
        optimizer=None,
        scheduler=None,
    )
    assert state["provenance"]["completed_update"] == completed
    assert state["provenance"]["next_rollout_id"] == completed
    assert admitted[0]["remaining_generations"] == (2 if completed == 20 else 1)
    if completed == 40:
        assert admitted[0]["expected_payload_bytes"] == 3_000_000_000_000
