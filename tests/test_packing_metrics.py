import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.packing_metrics import record_actor_packing, summarize_plan


def plan():
    return dict(
        lengths=[10000, 12000, 8000, 9000],
        response_lengths=[100, 200, 300, 400],
        partitions=[[0, 1], [2, 3]],
        microbatches=[1, 1],
        cp_size=4,
        tp_size=4,
        pad_multiplier=1024,
    )


def test_actual_plan_aggregation_and_padding_do_not_modify_inputs():
    data = plan()
    original = copy.deepcopy(data)
    result = summarize_plan(**data)
    assert data == original
    assert result == {
        "sample_count": 4,
        "processed_tokens": 39000,
        "response_tokens": 1000,
        "microbatches_per_update": [1, 1],
        "padded_tokens": 65536,
        "max_packed_tokens": 22000,
        "max_padded_tokens_per_cp_rank": 8192,
        "padding_multiple": 16384,
    }
    assert "partitions" not in result and "lengths" not in result


@pytest.mark.parametrize(
    "change",
    [
        {"partitions": [[0, 1], [1, 3]]},
        {"partitions": [[0, 1], []]},
        {"microbatches": [2, 1]},
        {"microbatches": [0, 2]},
        {"response_lengths": [100, 200]},
        {"response_lengths": [10001, 200, 300, 400]},
        {"cp_size": 0},
    ],
)
def test_invalid_plan_rejected(change):
    with pytest.raises(ValueError):
        summarize_plan(**(plan() | change))


def record_inputs():
    def group(rank=0, size=1):
        return SimpleNamespace(rank=rank, size=size)

    return dict(
        rollout_id=3,
        args=SimpleNamespace(
            use_dynamic_batch_size=True,
            qkv_format="thd",
            allgather_cp=True,
            max_tokens_per_gpu=8192,
            data_pad_size_multiplier=1024,
        ),
        rollout_data=dict(total_lengths=plan()["lengths"], response_lengths=plan()["response_lengths"]),
        data_iterator=[SimpleNamespace(micro_batch_indices=plan()["partitions"])],
        microbatches=[1, 1],
        parallel_state=SimpleNamespace(tp=group(size=4), cp=group(size=4), pp=group(size=8), effective_dp=group()),
        timers={"log_probs": 52.0, "actor_train": 413.0, "train": 466.0, "train_wait": 173.0},
    )


def test_observer_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MILES_RECORD_THROUGHPUT", raising=False)
    record_actor_packing(**{key: None for key in record_inputs()})


@pytest.mark.parametrize("group", ["tp", "cp", "pp"])
def test_only_one_model_parallel_representative_writes(monkeypatch, tmp_path, group):
    monkeypatch.setenv("MILES_RECORD_THROUGHPUT", "1")
    monkeypatch.setenv("MILES_RUN_DIR", str(tmp_path))
    data = record_inputs()
    getattr(data["parallel_state"], group).rank = 1
    record_actor_packing(**data)
    assert not (tmp_path / "performance").exists()


def test_receipt_is_aggregate_only_and_repeat_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("MILES_RECORD_THROUGHPUT", "1")
    monkeypatch.setenv("MILES_RUN_DIR", str(tmp_path))
    data = record_inputs()
    data["rollout_data"]["tokens"] = ["fake-sensitive-token-array"]
    data["rollout_data"]["prompt"] = "fake-sensitive-prompt"
    record_actor_packing(**data)
    path = tmp_path / "performance/packing/rollout-3-dp-0.json"
    result = json.loads(path.read_text())
    assert result["effective_dp_size"] == 1 and result["processed_tokens"] == 39000
    assert result["phase_seconds"]["actor_train"] == 413.0
    assert "fake-sensitive" not in path.read_text()
    assert "train_wait" not in result["phase_seconds"]
    with pytest.raises(FileExistsError):
        record_actor_packing(**data)


def test_observer_is_after_actor_optimization_and_before_dump():
    actor = Path(__file__).resolve().parents[1] / "miles/backends/megatron_utils/actor.py"
    module = ast.parse(actor.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "train_actor")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    observer = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "record_actor_packing")
    actor_timer = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "timer" and isinstance(node.args[0], ast.Constant) and node.args[0].value == "actor_train")
    dump = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "save_debug_train_data")
    assert actor_timer.lineno < observer.lineno < dump.lineno
