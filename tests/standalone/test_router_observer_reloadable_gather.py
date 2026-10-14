"""Actual CPU Gloo gather with Miles' reloadable wrapper; no model or provider."""

import importlib.util
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]


def _worker(rank, rendezvous, output, wrapped):
    sys.path.insert(0, str(ROOT))
    from miles.utils.reloadable_process_group import (
        ReloadableProcessGroup,
        monkey_patch_torch_dist,
    )

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    monkey_patch_torch_dist()
    group = dist.new_group(ranks=[0, 1], backend="gloo")
    if wrapped:
        group = ReloadableProcessGroup(group, inner_args=(),
                                       inner_kwargs={"ranks": [0, 1], "backend": "gloo"})
    spec = importlib.util.spec_from_file_location(
        "actual_router_observer", ROOT / "miles/backends/megatron_utils/router_load_observer.py"
    )
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    logged = []
    sys.modules["miles.utils.tracking_utils"] = SimpleNamespace(
        tracking=SimpleNamespace(log=lambda args, metrics, step_key: logged.append(metrics))
    )
    obj = observer.RouterLoadObserver.__new__(observer.RouterLoadObserver)
    obj.args = SimpleNamespace(router_load_output=output, router_load_snapshot_interval=10)
    obj.representative = True
    obj.mpu = SimpleNamespace(get_pipeline_model_parallel_group=lambda: group)
    layer_ids = range(1, 38) if rank == 0 else range(38, 76)
    obj.routers = [SimpleNamespace(layer_number=layer, local_tokens_per_expert=torch.zeros(256))
                   for layer in layer_ids]
    obj.counter_dtypes = ("torch.bfloat16",) * len(obj.routers)
    obj.pending_dtype = "torch.float32"
    obj.pending = torch.ones((len(obj.routers), 256), dtype=torch.int64)
    obj.real_positions, obj.packed_positions = 28, 32
    obj.config = SimpleNamespace(moe_router_topk=8)
    obj.rank_experts = [list(range(16 * r, 16 * (r + 1))) for r in range(16)]
    obj.first_check, obj.accounting_valid, obj.snapshot = True, None, ()
    calls = []
    original_gather = dist.gather_object

    def counted_gather(*args, **kwargs):
        calls.append(kwargs["group"])
        return original_gather(*args, **kwargs)

    dist.gather_object = counted_gather
    obj.finish(rollout_id=0, step_id=0, num_steps=2)
    if wrapped:
        ReloadableProcessGroup.destroy_process_groups()
        ReloadableProcessGroup.reload_process_groups()
    obj.finish(rollout_id=0, step_id=1, num_steps=2)
    assert len(calls) == 2
    assert all(not isinstance(g, ReloadableProcessGroup) for g in calls)
    if rank == 1:
        assert [row["router_load/completed_updates"] for row in logged] == [1, 2]
        assert all(row["router_load/accounting_passed"] == 1 for row in logged)
        receipt = json.loads((Path(output) / "first-update-accounting.json").read_text())
        assert len(receipt["layers"]) == 75
        assert {row["layer"] for row in receipt["layers"]} == set(range(1, 76))
        assert all(row["accounting"]["counts_match_packed_work"] for row in receipt["layers"])
        assert all(row["accounting"]["local_counter_dtype"] == "torch.float32" for row in receipt["layers"])
        assert all(row["accounting"]["reduced_counter_dtype"] == "torch.float32" for row in receipt["layers"])
    dist.destroy_process_group()


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_pipeline_gather_accepts_plain_and_reloaded_group(tmp_path, wrapped):
    context = mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path / "observations"), wrapped),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 45
        while not context.join(timeout=max(0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "CPU process-group regression timed out"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
