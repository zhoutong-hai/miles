"""DP-sharded source tensors must be audited as a union, excluding frozen tensors."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.utils.reinforce_audits import optimizer_membership


def _worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = torch.nn.Module()
        model.register_parameter("lora_a", torch.nn.Parameter(torch.ones(3)))
        model.register_parameter("lora_b", torch.nn.Parameter(torch.ones(5)))
        model.register_parameter("base", torch.nn.Parameter(torch.ones(2), requires_grad=False))
        child = SimpleNamespace(model_float16_groups=[[model.lora_a if rank == 0 else model.lora_b]], model_fp32_groups=[], data_parallel_group=dist.group.WORLD)
        optimizer = SimpleNamespace(chained_optimizers=[child])
        dp = SimpleNamespace(size=2, group=dist.group.WORLD)
        result = optimizer_membership([model], optimizer, dp)
        assert result["source_parameters"] == 2 and result["local_shard_source_parameters"] == 1
        assert result["DP_union_verified"]
        # A frozen source on either rank must fail on both ranks.
        if rank == 0:
            child.model_float16_groups[0].append(model.base)
        with pytest.raises(ValueError, match="DP source membership"):
            optimizer_membership([model], optimizer, dp)
        child.model_float16_groups = [[model.lora_a]]
        # Missing globally cannot be excused by local shard ownership.
        with pytest.raises(ValueError, match="DP source membership"):
            optimizer_membership([model], optimizer, dp)
        Path(output, f"rank-{rank}").write_text("PASS")
    finally:
        dist.destroy_process_group()


def test_sharded_union_covers_trainables_and_rejects_frozen_or_missing(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert [Path(tmp_path, f"rank-{rank}").read_text() for rank in range(2)] == ["PASS", "PASS"]
