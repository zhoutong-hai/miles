#!/usr/bin/env python3
"""Dependency-light local tests for the e006 topology and IPC lifetime patch."""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
UPDATE_SOURCE = ROOT / "miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py"


def load_launcher_module():
    @dataclass
    class ExecuteTrainConfig:
        num_nodes: int = 1

    class Typer:
        def command(self):
            return lambda function: function

    typer = ModuleType("typer")
    typer.Typer = Typer
    command_utils = ModuleType("miles.utils.external_utils.command_utils")
    command_utils.ExecuteTrainConfig = ExecuteTrainConfig
    command_utils.create_run_id = lambda: "local-test"
    command_utils.dataclass_cli = lambda function: function
    command_utils.repo_base_dir = str(ROOT)
    modules = {
        "typer": typer,
        "miles": ModuleType("miles"),
        "miles.utils": ModuleType("miles.utils"),
        "miles.utils.external_utils": ModuleType("miles.utils.external_utils"),
        "miles.utils.external_utils.command_utils": command_utils,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    name = "_e006_launcher_under_test"
    path = ROOT / "scripts/run_kimi_k3_lora.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
        for module_name, old in previous.items():
            if old is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old
    return module


class FakeFlattenedTensorBucket:
    def __init__(self, named_tensors):
        self.named_tensors = named_tensors

    def get_flattened_tensor(self):
        return torch.cat([tensor.contiguous().view(-1) for _name, tensor in self.named_tensors])

    def get_metadata(self):
        return [
            (name, tuple(tensor.shape), str(tensor.dtype), tensor.numel())
            for name, tensor in self.named_tensors
        ]


def load_refresh_function():
    tree = ast.parse(UPDATE_SOURCE.read_text(), filename=str(UPDATE_SOURCE))
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_refresh_flattened_lora_ipc_payload"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    namespace = {"torch": torch, "FlattenedTensorBucket": FakeFlattenedTensorBucket}
    exec(compile(ast.fix_missing_locations(module), str(UPDATE_SOURCE), "exec"), namespace)
    return namespace[node.name]


def load_prepare_method():
    tree = ast.parse(UPDATE_SOURCE.read_text(), filename=str(UPDATE_SOURCE))
    owner = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "UpdateWeightFromTensor")
    method = next(item for item in owner.body if isinstance(item, ast.FunctionDef) and item.name == "_prepare_lora_ipc_payload_for_reuse")
    cls = ast.ClassDef(
        name="Updater",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    namespace = {"torch": torch, "LORA_ADAPTER_NAME": "default"}
    exec(compile(ast.fix_missing_locations(module), str(UPDATE_SOURCE), "exec"), namespace)
    return namespace["Updater"], namespace


def test_stable_refresh() -> None:
    refresh = load_refresh_function()
    first_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    payload = refresh([("layer.lora_A.weight", first_tensor)], None)
    pointer = payload["flattened_tensor"].untyped_storage().data_ptr()
    refreshed = refresh([("layer.lora_A.weight", first_tensor + 10)], payload)
    assert refreshed is payload
    assert refreshed["flattened_tensor"].untyped_storage().data_ptr() == pointer
    torch.testing.assert_close(refreshed["flattened_tensor"], (first_tensor + 10).reshape(-1))
    try:
        refresh([("renamed.lora_A.weight", first_tensor)], payload)
    except RuntimeError as error:
        assert "metadata changed" in str(error)
    else:
        raise AssertionError("metadata drift was accepted")


def test_unload_ack_barrier_before_refresh() -> None:
    updater_cls, namespace = load_prepare_method()
    events = []

    class Dist:
        @staticmethod
        def get_rank():
            return 3

        @staticmethod
        def broadcast_object_list(*_args, **_kwargs):
            events.append("broadcast")

        @staticmethod
        def barrier(*_args, **_kwargs):
            events.append("barrier")

    class Ray:
        @staticmethod
        def get(ref):
            assert ref == "unload_ref"
            events.append("unload_ack")

    class Remote:
        @staticmethod
        def remote(**_kwargs):
            events.append("unload_submit")
            return "unload_ref"

    namespace["dist"] = Dist
    namespace["ray"] = Ray
    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: events.append("cuda_sync")
    try:
        updater = updater_cls()
        updater._ipc_gather_group = "group"
        updater._ipc_gather_src = 3
        updater._ipc_engine = SimpleNamespace(unload_lora_adapter=Remote())
        updater._lora_ipc_live_tensors = [{"flattened_tensor": torch.empty(1), "metadata": []}]
        updater._lora_loaded = True
        updater._prepare_lora_ipc_payload_for_reuse()
        assert updater._lora_loaded is False
    finally:
        torch.cuda.synchronize = original_sync
    assert events == ["unload_submit", "unload_ack", "broadcast", "barrier", "cuda_sync"]


def test_source_contracts() -> None:
    launcher = (ROOT / "scripts/run_kimi_k3_lora.py").read_text()
    lora_utils = (ROOT / "miles/backends/megatron_utils/lora_utils.py").read_text()
    assert '"--no-gradient-accumulation-fusion "' in launcher
    assert "else 32" in launcher and "else 128" in launcher
    assert "return self.total_actor_gpus // model_parallel" in launcher
    assert "_lora_grad_sum_group" in lora_utils
    assert 'gradient = getattr(parameter, "main_grad", None)' in lora_utils
    assert "dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)" in lora_utils


def test_launcher_positive_and_negative_layouts() -> None:
    module = load_launcher_module()
    args = module.ScriptArgs(
        model_variant="full",
        num_nodes=16,
        num_gpus_per_node=8,
        checkpoint_load_mode="shared",
        lora_rank=32,
        lora_alpha=64,
        check_lora_weight_equal=True,
        mode="normal",
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=64,
        rollout_max_response_len=1024,
    )
    assert (
        args.total_actor_gpus,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
        args.context_parallel_size,
        args.expert_parallel_size,
        args.expert_tensor_parallel_size,
        args.trainer_data_parallel_size,
        args.rollout_engine_count,
    ) == (128, 32, 1, 1, 128, 1, 4, 4)
    for nodes, gpus_per_node in ((16, 4), (8, 8), (24, 8), (16, 7)):
        try:
            module.ScriptArgs(
                model_variant="full",
                num_nodes=nodes,
                num_gpus_per_node=gpus_per_node,
                checkpoint_load_mode="shared",
                lora_rank=32,
                lora_alpha=64,
                check_lora_weight_equal=True,
                mode="normal",
                num_rollout=2,
                rollout_batch_size=8,
                n_samples_per_prompt=8,
                global_batch_size=64,
                rollout_max_response_len=1024,
            )
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"invalid launcher layout accepted: {nodes}x{gpus_per_node}")


def main() -> None:
    test_stable_refresh()
    test_unload_ack_barrier_before_refresh()
    test_source_contracts()
    test_launcher_positive_and_negative_layouts()
    print("KIMI_K3_H200_LOCAL_TESTS_OK")


if __name__ == "__main__":
    main()
