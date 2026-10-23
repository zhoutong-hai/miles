"""Model-free tests of the actual Megatron controller boundary functions."""

import ast
import math
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from miles.backends.training_utils.adaptive_entropy import next_entropy_coefficient, saved_entropy_coefficient


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return Scalar(self.values[index])


def actual_functions():
    path = Path(__file__).resolve().parents[2] / "miles/backends/megatron_utils/adaptive_entropy.py"
    tree = ast.parse(path.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"advance_adaptive_entropy", "restore_adaptive_entropy"}
    ]
    assert len(selected) == 2
    distributed = types.SimpleNamespace(
        all_reduce=lambda tensor, group: None,
        get_world_size=lambda group: 1,
        get_rank=lambda: 0,
        broadcast_object_list=lambda packet, src, group: None,
    )
    namespace = {
        "Namespace": Namespace,
        "math": math,
        "torch": types.SimpleNamespace(
            tensor=lambda values, **kwargs: FakeTensor(values),
            float64="float64",
            distributed=distributed,
        ),
        "get_gloo_group": lambda: None,
        "is_first_replica_megatron_main_rank": lambda: True,
        "next_entropy_coefficient": next_entropy_coefficient,
        "saved_entropy_coefficient": saved_entropy_coefficient,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class MegatronBoundaryTests(unittest.TestCase):
    def test_only_successful_optimizer_step_changes_next_coefficient(self):
        functions = actual_functions()
        args = Namespace(
            entropy_coef=0.0,
            adaptive_entropy_target=0.4,
            adaptive_entropy_rate=0.0001,
            adaptive_entropy_max_coef=0.001,
        )
        metrics = functions["advance_adaptive_entropy"](args, {"entropy_loss": 0.2}, True)
        self.assertEqual(metrics["entropy_coef_applied"], 0.0)
        self.assertAlmostEqual(metrics["entropy_coef_next"], 0.00002)
        self.assertAlmostEqual(args.entropy_coef, 0.00002)
        metrics = functions["advance_adaptive_entropy"](args, {"entropy_loss": 0.1}, False)
        self.assertEqual(metrics["entropy_controller_advanced"], 0.0)
        self.assertAlmostEqual(args.entropy_coef, 0.00002)

    def test_nonfinite_global_entropy_fails_closed(self):
        functions = actual_functions()
        args = Namespace(
            entropy_coef=0.0,
            adaptive_entropy_target=0.4,
            adaptive_entropy_rate=0.0001,
            adaptive_entropy_max_coef=0.001,
        )
        with self.assertRaisesRegex(RuntimeError, "finite global"):
            functions["advance_adaptive_entropy"](args, {"entropy_loss": float("nan")}, True)

    def test_non_source_rank_applies_same_global_coefficient(self):
        functions = actual_functions()
        functions["is_first_replica_megatron_main_rank"] = lambda: False
        distributed = functions["torch"].distributed
        distributed.get_world_size = lambda group: 128
        distributed.all_reduce = lambda tensor, group: tensor.values.__setitem__(slice(None), [0.1, 1, 0, 128])
        args = Namespace(
            entropy_coef=0.0002,
            adaptive_entropy_target=0.4,
            adaptive_entropy_rate=0.0001,
            adaptive_entropy_max_coef=0.001,
        )
        functions["advance_adaptive_entropy"](args, {}, True)
        self.assertAlmostEqual(args.entropy_coef, 0.00023)

    def test_exact_restore_reads_megatron_common_state(self):
        functions = actual_functions()
        args = Namespace(
            load="/checkpoint-root",
            finetune=False,
            no_load_optim=False,
            no_load_rng=False,
            entropy_coef=0.0,
            adaptive_entropy_target=0.4,
            adaptive_entropy_rate=0.0001,
            adaptive_entropy_max_coef=0.001,
        )
        saved = Namespace(
            entropy_coef=0.00042,
            adaptive_entropy_target=0.4,
            adaptive_entropy_rate=0.0001,
            adaptive_entropy_max_coef=0.001,
        )
        core = types.ModuleType("megatron.core")
        core.dist_checkpointing = types.SimpleNamespace(
            load_common_state_dict=lambda path: {"iteration": 21, "args": saved}
        )
        checkpointing = types.ModuleType("megatron.training.checkpointing")
        checkpointing.get_checkpoint_name = lambda root, iteration, return_base_dir: f"{root}/iter_{iteration:07d}"
        with patch.dict(sys.modules, {"megatron": types.ModuleType("megatron"), "megatron.core": core,
                                      "megatron.training": types.ModuleType("megatron.training"),
                                      "megatron.training.checkpointing": checkpointing}):
            functions["restore_adaptive_entropy"](args, 21)
        self.assertAlmostEqual(args.entropy_coef, 0.00042)


if __name__ == "__main__":
    unittest.main()
