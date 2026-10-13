"""Source/API preservation evidence, not heavy GPU/Ray import acceptance."""
import ast
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARRAKIS = Path(os.environ.get("MILES_I004_ARRAKIS_ROOT", "/Users/zhoutong/code/e010-i002-arrakis-pr678-r1"))
PARENT = "c76fca997e6777e95ace05a1b8806b2271ea367d"


def test_frozen_arrakis_import_paths_and_training_factory_arity():
    driver = ast.parse((ARRAKIS / "projects/miles_rl/main_model/common/train.py").read_text())
    for node in driver.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("miles."):
            module_path = ROOT / node.module.replace(".", "/")
            assert module_path.with_suffix(".py").is_file() or module_path.is_dir(), node.module
    definitions = ast.parse((ROOT / "miles/ray/placement_group.py").read_text())
    functions = {n.name: n for n in definitions.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "create_rollout_manager" in functions
    assert [arg.arg for arg in functions["create_training_models"].args.args] == ["args", "pgs", "rollout_manager"]
    calls = [n for n in ast.walk(driver) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "create_training_models"]
    assert len(calls) == 1 and len(calls[0].args) == 3


def test_qualified_lifecycle_flush_reward_packing_and_replay_sources_unchanged():
    paths = ["miles/ray/placement_group.py", "miles/ray/actor_group.py", "miles/backends/megatron_utils/actor.py",
             "miles/utils/hf_config.py", "miles/utils/packing_metrics.py", "miles/utils/replay_base.py",
             "miles/utils/tracking_utils", "miles/ray/rollout", "miles/backends/training_utils/loss_hub"]
    subprocess.run(["git", "-C", str(ROOT), "diff", "--exit-code", PARENT, "--", *paths], check=True, capture_output=True)


def test_frozen_arrakis_is_exact_clean_pin():
    actual = subprocess.run(["git", "-C", str(ARRAKIS), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ARRAKIS), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    assert actual == "410b28164eff54256236d3c224007c5202e2aae5" and not dirty
