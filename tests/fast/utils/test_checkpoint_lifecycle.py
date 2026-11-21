"""CPU coverage of native save ordering without importing model backends."""

import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.checkpoint_lifecycle import CheckpointLifecycle


def test_disabled_checkpoint_lifecycle_needs_no_deployment_module():
    lifecycle = CheckpointLifecycle.from_args(SimpleNamespace(custom_checkpoint_lifecycle_hook_path=None))
    asyncio.run(lifecycle.event("start"))
    asyncio.run(lifecycle.close())


def native_save(callback, *, fail_model=False):
    # Exercise the actual native closure while leaving CUDA/Ray imports out of
    # this CPU test. Both native writers and the lifecycle callback are real awaits.
    tree = ast.parse((Path(__file__).resolve().parents[3] / "train.py").read_text())
    train = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "train")
    save = next(n for n in train.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "save")

    async def rollout(rollout_id):
        callback("rollout", rollout_id)

    async def model(rollout_id, *, force_sync):
        callback("model_sync" if force_sync else "model", rollout_id)
        if fail_model:
            raise RuntimeError("native save failed")

    namespace = dict(
        args=SimpleNamespace(num_rollout=219, use_critic=False),
        checkpoints=CheckpointLifecycle(lambda _unused=None, *, phase, rollout_id: callback(phase, rollout_id)),
        rollout_executor=SimpleNamespace(save=rollout),
        actor_model=SimpleNamespace(save_model=model),
    )
    exec(compile(ast.Module(body=[save], type_ignores=[]), "native-save", "exec"), namespace)
    return namespace["save"]


def test_admission_precedes_both_writers_and_completion_follows_them():
    events, driver_thread = [], threading.get_ident()

    def callback(phase, rollout_id):
        events.append((phase, rollout_id))
        if phase in ("before_save", "after_save"):
            assert threading.get_ident() != driver_thread

    asyncio.run(native_save(callback)(19))
    assert events == [("before_save", 19), ("rollout", 19), ("model", 19), ("after_save", 19)]


def test_failed_admission_does_not_start_either_native_writer():
    events = []

    def callback(phase, rollout_id):
        events.append(phase)
        if phase == "before_save":
            raise ValueError("no capacity")

    with pytest.raises(ValueError, match="capacity"):
        asyncio.run(native_save(callback)(19))
    assert events == ["before_save"]


def test_failed_model_save_never_announces_complete():
    events = []
    with pytest.raises(RuntimeError, match="native save failed"):
        asyncio.run(native_save(lambda p, _: events.append(p), fail_model=True)(19))
    assert events == ["before_save", "rollout", "model"]


def test_terminal_checkpoint_is_synchronous_and_uses_same_guard():
    events = []
    asyncio.run(native_save(lambda p, _: events.append(p))(218))
    assert events == ["before_save", "rollout", "model_sync", "after_save"]


@pytest.mark.parametrize(
    "changed", [{"save": None}, {"async_save": True}, {"fully_async": True}, {"advantage_estimator": "ppo"}]
)
def test_unsupported_checkpoint_config_is_rejected_before_backend_start(changed):
    source = Path(__file__).resolve().parents[3] / "miles/utils/arguments.py"
    tree = ast.parse(source.read_text())
    validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "miles_validate_args")
    guard = next(
        n
        for n in validate.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "args.custom_checkpoint_lifecycle_hook_path is not None"
    )
    args = SimpleNamespace(
        custom_checkpoint_lifecycle_hook_path="checkpoints.lifecycle",
        save="/checkpoint",
        async_save=False,
        fully_async=False,
        advantage_estimator="grpo",
    )
    # use_critic has not been derived at this point in native validation.
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"), {"args": args})
    vars(args).update(changed)
    with pytest.raises(AssertionError):
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(source), "exec"), {"args": args})
