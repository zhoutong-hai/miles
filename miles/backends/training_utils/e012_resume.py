"""Strict native E012 continuation guards; no model/optimizer math changes."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_plan():
    value = os.environ.get("MILES_E012_RESUME_PLAN")
    if not value:
        return None
    path = Path(value)
    require(path.is_absolute() and not path.is_symlink(), "Unsafe E012 resume plan")
    wire = path.read_bytes()
    require(hashlib.sha256(wire).hexdigest() == os.environ["MILES_E012_RESUME_PLAN_SHA256"], "E012 resume plan changed")
    plan = json.loads(wire)
    require(plan["schema_version"] == 1 and plan["resume_update"] == 20 and plan["terminal_update"] == 40, "E012 resume horizon drift")
    require(plan["run_id"] == os.environ["MILES_EXPERIMENT_RUN_ID"] == "e012-i001-baseline-validation-40", "E012 resume run drift")
    require(os.environ["MILES_EXPERIMENT_ATTEMPT_ID"] == plan["attempt"] == "r16", "E012 resume attempt drift")
    require(plan["source_commit"] == os.environ["MILES_SOURCE_COMMIT"], "E012 resume source drift")
    root = Path(plan["checkpoint_root"])
    marker = root / "iter_0000019/adapter/FULL_STATE_COMPLETE.json"
    require(not marker.is_symlink() and hashlib.sha256(marker.read_bytes()).hexdigest() == plan["checkpoint_marker_sha256"], "E012 checkpoint20 seal changed")
    saved = json.loads(marker.read_text())
    require(saved["state"] == "VALIDATED_LOCAL_SAVE_RESTORE_NOT_YET_PROVEN" and saved["provenance"]["completed_update"] == 20 and saved["provenance"]["next_rollout_id"] == 20, "E012 wrong checkpoint boundary")
    require(saved["provenance"]["source"] == plan["source_commit"] and saved["provenance"]["configuration_sha256"] == plan["parent_config_sha256"], "E012 checkpoint lineage drift")
    cursor = root / "rollout/global_dataset_state_dict_19.pt"
    require(not cursor.is_symlink() and hashlib.sha256(cursor.read_bytes()).hexdigest() == plan["cursor_sha256"], "E012 cursor seal changed")
    terminal = json.loads(Path(plan["parent_terminal_path"]).read_text())
    require(terminal["status"] == "FAILED" and terminal["submission_id"] == "e012-i001-baseline-validation-40-r15", "E012 predecessor is unresolved")
    return plan


def resume_update(args=None):
    plan = read_plan()
    return plan["resume_update"] if plan else 0


def validate_args(args):
    plan = read_plan()
    if not plan:
        return
    required = {"num_rollout": 40, "start_rollout_id": 20, "global_batch_size": 384, "rollout_batch_size": 24, "n_samples_per_prompt": 16,
                "tensor_model_parallel_size": 8, "pipeline_model_parallel_size": 4, "context_parallel_size": 1, "expert_model_parallel_size": 32,
                "actor_num_nodes": 16, "actor_num_gpus_per_node": 8, "max_tokens_per_gpu": 8192, "optimizer_offload_fraction": 0.8}
    for key, expected in required.items():
        require(getattr(args, key) == expected, f"E012 resume argument drift: {key}")
    require(args.lora_adapter_path == plan["checkpoint_root"] + "/iter_0000019/adapter", "E012 adapter source drift")
    require(args.save == plan["checkpoint_root"], "E012 must retain10/20 and save30/40 in the same four slots")
    require(args.load == args.ref_load == plan["base_checkpoint"], "E012 base reconstruction source drift")
    require(args.skip_eval_before_train, "E012 eval20 is already accepted")


@contextmanager
def pretrained_base_context(args):
    keys = ("lora_adapter_path", "no_load_optim", "no_load_rng", "finetune")
    saved = {key: getattr(args, key) for key in keys}
    args.lora_adapter_path = None
    args.no_load_optim = args.no_load_rng = args.finetune = True
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(args, key, value)


def load_checkpoint(base_loader, args, model, optimizer, scheduler, context, skip, native_loader):
    plan = read_plan()
    if not plan:
        return base_loader(model, optimizer, scheduler, context, skip)
    validate_args(args)
    with pretrained_base_context(args):
        base = base_loader(model, optimizer, scheduler, context, skip)
    args.no_load_optim = args.no_load_rng = args.finetune = False
    loaded, iteration = native_loader(model, args.lora_adapter_path, optimizer=optimizer, opt_param_scheduler=scheduler, load_optimizer=True)
    require(loaded and iteration == 19, "E012 did not restore complete update20 state")
    require(scheduler.num_steps == 7680, "E012 scheduler reset or double-count")
    return iteration, base[1]


def cursor_root(default):
    plan = read_plan()
    return plan["checkpoint_root"] if plan else default


def validate_cursor(args, state, next_samples):
    plan = read_plan()
    if not plan:
        return
    expected = {"sample_offset": 480, "epoch_id": 0, "sample_group_index": 480, "sample_index": 7680}
    require({key: state[key] for key in expected} == expected, "E012 data cursor reset or double-count")
    require(len(next_samples) == 24, "E012 next prompt count drift")
    hashes = [hashlib.sha256(sample.prompt.encode()).hexdigest() for sample in next_samples]
    digest = hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()
    require(digest == plan["next_prompt_hashes_sha256"], "E012 restored shuffle/cursor differs from original next24 prompts")
    from miles.backends.training_utils.loss_hub.policy_metrics import write_json
    directory = Path(args.policy_diagnostics_dir) / "native_resume"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(directory / "cursor20.json", {"state": "PASS", "cursor": expected, "next24_prompt_hashes_sha256": digest, "source_cursor_sha256": plan["cursor_sha256"]})


def verify_rng(args, saved):
    if not read_plan():
        return
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    require(random.getstate() == saved["python"], "E012 Python RNG restore drift")
    current = np.random.get_state()
    require(current[0] == saved["numpy"][0] and np.array_equal(current[1], saved["numpy"][1]) and current[2:] == saved["numpy"][2:], "E012 NumPy RNG restore drift")
    require(torch.equal(torch.get_rng_state(), saved["torch_cpu"]), "E012 torch CPU RNG restore drift")
    cuda = torch.cuda.get_rng_state_all()
    require(len(cuda) == len(saved["torch_cuda_all"]) and all(torch.equal(a, b) for a, b in zip(cuda, saved["torch_cuda_all"])), "E012 CUDA RNG restore drift")
    megatron = get_cuda_rng_tracker().get_states()
    require(megatron.keys() == saved["megatron"].keys() and all(torch.equal(v, saved["megatron"][k]) for k, v in megatron.items()), "E012 Megatron RNG restore drift")
    from miles.backends.training_utils.loss_hub.policy_metrics import write_json
    directory = Path(args.policy_diagnostics_dir) / "native_resume"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(directory / f"rng-rank-{dist.get_rank():03d}.json", {"state": "PASS", "completed_update": 20, "five_RNG_families_exact": True})
