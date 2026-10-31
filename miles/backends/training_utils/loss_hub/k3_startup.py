"""Selected K3 startup observations; no optimizer or scientific-state updates."""

import copy
import hashlib
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub.policy_metrics import write_json
from miles.backends.training_utils.parallel import get_parallel_state

_ROUTER_SEALS = {}


def rng_seal(tracker):
    value = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())))
    for state in [torch.get_rng_state(), torch.cuda.get_rng_state(), *[v for _, v in sorted(tracker.get_states().items())]]:
        value.update(state.cpu().numpy().tobytes())
    return value.hexdigest()


def router_seals(model):
    result = {}
    for chunk_id, chunk in enumerate(model):
        tensors = list(chunk.named_parameters()) + list(chunk.named_buffers())
        for name, tensor in tensors:
            if ".router." not in name:
                continue
            # These nonpersistent counters are accounting state, not loaded gates/biases.
            if name.endswith((".local_tokens_per_expert", ".global_tokens_per_expert", ".ga_steps")):
                continue
            if tensor.requires_grad:
                raise ValueError("K3 router must be frozen")
            value = hashlib.sha256()
            for part in tensor.detach().reshape(-1).split(8 << 20):
                value.update(part.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            result[f"{chunk_id}:{name}"] = value.hexdigest()
    return result


def validate_trainable(model):
    trainable = [(name, tensor) for chunk in model for name, tensor in chunk.named_parameters() if tensor.requires_grad]
    if not trainable or any("lora" not in name.lower() for name, _ in trainable):
        raise ValueError("Selected K3 run permits only LoRA trainable parameters")
    return {"trainable_tensors": len(trainable), "trainable_elements": sum(t.numel() for _, t in trainable)}


def run_startup(args, model, optimizer):
    if not os.environ.get("MILES_K3_LONGEST_PROBE"):
        return
    # These model/distributed dependencies are valid only inside an initialized actor.
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from megatron.core.utils import get_model_config
    from miles.backends.megatron_utils.actor import all_replay_managers
    from miles.backends.megatron_utils.model import forward_only
    from miles.backends.training_utils.data import get_data_iterator
    from miles.backends.training_utils.loss_hub.policy_probe import _probe_callback, run_fixed_probe

    parallel = get_parallel_state()
    assert (parallel.tp.size, parallel.pp.size, parallel.cp.size, parallel.effective_dp.size, parallel.ep.size) == (8, 4, 1, 4, 32)
    path = Path(os.environ["MILES_K3_LONGEST_PROBE"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == os.environ["MILES_K3_LONGEST_PROBE_SHA256"]
    packet = torch.load(path, map_location="cpu", weights_only=False)
    assert packet["prompt_tokens"] == 46033 and len(packet["tokens"]) == 46545
    before = router_seals(model)
    trainable = validate_trainable(model)
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    frozen_ids = {id(p) for chunk in model for p in chunk.parameters() if not p.requires_grad}
    assert not parameter_ids.intersection(frozen_ids)
    configuration = copy.copy(args)
    configuration.global_batch_size = 4
    configuration.num_steps_per_rollout = 1
    data = {"tokens": [packet["tokens"].to(device=torch.cuda.current_device(), dtype=torch.long)],
            "total_lengths": [46545], "response_lengths": [512],
            "loss_masks": [torch.ones(512, dtype=torch.int, device=torch.cuda.current_device())]}
    if configuration.use_dynamic_global_batch_size:
        data["dynamic_global_batch_size"] = 4
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    tracker = get_cuda_rng_tracker()
    initial_rng_seal = rng_seal(tracker)
    rng_megatron = {key: value.clone() for key, value in tracker.get_states().items()}
    modes = [chunk.training for chunk in model]
    replay = [(manager, manager.stage) for manager in all_replay_managers if manager.enabled]
    config = get_model_config(model[0])
    timers = config.timers
    for manager, _ in replay:
        manager.stage = "fallthrough"
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            iterator, microbatches = get_data_iterator(configuration, model, data)
            result = forward_only(_probe_callback, configuration, model, iterator, microbatches, rollout_id=0)
            torch.cuda.synchronize()
            if parallel.is_pp_last_stage:
                assert sum(int(value.sum()) for value in result["probe_count"]) == 512
                assert all(torch.isfinite(value).all() for key in result for value in result[key])
    finally:
        for manager, stage in replay:
            manager.stage = stage
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
        tracker.set_states(rng_megatron)
        config.timers = timers
        for chunk, mode in zip(model, modes, strict=True):
            chunk.train(mode)
    assert router_seals(model) == before
    _ROUTER_SEALS.update(before)
    directory = Path(args.policy_diagnostics_dir) / "startup"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"rank-{dist.get_rank():03d}.json", {
        "rank": dist.get_rank(), "world": dist.get_world_size(), **trainable,
        "longest_native_prompt_tokens": 46033, "synthetic_memory_only_response_tokens": 512,
        "forward_seconds": time.perf_counter() - started, "backward_memory_qualified": False,
        "optimizer_updates": 0, "frozen_parameters_excluded_from_optimizer": True,
        "router_tensor_sha256": before, "router_preserved": True,
        "source": os.environ["MILES_SOURCE_COMMIT"], "probe_sha256": os.environ["MILES_K3_LONGEST_PROBE_SHA256"],
        "cuda_allocated_bytes": torch.cuda.memory_allocated(), "cuda_reserved_bytes": torch.cuda.memory_reserved()})
    run_fixed_probe(args, model, completed=0)
    assert rng_seal(tracker) == initial_rng_seal
    assert router_seals(model) == before
    write_json(directory / f"probe-neutrality-rank-{dist.get_rank():03d}.json", {
        "rank": dist.get_rank(), "longest_and_fixed64_RNG_neutral": True,
        "router_weights_and_loaded_biases_preserved": True, "optimizer_updates": 0})


def verify_first_update(args, model, completed):
    if completed != 1 or not os.environ.get("MILES_K3_LONGEST_PROBE"):
        return
    assert router_seals(model) == _ROUTER_SEALS
    validate_trainable(model)
    write_json(Path(args.policy_diagnostics_dir) / "startup" / f"first-update-rank-{dist.get_rank():03d}.json", {
        "rank": dist.get_rank(), "completed_update": completed,
        "frozen_router_loaded_weights_and_biases_preserved": True,
        "router_tensor_count": len(_ROUTER_SEALS), "source": os.environ["MILES_SOURCE_COMMIT"]})
