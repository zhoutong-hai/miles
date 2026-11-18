"""Megatron single-LoRA RNG continuation on CUDA, stored inside the committed training state.

RNG streams belong to ranks, so exact continuation requires the saved parallel
layout. Adapter weights and optimizer tensors may still be resharded when the
caller explicitly selects --no-load-rng.
"""

import random

import numpy as np
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.random import convert_cuda_rng_state, get_cuda_rng_tracker

from miles.backends.training_utils.checkpoint.adapter.adapter_state import agree_on_failure_context, gloo_group
from miles.backends.training_utils.parallel import get_parallel_state


def _rank_layout():
    parallel = get_parallel_state()
    groups = (parallel.tp, parallel.pp, parallel.cp, parallel.ep, parallel.etp, parallel.effective_dp)
    return {"sizes": [group.size for group in groups], "ranks": [group.rank for group in groups]}


def capture_rng_state() -> dict:
    """Every trainer rank contributes its Python, NumPy, Torch and Megatron streams as JSON values."""
    with agree_on_failure_context():
        numpy_state = np.random.get_state()
        local = {
            "layout": _rank_layout(),
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state().tolist(),
            "cuda": torch.cuda.get_rng_state().tolist(),
            "tracker": {
                name: convert_cuda_rng_state(state, to_graphable=False).cpu().tolist()
                for name, state in get_cuda_rng_tracker().get_states().items()
            },
        }
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local, group=gloo_group())
    return {"version": 1, "ranks": ranks}


def restore_rng_state(snapshot: dict) -> None:
    """Validate on every rank before changing streams; refuse a different topology instead of remapping RNG."""
    tracker = get_cuda_rng_tracker()
    with agree_on_failure_context():
        assert snapshot["version"] == 1, "unsupported LoRA RNG state version"
        assert len(snapshot["ranks"]) == dist.get_world_size(), "LoRA RNG world size differs; use --no-load-rng"
        local = snapshot["ranks"][dist.get_rank()]
        assert local["layout"] == _rank_layout(), "LoRA RNG rank layout differs; use --no-load-rng"
        python_state = local["python"]
        python_state = (python_state[0], tuple(python_state[1]), python_state[2])
        numpy_state = local["numpy"]
        numpy_state = (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        cpu_state = torch.tensor(local["torch"], dtype=torch.uint8)
        cuda_state = torch.tensor(local["cuda"], dtype=torch.uint8)
        current_streams = tracker.get_states()
        assert local["tracker"].keys() == current_streams.keys(), "LoRA RNG tracker streams differ"
        tracker_states = {
            name: convert_cuda_rng_state(
                torch.tensor(values, dtype=torch.uint8),
                to_graphable=isinstance(current_streams[name], torch.Generator),
            )
            for name, values in local["tracker"].items()
        }
    with agree_on_failure_context():
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        tracker.set_states(tracker_states)
