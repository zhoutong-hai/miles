"""Entropy feedback sidecar for synchronous native Megatron checkpoints.

Megatron owns model/optimizer/RNG persistence. This file stores only Miles'
publication-dependent feedback, which is not part of Megatron's state dict.
"""

import json
import os
from pathlib import Path
import tempfile

from miles.utils.entropy_control import EntropyController, digest

FILENAME = "miles_entropy_control.json"


def write_entropy_checkpoint(directory, iteration, state):
    controller = EntropyController.restore(state)
    if controller.completed + (controller.pending is not None) != iteration + 1:
        raise ValueError("Native checkpoint and entropy cursor differ")
    value = dict(protocol="miles_full_entropy_v1", iteration=iteration, entropy_control=state,
                 entropy_control_sha256=digest(state))
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("Native checkpoint directory is absent")
    destination = directory / FILENAME
    if destination.exists():
        if json.loads(destination.read_text()) != value:
            raise ValueError("Conflicting immutable native entropy checkpoint")
        return
    fd, temporary = tempfile.mkstemp(prefix=".entropy-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes atomically without replacing an existing checkpoint.
        os.link(temporary, destination)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


def read_entropy_checkpoint(backend):
    from megatron.training.checkpointing import (
        get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata,
    )

    root = Path(backend.load)
    iteration = getattr(backend, "ckpt_step", None)
    if iteration is not None:
        directory = Path(get_checkpoint_name(str(root), iteration, return_base_dir=True))
    elif root.name.startswith("iter_") and root.name[5:].isdigit():
        iteration, directory = int(root.name[5:]), root
    else:
        iteration, release = read_metadata(get_checkpoint_tracker_filename(str(root)))
        if release:
            raise ValueError("Pretrained release cannot restore trained entropy feedback")
        directory = Path(get_checkpoint_name(str(root), iteration, return_base_dir=True))
    value = json.loads((directory / FILENAME).read_text())
    if (value.get("protocol") != "miles_full_entropy_v1" or value.get("iteration") != iteration
            or value.get("entropy_control_sha256") != digest(value.get("entropy_control"))):
        raise ValueError("Native entropy checkpoint identity or seal differs")
    controller = EntropyController.restore(value["entropy_control"])
    if controller.completed + (controller.pending is not None) != iteration + 1:
        raise ValueError("Native checkpoint and entropy cursor differ")
    return value
