"""Numeric local spool before tracking delivery; transport loss preserves work."""

import json
import math
from numbers import Real
import os
from pathlib import Path
import socket
import time


def _write(root, prefix, payload):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / f"{prefix}-{socket.gethostname()}-{os.getpid()}-{time.time_ns()}.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def numeric_metrics(metrics):
    output = {}
    for key, value in metrics.items():
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
        if isinstance(value, Real):
            if not math.isfinite(value):
                raise ValueError("Nonfinite native tracking scalar: " + key)
            output[key] = float(value)
    return output


def log_durable(metrics, *, step_key, deliver, writer):
    root = Path(os.environ["MILES_METRICS_SPOOL_ROOT"])
    scalars = numeric_metrics(metrics)
    if step_key not in scalars:
        raise ValueError("Durable native metrics require an explicit numeric step")
    receipt = {"step_key": step_key, "metrics": scalars, "unix_s": time.time()}
    _write(root, "metrics", receipt)
    if not deliver:
        return
    try:
        writer(metrics, step=metrics[step_key], step_key=step_key)
    except Exception as error:
        _write(root, "transport-degraded", {"exception_type": type(error).__name__, **receipt})


def record_online_identity(run, args):
    if (run is None or run.id != args.wandb_run_id or run.project != args.wandb_project
            or run.entity != args.wandb_team or args.wandb_mode != "online"):
        raise ValueError("Online primary identity differs from the selected K3 run")
    _write(Path(os.environ["MILES_METRICS_SPOOL_ROOT"]), "online-identity", {
        "id": run.id, "project": run.project, "entity": run.entity,
        "run_id": os.environ["MILES_EXPERIMENT_RUN_ID"], "unix_s": time.time(),
    })
