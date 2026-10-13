"""Opt-in, process-owned first-dispatch receipts; no tensor scans or public logs."""
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import torch


def package_identity(name):
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "direct_url": None}
    direct = distribution.read_text("direct_url.json")
    return {"version": distribution.version, "direct_url": safe_direct_url(json.loads(direct)) if direct else None}


def safe_direct_url(value):
    """Retain immutable source identity, never URL credentials/query tokens."""
    url = urlsplit(value.get("url", ""))
    authority = (url.hostname or "") + (f":{url.port}" if url.port else "")
    result = {"url": urlunsplit((url.scheme, authority, url.path, "", ""))}
    commit = value.get("vcs_info", {}).get("commit_id", "")
    if re.fullmatch(r"[0-9a-f]{7,64}", commit):
        result["commit_id"] = commit
    return result


def module_identity(name):
    module = sys.modules.get(name)
    path = getattr(module, "__file__", None)
    return {"file": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and path.endswith(".py") else None}


class BackendObserver:
    """Owns bounded receipts per selected tuning/phase, once per actor process.

    Kernel return is dispatch evidence only. Numerical and timing acceptance
    still belongs to qualification; this does not synchronize GPU execution.
    """

    def __init__(self, directory=None, require_forward=None):
        self.directory = Path(directory) if directory else None
        self.require_forward = require_forward
        self.seen = set()
        self.packages = None

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get("MILES_DSA_DISPATCH_RECEIPT_DIR"), os.environ.get("MILES_DSA_REQUIRE_FORWARD_BACKEND"))

    def require(self, config):
        if self.require_forward and config.forward_backend != self.require_forward:
            raise RuntimeError(f"Required DSA backend {self.require_forward}, selected {config.forward_backend}; qualification cannot silently fall back")

    def record(self, phase, q, kv, indices, d_v, config):
        if self.directory is None:
            return
        key = (phase, q.device, q.dtype, q.shape[2:], kv.shape[2:], d_v, config)
        if key in self.seen:
            return
        if self.packages is None:
            self.packages = {name: package_identity(name) for name in ("torch", "tilelang", "apache-tvm-ffi", "flash-mla")}
        packet = {
            "status": "KERNEL_RETURNED_NOT_NUMERIC_ACCEPTANCE", "phase": phase,
            "backend": config.forward_backend if phase == "forward" else "tilelang",
            "device": str(q.device), "architecture": torch.cuda.get_device_capability(q.device),
            "device_name": torch.cuda.get_device_name(q.device), "q_shape": list(q.shape),
            "kv_shape": list(kv.shape), "indices_shape": list(indices.shape), "dtype": str(q.dtype),
            "d_v": d_v, "tuning": asdict(config), "packages": self.packages,
            "imports": {name: module_identity(name) for name in (
                "miles.kernels.attention.dsa.sparse_attention", "flash_mla", "flash_mla.flash_mla_interface",
                "flash_mla_cuda", "miles.kernels.attention.dsa.tilelang.sparse_attention_fwd",
                "miles.kernels.attention.dsa.tilelang.sparse_attention_bwd")},
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"dispatch-{os.getpid()}-{len(self.seen)}.json"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(packet, stream, indent=2, sort_keys=True)
            stream.write("\n")
        self.seen.add(key)
