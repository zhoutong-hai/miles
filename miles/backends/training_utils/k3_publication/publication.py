"""Structural-only LoRA publication for the authorized40-update treatment.

Retain transport,lifetime,owned staging,manifest,ACK,abort,install/version guards.
Routine tensor SHA and installed-byte qualification audit are not executed.
The archived v8/r9 byte-parity diagnostic is provenance,not repeated work.
"""

import hashlib
import json
import os
from pathlib import Path
import socket
import time

from miles.backends.training_utils.k3_publication.shard_manifest import is_routed_inner, verify_receiver_receipt as audited_verify
from miles.backends.training_utils.k3_publication.publication_paths import publication_root_matches

PREFIX = "e012-structure-v1:"


def publication_mode(version):
    if os.environ.get("MILES_K3_PUBLICATION_MODE") != "structural":
        raise RuntimeError("structural publication must be explicitly selected")
    if type(version) is not int or not 1 <= version <= 41:
        raise ValueError("full treatment permits initial publication plus40 accepted updates")
    return "structural"


def metadata(name, tensor, owner=None):
    shape = list(tensor.shape)
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid factor shape")
    dtype = str(tensor.dtype)
    if dtype not in ("torch.bfloat16", "torch.float16", "torch.float32"):
        raise ValueError("unsupported factor dtype")
    routed = is_routed_inner(name)
    if routed and (len(shape) != 3 or shape[0] != 896):
        raise ValueError("routed factor requires 896 experts")
    if owner is not None and (type(owner) is not int or not 0 <= owner < 32):
        raise ValueError("invalid EP32 owner")
    return {"shape": shape, "dtype": dtype, "owner": owner,
            "owned_shape": ([28, *shape[1:]] if routed else shape),
            "ep_size": 32 if routed else None}


def structural_receipt(name, tensor, owner=None):
    return PREFIX + json.dumps(metadata(name, tensor, owner), sort_keys=True, separators=(",", ":"))


def verify_receipt(name, received, expected, ep_rank):
    if not expected.startswith(PREFIX):
        return audited_verify(name, received, expected, ep_rank)
    if not received.startswith(PREFIX):
        return False
    try:
        want, got = (json.loads(value[len(PREFIX):]) for value in (expected, received))
        for item in (want, got):
            if set(item) != {"shape", "dtype", "owner", "owned_shape", "ep_size"}:
                return False
            shape = item["shape"]
            if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
                return False
            if item["dtype"] not in ("torch.bfloat16", "torch.float16", "torch.float32"):
                return False
            routed = is_routed_inner(name)
            if routed and (len(shape) != 3 or shape[0] != 896):
                return False
            if item["owned_shape"] != ([28, *shape[1:]] if routed else shape):
                return False
            if item["ep_size"] != (32 if routed else None):
                return False
        if type(ep_rank) is not int or not 0 <= ep_rank < 32:
            return False
        if want["owner"] is not None or got.get("owner") != ep_rank:
            return False
        got = dict(got, owner=None)
        return got == want
    except (ValueError, TypeError, KeyError):
        return False


def record_structural(bucket, checksums, profile):
    start = time.perf_counter()
    for key, tensor in bucket:
        adapter, name = key.split(":", 1)
        if name in checksums.setdefault(adapter, {}):
            raise ValueError("duplicate sender factor")
        checksums[adapter][name] = structural_receipt(name, tensor)
    if profile is not None:
        profile["metadata_s"] = profile.get("metadata_s", 0.0) + time.perf_counter() - start


def installed_audit(updater):
    """Qualification-only: read the live pool, never a sender/stash substitute."""
    manager = updater.tp_worker.model_runner.lora_manager
    # Streamed names identify a registered LoRARef; the live pool is keyed by
    # that ref\'s opaque lora_id, not by its public name.
    uid = manager.require_registered("miles_lora").lora_id
    pool = manager.memory_pool
    if uid in pool.invalid_uids:
        raise ValueError("cannot audit invalid live adapter")
    if uid not in pool.uid_to_buffer_id:
        raise ValueError("registered adapter is not resident")
    slot = pool.uid_to_buffer_id[uid]
    if (type(slot) is not int or not 0 <= slot < len(pool.buffer_id_to_uid)
            or pool.buffer_id_to_uid[slot] != uid):
        raise ValueError("registered adapter pool slot ownership mismatch")
    import torch
    values = {}
    copied = 0
    for category in ("A_buffer", "B_buffer", "embedding_A_buffer", "embedding_B_buffer",
                     "lm_head_A_buffer", "lm_head_B_buffer", "new_embeddings_buffer"):
        for key, buffers in sorted(getattr(pool, category).items()):
            if not isinstance(buffers, (list, tuple)):
                buffers = [buffers]
            for index, buffer in enumerate(buffers):
                value = buffer[slot].detach().to(device="cpu", copy=True).contiguous()
                wire = value.reshape(-1).view(torch.uint8).numpy()
                values[f"{category}/{key}/{index}"] = {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(memoryview(wire)).hexdigest()}
                copied += value.numel() * value.element_size()
    if not values or copied == 0:
        raise ValueError("empty installed adapter audit")
    return values, copied


def finish_receiver(updater, success, abort):
    """Collective installation guard and aggregate evidence,no tensor reads."""
    if abort:
        return success, "Aborted"
    try:
        if not success:
            raise ValueError("adapter installation rejected")
        version = updater._e008_publication_index
        manager = updater.tp_worker.model_runner.lora_manager
        uid = manager.require_registered("miles_lora").lora_id
        pool = manager.memory_pool
        if uid in pool.invalid_uids or uid not in pool.uid_to_buffer_id:
            raise ValueError("installed adapter invalid or nonresident")
        slot = pool.uid_to_buffer_id[uid]
        if (type(slot) is not int or not 0 <= slot < len(pool.buffer_id_to_uid)
                or pool.buffer_id_to_uid[slot] != uid):
            raise ValueError("installed adapter slot ownership mismatch")
        root = os.environ.get("E007_OPT_DIAG_ROOT", "")
        if not publication_root_matches(root):
            raise ValueError("receiver evidence root is not attempt-bound")
        profile = dict(updater._e007_publication_profile)
        profile.update(version=version, mode=publication_mode(version),
                       qualification_installed_audit_s=0.0,
                       qualification_installed_bytes=0, tensor_sha_enabled=False,
                       status="STRUCTURAL_INSTALL_PASS",
                       ep_rank=updater.tp_worker.ps.moe_ep_rank)
        target = Path(root) / f"receiver-{socket.gethostname()}-ep{profile['ep_rank']}-v{version}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as handle:
            json.dump(profile, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
        report = True, "Success"
    except Exception as error:
        report = False, f"[K3-PUBLICATION] {type(error).__name__}: {error}"
    import torch.distributed as dist
    reports = [None] * dist.get_world_size(group=updater.tp_cpu_group)
    dist.all_gather_object(reports, report, group=updater.tp_cpu_group)
    failures = [message for ok, message in reports if not ok]
    return (False, failures[0]) if failures else (True, "Success")

