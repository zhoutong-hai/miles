"""Byte-bounded, immutable private rollout snapshots with a completion marker last.

The requested .pt is a small manifest. Array chunks live in its .parts directory;
NumPy routing arrays never enter the metadata pickle. A single writer owns at
most one pending rollout and a bounded queue of copied CPU chunks.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import pickle
from queue import Full, Queue
import time

import numpy as np
import torch

FORMAT = "miles.rollout-shards.v1"
ARRAY_KEY = "__miles_private_array_v1__"


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _write_part(path, value):
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        if isinstance(value, bytes):
            stream.write(value)
        else:
            torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": path.stat().st_size, "sha256": _digest(path)}


class PrivateDumpWriter:
    def __init__(self, *, chunk_bytes=64 << 20, queue_chunks=2):
        if chunk_bytes < 1024 or queue_chunks < 1:
            raise ValueError("Private dump chunk/queue limits must be positive and useful")
        self.chunk_bytes = chunk_bytes
        self.queue_chunks = queue_chunks
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="private-rollout-dump")
        self._future = None
        self._closed = False

    def drain(self):
        if self._future is not None:
            self._future.result()

    def close(self):
        try:
            self.drain()
        finally:
            self._closed = True
            self._pool.shutdown(wait=True)

    def _put(self, queue, item):
        while True:
            if self._future.done():
                self._future.result()  # Propagate disk failures instead of deadlocking a full queue.
                raise RuntimeError("Private writer completed before the snapshot was submitted")
            try:
                queue.put(item, timeout=0.1)
                return
            except Full:
                continue

    def _freeze(self, value, queue, counter):
        if isinstance(value, (np.ndarray, torch.Tensor)):
            numpy = isinstance(value, np.ndarray)
            if not numpy and (value.device.type != "cpu" or not value.is_contiguous()):
                raise ValueError("Private tensor snapshots require contiguous CPU tensors")
            size = value.size if numpy else value.numel()
            itemsize = value.dtype.itemsize if numpy else value.element_size()
            per_chunk = max(1, self.chunk_bytes // itemsize)
            names = []
            for start in range(0, size, per_chunk):
                # flat slicing copies only this chunk, including for strided NumPy inputs.
                chunk = torch.from_numpy(value.flat[start : start + per_chunk]) if numpy else (
                    value.detach().view(-1)[start : start + per_chunk].clone()
                )
                name = f"array-{counter[0]:08d}.pt"
                counter[0] += 1
                self._put(queue, (name, chunk))
                names.append(name)
            return {ARRAY_KEY: {"kind": "numpy" if numpy else "torch", "shape": list(value.shape),
                                "dtype": str(value.dtype), "chunks": names}}
        if isinstance(value, dict):
            if ARRAY_KEY in value:
                raise ValueError("Private metadata contains the reserved array descriptor key")
            return {deepcopy(key): self._freeze(item, queue, counter) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, (int, float, str, bool, type(None))) for item in value):
                return tuple(value) if isinstance(value, tuple) else value.copy()
            items = [self._freeze(item, queue, counter) for item in value]
            return tuple(items) if isinstance(value, tuple) else items
        return deepcopy(value)

    def _metadata_part(self, value, name, queue, counter):
        frozen = self._freeze(value, queue, counter)
        blob = pickle.dumps(frozen, protocol=5)
        if len(blob) > self.chunk_bytes:
            raise ValueError("One private metadata record exceeds the configured chunk bound")
        self._put(queue, (name, blob))

    def submit(self, path, samples, *, rollout_id, evaluation, metadata):
        if self._closed:
            raise RuntimeError("Private dump writer is closed")
        self.drain()  # One unresolved artifact, including eval/train overlap.
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        parts = path.with_name(path.name + ".parts")
        parts.mkdir(mode=0o700, parents=True, exist_ok=False)
        queue = Queue(maxsize=self.queue_chunks)
        self._future = self._pool.submit(self._consume, path, parts, queue, rollout_id, evaluation)
        counter, sample_files = [0], []
        try:
            self._metadata_part(metadata or {}, "metadata.pkl", queue, counter)
            for index, sample in enumerate(samples):
                name = f"sample-{index:06d}.pkl"
                self._metadata_part(sample.to_dict(), name, queue, counter)
                sample_files.append(name)
            self._put(queue, (None, sample_files))
        except BaseException as error:
            if not self._future.done():
                self._put(queue, (False, error))
            raise
        return self._future

    def _consume(self, path, parts, queue, rollout_id, evaluation):
        started = time.monotonic()
        files = {}
        while True:
            name, value = queue.get()
            if name is False:
                raise value
            if name is None:
                sample_files = value
                break
            files[name] = _write_part(parts / name, value)
        manifest = {"format": FORMAT, "rollout_id": rollout_id, "evaluation": evaluation,
                    "state": "COMPLETE",
                    "provenance": {key: os.environ[key] for key in (
                        "MILES_SOURCE_COMMIT", "MILES_EXPERIMENT_RUN_ID", "MILES_EXPERIMENT_ATTEMPT_ID",
                        "MILES_INPUT_SHA256", "MILES_TOKENIZER_REVISION", "MILES_RESOLVED_CONFIG_SHA256",
                    ) if key in os.environ},
                    "sample_files": sample_files, "metadata_file": "metadata.pkl", "files": files,
                    "limits": {"chunk_bytes": self.chunk_bytes, "queue_chunks": self.queue_chunks},
                    "writer_seconds": time.monotonic() - started,
                    "payload_bytes": sum(item["bytes"] for item in files.values())}
        temporary = path.with_name(path.name + ".incomplete")
        _write_part(temporary, manifest)
        # An atomic, no-overwrite completion marker. Partial .parts directories are unaccepted.
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return manifest


def _read_part(parts, name, files, *, tensor=False):
    if Path(name).name != name or name not in files:
        raise ValueError("Invalid private dump part reference")
    path = parts / name
    expected = files[name]
    if path.stat().st_size != expected["bytes"] or _digest(path) != expected["sha256"]:
        raise ValueError("Private dump part integrity mismatch")
    if tensor:
        return torch.load(path, map_location="cpu", weights_only=True)
    with path.open("rb") as stream:
        return pickle.load(stream)


def _restore(value, parts, files):
    if isinstance(value, dict) and set(value) == {ARRAY_KEY}:
        desc = value[ARRAY_KEY]
        shape = tuple(desc["shape"])
        if desc["kind"] == "numpy":
            result = np.empty(shape, dtype=desc["dtype"])
            flat = result.reshape(-1)
        elif desc["kind"] == "torch":
            result = torch.empty(shape, dtype=getattr(torch, desc["dtype"].removeprefix("torch.")))
            flat = result.view(-1)
        else:
            raise ValueError("Unknown private array kind")
        offset = 0
        for name in desc["chunks"]:
            chunk = _read_part(parts, name, files, tensor=True)
            count = chunk.numel()
            if desc["kind"] == "numpy":
                if chunk.numpy().dtype != result.dtype:
                    raise ValueError("Private array dtype mismatch")
                flat[offset : offset + count] = chunk.numpy()
            else:
                if chunk.dtype != result.dtype:
                    raise ValueError("Private tensor dtype mismatch")
                flat[offset : offset + count].copy_(chunk)
            offset += count
        flat_size = flat.size if isinstance(flat, np.ndarray) else flat.numel()
        if offset != flat_size:
            raise ValueError("Private array length mismatch")
        return result
    if isinstance(value, dict):
        return {key: _restore(item, parts, files) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_restore(item, parts, files) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


def iter_private_samples(path, manifest):
    parts = Path(path).with_name(Path(path).name + ".parts")
    for name in manifest["sample_files"]:
        yield _restore(_read_part(parts, name, manifest["files"]), parts, manifest["files"])


def load_rollout_payload(path, *, mmap=False):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    if payload.get("format") != FORMAT:
        return payload  # Existing monolithic debug dumps remain readable.
    parts = Path(path).with_name(Path(path).name + ".parts")
    if {item.name for item in parts.iterdir()} != set(payload["files"]):
        raise ValueError("Private dump inventory differs from its completed manifest")
    metadata = _restore(_read_part(parts, payload["metadata_file"], payload["files"]), parts, payload["files"])
    return {"rollout_id": payload["rollout_id"], "metadata": metadata,
            "samples": list(iter_private_samples(path, payload))}
