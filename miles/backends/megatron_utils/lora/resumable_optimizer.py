"""Same-topology LoRA master/moment persistence, derived from the qualified K3 recovery writer."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_OPTIMIZER_STEP = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"LoRA optimizer receipt already exists: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.chmod(0o600)
    os.link(temporary, path)
    temporary.unlink()


def _iter_suboptimizers(optimizer: Any) -> list[Any]:
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        return [optimizer]
    if not children:
        raise RuntimeError("LoRA optimizer has an empty chained_optimizers list")
    result = []
    for child in children:
        result.extend(_iter_suboptimizers(child))
    return result


def _clone_cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(child) for child in value)
    return value


def _snapshot_rows(
    snapshot: Mapping[Any, Any], suboptimizer_index: int
) -> list[dict[str, Any]]:
    rows = []
    for gbuf_index in sorted(key for key in snapshot if isinstance(key, int)):
        dtype_state = snapshot[gbuf_index]
        if not isinstance(dtype_state, Mapping) or len(dtype_state) != 1:
            raise RuntimeError(
                f"LoRA optimizer gbuf {gbuf_index} must contain exactly one dtype"
            )
        for dtype, buckets in dtype_state.items():
            for bucket_index, bucket in enumerate(buckets):
                for item in bucket:
                    rows.append(
                        {
                            "key": (
                                suboptimizer_index,
                                int(gbuf_index),
                                str(dtype),
                                int(bucket_index),
                                int(item["gbuf_local_start"]),
                                int(item["gbuf_local_end"]),
                            ),
                            "item": item,
                        }
                    )
    return rows


def _checkpoint_snapshot(snapshot: Mapping[Any, Any]) -> dict[Any, Any]:
    saved = _clone_cpu(snapshot)
    for row in _snapshot_rows(saved, 0):
        row["item"]["padding"] = False
        # The inverse Megatron loader explicitly skips this derived field.
        row["item"].pop("step", None)
    return saved


def _rehydrate_like(saved: Any, template: Any) -> Any:
    import torch

    if isinstance(saved, torch.Tensor):
        if not isinstance(template, torch.Tensor):
            raise RuntimeError("i004 saved optimizer tensor has no live template")
        return saved.to(device=template.device, dtype=template.dtype)
    if isinstance(saved, Mapping):
        result = {}
        for key, child in saved.items():
            if key == "padding":
                if child is not False:
                    raise RuntimeError("LoRA optimizer snapshot has non-false padding")
                result[key] = False
            else:
                if not isinstance(template, Mapping) or key not in template:
                    raise RuntimeError(
                        f"LoRA optimizer snapshot structure drift at key {key!r}"
                    )
                result[key] = _rehydrate_like(child, template[key])
        return result
    if isinstance(saved, list):
        if not isinstance(template, list) or len(saved) != len(template):
            raise RuntimeError("LoRA optimizer snapshot list structure drift")
        return [
            _rehydrate_like(child, template[index])
            for index, child in enumerate(saved)
        ]
    if isinstance(saved, tuple):
        if not isinstance(template, tuple) or len(saved) != len(template):
            raise RuntimeError("LoRA optimizer snapshot tuple structure drift")
        return tuple(
            _rehydrate_like(child, template[index])
            for index, child in enumerate(saved)
        )
    return saved


def _tensor_record(tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"i004 expected optimizer tensor, got {type(tensor).__name__}"
        )
    value = tensor.detach().contiguous()
    raw = value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(order="C")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": int(value.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(torch.isfinite(value).all().item()),
        "nonzero": int(torch.count_nonzero(value).item()),
    }


def _state_manifest(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_record(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _state_manifest(child)
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_state_manifest(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _scalar_step(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise RuntimeError("LoRA optimizer step is not one finite scalar")
        value = value.item()
    result = int(value)
    if float(value) != float(result):
        raise RuntimeError(f"LoRA optimizer step is not integral: {value!r}")
    return result


def _model_parameter_names(
    models: Sequence[Any],
) -> tuple[dict[int, str], list[str]]:
    names = {}
    trainable = []
    for chunk_index, chunk in enumerate(models):
        for name, parameter in chunk.named_parameters():
            fq_name = f"chunk{chunk_index}.{name}"
            if id(parameter) in names:
                raise RuntimeError(f"i004 model parameter has duplicate name: {fq_name}")
            names[id(parameter)] = fq_name
            if parameter.requires_grad:
                if "lora" not in name.lower() and ".adapter." not in name:
                    raise RuntimeError(
                        f"LoRA optimizer model has non-LoRA trainable parameter: {fq_name}"
                    )
                trainable.append(fq_name)
    return names, sorted(trainable)


def _validate_dp_interval_union(
    payloads: Sequence[Mapping[str, Any]], suboptimizer_index: int
) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("LoRA optimizer DP coverage has no rank payloads")
    ordered = sorted(payloads, key=lambda row: int(row["physical_rank"]))
    ranks = [int(row["physical_rank"]) for row in ordered]
    if len(set(ranks)) != len(ranks):
        raise RuntimeError("LoRA optimizer DP coverage repeats a physical rank")
    expected = ordered[0].get("expected_parameters")
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeError("LoRA optimizer DP coverage has no expected parameters")
    expected = {str(name): int(size) for name, size in expected.items()}
    if any(size <= 0 for size in expected.values()):
        raise RuntimeError("LoRA optimizer DP coverage has a non-positive parameter")
    for row in ordered:
        if row.get("expected_parameters") != expected:
            raise RuntimeError(
                "LoRA optimizer DP peers disagree on expected parameter inventory"
            )
    intervals: dict[str, list[tuple[int, int, int]]] = {
        name: [] for name in expected
    }
    for row in ordered:
        rank = int(row["physical_rank"])
        rank_rows = row.get("owned_intervals")
        if not isinstance(rank_rows, list):
            raise RuntimeError("LoRA optimizer DP rank has invalid interval rows")
        for interval in rank_rows:
            if interval.get("suboptimizer") != suboptimizer_index:
                raise RuntimeError("LoRA optimizer DP interval suboptimizer drift")
            name = interval.get("name")
            if name not in expected:
                raise RuntimeError(f"LoRA optimizer DP interval has unknown name: {name}")
            start = int(interval.get("param_start", -1))
            end = int(interval.get("param_end", -1))
            if not 0 <= start < end <= expected[name]:
                raise RuntimeError(
                    f"LoRA optimizer DP interval is invalid for {name}: {start}:{end}"
                )
            intervals[name].append((start, end, rank))
    normalized = []
    for name, size in sorted(expected.items()):
        cursor = 0
        rows = sorted(intervals[name])
        if not rows:
            raise RuntimeError(f"LoRA optimizer DP coverage entirely misses {name}")
        for start, end, rank in rows:
            if start != cursor:
                kind = "overlap" if start < cursor else "gap"
                raise RuntimeError(
                    f"LoRA optimizer DP coverage {kind} for {name}: "
                    f"cursor={cursor} interval={start}:{end} rank={rank}"
                )
            normalized.append(
                {"name": name, "start": start, "end": end, "physical_rank": rank}
            )
            cursor = end
        if cursor != size:
            raise RuntimeError(
                f"LoRA optimizer DP coverage truncates {name}: {cursor} != {size}"
            )
    proof = {
        "suboptimizer": suboptimizer_index,
        "dp_group_ranks": ranks,
        "expected_parameters": dict(sorted(expected.items())),
        "intervals": normalized,
    }
    return {
        "suboptimizer": suboptimizer_index,
        "dp_group_ranks": ranks,
        "dp_group_size": len(ranks),
        "expected_parameter_count": len(expected),
        "expected_parameter_elements": sum(expected.values()),
        "interval_count": len(normalized),
        "coverage_sha256": _canonical_sha256(proof),
        "complete_no_gap_no_overlap": True,
    }


def _gather_dp_interval_coverage(
    suboptimizer: Any,
    suboptimizer_index: int,
    names_by_id: Mapping[int, str],
    local_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import torch.distributed as dist

    expected: dict[str, int] = {}
    buffers = getattr(suboptimizer, "buffers", None)
    if not isinstance(buffers, list) or not buffers:
        raise RuntimeError("LoRA optimizer lacks the full parameter-buffer inventory")
    for buffer in buffers:
        for parameter in getattr(buffer, "params", ()):
            if not parameter.requires_grad:
                continue
            name = names_by_id.get(id(parameter))
            if name is None:
                raise RuntimeError("LoRA optimizer buffer contains an unnamed parameter")
            if name in expected:
                raise RuntimeError(f"LoRA optimizer buffers repeat parameter {name}")
            expected[name] = int(parameter.numel())
    payload = {
        "physical_rank": dist.get_rank() if dist.is_initialized() else 0,
        "expected_parameters": dict(sorted(expected.items())),
        "owned_intervals": [
            {
                "name": row["name"],
                "suboptimizer": row["suboptimizer"],
                "param_start": row["param_start"],
                "param_end": row["param_end"],
            }
            for row in local_entries
        ],
    }
    if not dist.is_initialized():
        return _validate_dp_interval_union([payload], suboptimizer_index)
    group = getattr(suboptimizer, "data_parallel_group_gloo", None)
    if group is None:
        raise RuntimeError("LoRA optimizer lacks a Gloo DP checkpoint group")
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, payload, group=group)
    return _validate_dp_interval_union(gathered, suboptimizer_index)


def _manifest_categories(content: Mapping[str, Any]) -> dict[str, Any]:
    entries = content.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("LoRA optimizer manifest entries are invalid")
    optimizer_step = content.get("optimizer_step")
    identity = []
    tensors = []
    steps = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("LoRA optimizer manifest entry is invalid")
        locator = {
            key: entry.get(key)
            for key in ("name", "suboptimizer", "param_start", "param_end")
        }
        identity.append(
            {
                key: value
                for key, value in entry.items()
                if key
                not in {
                    "tensors",
                    "checkpoint_param_group_step",
                    "state_step",
                    "param_group_step",
                }
            }
        )
        entry_tensors = entry.get("tensors")
        if not isinstance(entry_tensors, Mapping):
            raise RuntimeError("LoRA optimizer manifest tensors are invalid")
        # Megatron's reshardable snapshot omits the derived Adam step tensor
        # before serialization, while loading the optimizer state recreates it.
        # Parameter and moment tensors remain byte-authoritative; the scalar
        # step is compared semantically below from the same live snapshot.
        tensors.append(
            {
                **locator,
                "tensors": {
                    name: value
                    for name, value in entry_tensors.items()
                    if name != "step"
                },
            }
        )
        steps.append(
            {
                **locator,
                "checkpoint_param_group_step": entry.get(
                    "checkpoint_param_group_step"
                ),
                # Megatron restores Adam's clock from the optimizer group and
                # explicitly ignores the per-parameter snapshot ``step``.
                "state_step": optimizer_step,
                "param_group_step": (
                    optimizer_step
                    if entry.get("param_group_step") is None
                    else entry.get("param_group_step")
                ),
            }
        )
    topology_keys = (
        "schema_version",
        "physical_rank",
        "comparison_scope",
        "trainable_tensor_count",
        "owned_shard_count",
        "unowned_trainable_names",
        "dp_interval_coverage",
    )
    return {
        "topology_and_coverage": {
            key: content.get(key) for key in topology_keys
        },
        "entry_identity": identity,
        "optimizer_tensors": tensors,
        "optimizer_steps": {
            "optimizer_step": optimizer_step,
            "entries": steps,
        },
        "scheduler_state": content.get("scheduler_state"),
        "optimizer_metadata": content.get("optimizer_metadata"),
        "moment_population": content.get("nonzero_moment_tensors"),
    }


def _brief_manifest_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    if isinstance(value, (Mapping, list, tuple)):
        return {
            "type": type(value).__name__,
            "items": len(value),
            "sha256": _canonical_sha256(value),
        }
    return {"type": type(value).__name__, "repr": repr(value)[:160]}


def _bounded_manifest_differences(
    expected: Any,
    restored: Any,
    *,
    path: str = "$",
    limit: int = 24,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, location: str) -> None:
        if len(differences) >= limit or left == right:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right), key=str):
                if len(differences) >= limit:
                    return
                child = f"{location}.{key}"
                if key not in left:
                    differences.append(
                        {
                            "path": child,
                            "kind": "unexpected",
                            "restored": _brief_manifest_value(right[key]),
                        }
                    )
                elif key not in right:
                    differences.append(
                        {
                            "path": child,
                            "kind": "missing",
                            "expected": _brief_manifest_value(left[key]),
                        }
                    )
                else:
                    visit(left[key], right[key], child)
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                differences.append(
                    {
                        "path": location,
                        "kind": "length",
                        "expected": len(left),
                        "restored": len(right),
                    }
                )
            for index, (left_child, right_child) in enumerate(zip(left, right)):
                if len(differences) >= limit:
                    return
                visit(left_child, right_child, f"{location}[{index}]")
            return
        differences.append(
            {
                "path": location,
                "kind": "value",
                "expected": _brief_manifest_value(left),
                "restored": _brief_manifest_value(right),
            }
        )

    visit(expected, restored, path)
    return differences


def _restore_manifest_diagnostic(
    expected: Mapping[str, Any], restored: Mapping[str, Any]
) -> dict[str, Any]:
    expected_categories = _manifest_categories(expected)
    restored_categories = _manifest_categories(restored)
    categories = {}
    for name in expected_categories:
        expected_sha256 = _canonical_sha256(expected_categories[name])
        restored_sha256 = _canonical_sha256(restored_categories[name])
        categories[name] = {
            "equal": expected_sha256 == restored_sha256,
            "expected_sha256": expected_sha256,
            "restored_sha256": restored_sha256,
        }
    return {
        "categories": categories,
        "mismatched_categories": [
            name for name, value in categories.items() if not value["equal"]
        ],
        "bounded_differences": _bounded_manifest_differences(expected, restored),
        "difference_limit": 24,
        "contains_tensor_values": False,
    }


def _optimizer_manifest(
    models: Sequence[Any], optimizer: Any, scheduler: Any, *, retain_raw_snapshots: bool = True
) -> dict[str, Any]:
    import torch

    names_by_id, trainable_names = _model_parameter_names(models)
    entries = []
    owned_names = set()
    raw_snapshots = []
    dp_interval_coverage = []
    nonzero_moment_tensors = {"exp_avg": 0, "exp_avg_sq": 0}
    for suboptimizer_index, suboptimizer in enumerate(_iter_suboptimizers(optimizer)):
        entry_start = len(entries)
        required = (
            "model_param_gbuf_map",
            "model_param_group_index_map",
            "_get_model_param_range_map",
            "get_parameter_state_dp_reshardable",
            "load_parameter_state_from_dp_reshardable",
            "optimizer",
            "state_dict",
        )
        missing = [name for name in required if not hasattr(suboptimizer, name)]
        if missing:
            raise RuntimeError(
                f"i004 requires DistributedOptimizer APIs, missing {missing}"
            )
        checkpoint_state = suboptimizer.state_dict()
        checkpoint_groups = checkpoint_state.get("optimizer", {}).get(
            "param_groups", []
        )
        if not isinstance(checkpoint_groups, list):
            raise RuntimeError("LoRA optimizer checkpoint param_groups is not a list")
        snapshot = suboptimizer.get_parameter_state_dp_reshardable()
        if retain_raw_snapshots:
            raw_snapshots.append(snapshot)
        rows = _snapshot_rows(snapshot, suboptimizer_index)
        by_key = {row["key"]: row["item"] for row in rows}
        if len(by_key) != len(rows):
            raise RuntimeError("LoRA optimizer snapshot has duplicate bucket locations")
        inner = suboptimizer.optimizer
        gbuf_parameters = set(suboptimizer.model_param_gbuf_map)
        group_parameters = set(suboptimizer.model_param_group_index_map)
        if gbuf_parameters != group_parameters:
            raise RuntimeError("LoRA optimizer local parameter ownership is incomplete")
        for model_parameter, location in suboptimizer.model_param_gbuf_map.items():
            fq_name = names_by_id.get(id(model_parameter))
            if fq_name is None or not model_parameter.requires_grad:
                raise RuntimeError("LoRA optimizer owns an unnamed or frozen parameter")
            if fq_name in owned_names:
                raise RuntimeError(f"LoRA optimizer owns a parameter twice: {fq_name}")
            owned_names.add(fq_name)
            gbuf_index, dtype, bucket_index = location
            ranges = suboptimizer._get_model_param_range_map(model_parameter)
            local_range = ranges["gbuf_local"]
            key = (
                suboptimizer_index,
                int(gbuf_index),
                str(dtype),
                int(bucket_index),
                int(local_range.start),
                int(local_range.end),
            )
            item = by_key.pop(key, None)
            if item is None:
                raise RuntimeError(
                    f"LoRA optimizer snapshot has no named binding for {fq_name}: {key}"
                )
            tensor_keys = {
                name for name, value in item.items() if isinstance(value, torch.Tensor)
            }
            required_tensors = {"param", "exp_avg", "exp_avg_sq"}
            if not required_tensors.issubset(tensor_keys):
                raise RuntimeError(
                    f"LoRA optimizer state for {fq_name} lacks "
                    f"{sorted(required_tensors - tensor_keys)}"
                )
            records = {name: _tensor_record(item[name]) for name in sorted(tensor_keys)}
            for name, record in records.items():
                if not record["finite"]:
                    raise RuntimeError(
                        f"LoRA optimizer {name} is non-finite for {fq_name}"
                    )
            for name in required_tensors:
                if records[name]["dtype"] != "torch.float32":
                    raise RuntimeError(
                        f"LoRA optimizer {name} is not FP32 for {fq_name}"
                    )
            for moment in ("exp_avg", "exp_avg_sq"):
                nonzero_moment_tensors[moment] += int(records[moment]["nonzero"] > 0)
            group_index, group_order = suboptimizer.model_param_group_index_map[
                model_parameter
            ]
            if group_index >= len(checkpoint_groups):
                raise RuntimeError(
                    f"LoRA optimizer checkpoint lacks group {group_index}: {fq_name}"
                )
            checkpoint_group = checkpoint_groups[group_index]
            checkpoint_step = _scalar_step(checkpoint_group.get("step"))
            if checkpoint_step != EXPECTED_OPTIMIZER_STEP:
                raise RuntimeError(
                    f"LoRA optimizer checkpoint step for {fq_name} is "
                    f"{checkpoint_step}, expected {EXPECTED_OPTIMIZER_STEP}"
                )
            group = inner.param_groups[group_index]
            state_step = _scalar_step(item["step"]) if "step" in item else None
            group_step = (
                _scalar_step(group["step"]) if group.get("step") is not None else None
            )
            # ``checkpoint_group`` is the clock Megatron serializes
            # and restores, so its exact check above is authoritative. HDO also
            # exposes live per-parameter and inner-group clocks. After an
            # update-4 restore either live representation may retain the load
            # value while parameters and moments continue to update. Keep both
            # as diagnostics, but only require a sane range here.
            if group_step is not None and not (
                0 <= group_step <= EXPECTED_OPTIMIZER_STEP
            ):
                raise RuntimeError(
                    f"LoRA optimizer group step outside verified range for "
                    f"{fq_name}: {group_step} not in "
                    f"[0, {EXPECTED_OPTIMIZER_STEP}]"
                )
            if state_step is not None and not (
                0 <= state_step <= EXPECTED_OPTIMIZER_STEP
            ):
                raise RuntimeError(
                    f"LoRA optimizer parameter step outside verified range for "
                    f"{fq_name}: {state_step} not in "
                    f"[0, {EXPECTED_OPTIMIZER_STEP}]"
                )
            param_range = ranges["param"]
            entries.append(
                {
                    "name": fq_name,
                    "suboptimizer": suboptimizer_index,
                    "optimizer_group": int(group_index),
                    "optimizer_order": int(group_order),
                    "gbuf_index": int(gbuf_index),
                    "dtype": str(dtype),
                    "bucket_index": int(bucket_index),
                    "param_start": int(param_range.start),
                    "param_end": int(param_range.end),
                    "gbuf_local_start": int(local_range.start),
                    "gbuf_local_end": int(local_range.end),
                    "checkpoint_param_group_step": checkpoint_step,
                    "state_step": state_step,
                    "param_group_step": group_step,
                    "tensors": records,
                }
            )
        if by_key:
            raise RuntimeError(
                f"LoRA optimizer snapshot has {len(by_key)} unnamed bucket rows"
            )
        dp_interval_coverage.append(
            _gather_dp_interval_coverage(
                suboptimizer,
                suboptimizer_index,
                names_by_id,
                entries[entry_start:],
            )
        )
        if not retain_raw_snapshots:
            del snapshot, rows, by_key
            gc.collect()
            torch.cuda.empty_cache()
    unowned_trainable_names = sorted(set(trainable_names) - owned_names)
    if any(value <= 0 for value in nonzero_moment_tensors.values()):
        raise RuntimeError(
            f"LoRA optimizer has no post-update moment signal: {nonzero_moment_tensors}"
        )
    entries.sort(key=lambda row: (row["name"], row["suboptimizer"], row["param_start"]))
    content = {
        "schema_version": SCHEMA_VERSION,
        "physical_rank": int(torch.distributed.get_rank()),
        "comparison_scope": "same_topology_same_physical_global_rank_only",
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "trainable_tensor_count": len(trainable_names),
        "owned_shard_count": len(entries),
        "unowned_trainable_names": unowned_trainable_names,
        "dp_interval_coverage": dp_interval_coverage,
        "nonzero_moment_tensors": nonzero_moment_tensors,
        "optimizer_metadata": _state_manifest(optimizer.state_dict()),
        "scheduler_state": _state_manifest(
            scheduler.state_dict() if scheduler is not None else None
        ),
        "entries": entries,
    }
    return {
        "content": content,
        "sha256": _canonical_sha256(content),
        "raw_snapshots": raw_snapshots,
    }


def save_resumable_optimizer_state(
    adapter_dir: Path,
    *,
    iteration: int,
    model: Sequence[Any],
    optimizer: Any,
    scheduler: Any,
) -> dict[str, Any]:
    """Persist full DP-owned master parameters and Adam moments on every rank."""
    import torch
    import torch.distributed as dist

    global EXPECTED_OPTIMIZER_STEP
    EXPECTED_OPTIMIZER_STEP = int(iteration) + 1
    if EXPECTED_OPTIMIZER_STEP not in (20, 40):
        raise RuntimeError("E012 selected full-state saves are updates20 and40 only")
    rank = dist.get_rank() if dist.is_initialized() else 0
    manifest = _optimizer_manifest(model, optimizer, scheduler)
    raw_snapshots = manifest.pop("raw_snapshots")
    filename = f"optimizer_full_state_rank{rank}.pt"
    target = adapter_dir / filename
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": os.environ["MILES_EXPERIMENT_RUN_ID"],
        "physical_rank": rank,
        "comparison_scope": "same_topology_same_physical_global_rank_only",
        "iteration": int(iteration),
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "parameter_state": [
            _checkpoint_snapshot(snapshot) for snapshot in raw_snapshots
        ],
        "manifest": manifest,
    }
    temporary = target.with_suffix(f".pt.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.chmod(0o600)
    temporary.replace(target)
    from miles.backends.megatron_utils.lora.terminal_checkpoint import seal_file
    sealed_file = seal_file(target)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "filename": filename,
        "sha256": sealed_file["sha256"],
        "bytes": sealed_file["bytes"],
        "file_identity": sealed_file["file_identity"],
        "file_seal_seconds": sealed_file["seal_seconds"],
        "physical_rank": rank,
        "iteration": int(iteration),
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "manifest_sha256": manifest["sha256"],
        "trainable_tensor_count": manifest["content"]["trainable_tensor_count"],
        "owned_shard_count": manifest["content"]["owned_shard_count"],
        "unowned_trainable_names": manifest["content"]["unowned_trainable_names"],
        "dp_interval_coverage": manifest["content"]["dp_interval_coverage"],
        "nonzero_moment_tensors": manifest["content"]["nonzero_moment_tensors"],
        "all_finite": True,
        "same_topology_restore_only": True,
    }
    return summary


def validate_saved_optimizer_reference(
    adapter_dir: Path, value: Any, *, rank: int, iteration: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("i004 checkpoint lacks full optimizer-state reference")
    required = {
        "schema_version": SCHEMA_VERSION,
        "physical_rank": rank,
        "iteration": iteration,
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "all_finite": True,
        "same_topology_restore_only": True,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise RuntimeError(
                f"LoRA optimizer-state reference drift: {key}={value.get(key)!r}"
            )
    moments = value.get("nonzero_moment_tensors")
    if not isinstance(moments, Mapping) or any(
        int(moments.get(name, 0)) <= 0 for name in ("exp_avg", "exp_avg_sq")
    ):
        raise RuntimeError(f"i004 checkpoint has no Adam moment signal: {moments}")
    coverage = value.get("dp_interval_coverage")
    if not isinstance(coverage, list) or not coverage or any(
        not row.get("complete_no_gap_no_overlap")
        or int(row.get("expected_parameter_count", 0)) <= 0
        or int(row.get("interval_count", 0)) <= 0
        for row in coverage
    ):
        raise RuntimeError("i004 checkpoint lacks complete DP interval coverage")
    filename = value.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError("LoRA optimizer-state filename is invalid")
    path = adapter_dir / filename
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(value.get("bytes", -1))
        or _sha256(path) != value.get("sha256")
    ):
        raise RuntimeError(f"LoRA optimizer-state file integrity failed: {path}")
    return dict(value)


def _restore_model_parameters_from_optimizer(
    suboptimizers: Sequence[Any],
) -> dict[str, Any]:
    """Mirror the normal post-step master-to-model copy and synchronous DP gather."""
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
    hdo_parameter_count = 0
    ordinary_leaf_count = 0
    synchronized_leaf_count = 0
    for suboptimizer in suboptimizers:
        if bool(getattr(suboptimizer, "is_stub_optimizer", False)):
            continue
        inner = getattr(suboptimizer, "optimizer", None)
        if inner is None:
            raise RuntimeError("LoRA optimizer leaf lacks its inner optimizer")
        mapping = getattr(inner, "param_to_inner_param", None)
        if mapping is not None:
            if not isinstance(mapping, Mapping) or not mapping:
                raise RuntimeError("i004 HDO parameter mapping is empty or invalid")
            originals = [
                parameter
                for group in inner.param_groups
                for parameter in group["params"]
            ]
            if len(originals) != len(mapping) or {
                id(parameter) for parameter in originals
            } != {id(parameter) for parameter in mapping}:
                raise RuntimeError("i004 HDO parameter mapping coverage drift")
            inverse = getattr(inner, "inner_param_to_orig_param", None)
            if not isinstance(inverse, Mapping) or len(inverse) != len(mapping):
                raise RuntimeError("i004 HDO inverse parameter mapping drift")
            h2d_stream = getattr(inner, "_h2d_stream", None)
            if h2d_stream is None:
                raise RuntimeError("i004 HDO restore lacks its H2D stream")
            current_stream = torch.cuda.current_stream()
            h2d_stream.wait_stream(current_stream)
            with torch.cuda.stream(h2d_stream):
                for original in originals:
                    restored_inner = mapping.get(original)
                    if restored_inner is None or inverse.get(restored_inner) is not original:
                        raise RuntimeError("i004 HDO parameter mapping is not bijective")
                    if original.shape != restored_inner.shape:
                        raise RuntimeError("i004 HDO restored parameter shape drift")
                    original.data.copy_(restored_inner.data, non_blocking=True)
            h2d_stream.record_event().wait(current_stream)
            torch.cuda.synchronize()
            hdo_parameter_count += len(originals)
        else:
            copy_main = getattr(
                suboptimizer, "_copy_main_params_to_model_params", None
            )
            if not callable(copy_main):
                raise RuntimeError(
                    "LoRA optimizer lacks the supported master-to-model copy API"
                )
            copy_main()
            ordinary_leaf_count += 1
        sync_parameters = getattr(
            suboptimizer, "start_param_sync_for_bucket_group_subset", None
        )
        if not callable(sync_parameters):
            raise RuntimeError(
                "LoRA optimizer lacks the supported synchronous parameter API"
            )
        sync_parameters(force_sync=True)
        synchronized_leaf_count += 1
    if synchronized_leaf_count == 0:
        raise RuntimeError("i004 restored no non-stub optimizer leaves")
    if dist.is_initialized():
        dist.barrier()
    return {
        "hdo_parameter_count": hdo_parameter_count,
        "ordinary_leaf_count": ordinary_leaf_count,
        "synchronized_leaf_count": synchronized_leaf_count,
    }


def _verify_restored_model_against_native_adapter(
    adapter_dir: Path, model: Sequence[Any]
) -> dict[str, Any]:
    """Require the synchronized live adapter to equal this rank's saved source."""
    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    path = adapter_dir / f"adapter_megatron_rank{rank}.pt"
    try:
        source = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        source = torch.load(path, map_location="cpu", weights_only=True)
    live = {
        name: parameter
        for chunk in model
        for name, parameter in chunk.named_parameters()
        if name in source
    }
    if set(live) != set(source):
        raise RuntimeError(
            "i004 post-restore adapter inventory differs from its native source"
        )
    categories: dict[str, int] = {}
    for name in sorted(source):
        expected = source[name]
        observed = live[name].detach()
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            raise RuntimeError(
                f"i004 post-restore adapter metadata mismatch: {name}"
            )
        if not torch.equal(observed, expected.to(device=observed.device)):
            raise RuntimeError(
                f"i004 post-restore adapter differs from saved source: {name}"
            )
        lowered = name.lower()
        if "shared_expert" in lowered:
            category = "shared_expert"
        elif ".experts." in lowered:
            category = "routed_expert"
        elif "self_attention" in lowered or ".attention." in lowered:
            category = "attention"
        elif ".mlp." in lowered:
            category = "dense_mlp"
        else:
            category = "other"
        categories[category] = categories.get(category, 0) + 1

    from miles.backends.training_utils.parallel import get_parallel_state

    replica_entries = []
    for name, parameter in sorted(live.items()):
        if getattr(parameter, "allreduce", True) is False:
            continue
        value = parameter.detach().contiguous()
        digest = hashlib.sha256()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode()
            + b"\0"
        )
        digest.update(value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        replica_entries.append((name, digest.hexdigest()))
    replica_digest = _canonical_sha256(replica_entries)
    dp = get_parallel_state().effective_dp
    gathered: list[Any] = [None] * dp.size
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "tensor_count": len(replica_entries),
            "sha256": replica_digest,
        },
        group=dp.group,
    )
    if len({row["tensor_count"] for row in gathered}) != 1 or len(
        {row["sha256"] for row in gathered}
    ) != 1:
        raise RuntimeError(
            "i004 synchronized non-routed adapter differs across its DP group"
        )
    return {
        "source_rank": rank,
        "tensor_count": len(source),
        "category_tensor_counts": dict(sorted(categories.items())),
        "exact_saved_source_match": True,
        "non_routed_dp_replica_count": len(replica_entries),
        "non_routed_dp_replica_sha256": replica_digest,
        "dp_group_members": sorted(row["rank"] for row in gathered),
        "routed_expert_ownership_preserved": categories.get("routed_expert", 0) > 0,
    }


def restore_resumable_optimizer_state(
    adapter_dir: Path,
    training_state: Mapping[str, Any],
    *,
    model: Sequence[Any],
    optimizer: Any,
    scheduler: Any,
) -> dict[str, Any]:
    """Restore and hash-prove one same-rank, same-topology optimizer shard."""
    import torch
    import torch.distributed as dist

    global EXPECTED_OPTIMIZER_STEP
    rank = dist.get_rank() if dist.is_initialized() else 0
    iteration = int(training_state.get("iteration", -1))
    EXPECTED_OPTIMIZER_STEP = iteration + 1
    if EXPECTED_OPTIMIZER_STEP not in (20, 40):
        raise RuntimeError("E012 restore expects selected update20 or40")
    summary = validate_saved_optimizer_reference(
        adapter_dir,
        training_state.get("optimizer_full_state"),
        rank=rank,
        iteration=iteration,
    )
    path = adapter_dir / summary["filename"]
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_run_id = training_state["provenance"]["run_id"]
    if not (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("run_id") == expected_run_id
        and payload.get("physical_rank") == rank
        and payload.get("iteration") == iteration
        and payload.get("optimizer_step") == EXPECTED_OPTIMIZER_STEP
        and payload.get("comparison_scope")
        == "same_topology_same_physical_global_rank_only"
    ):
        raise RuntimeError(f"LoRA optimizer payload identity drift on rank {rank}")
    suboptimizers = _iter_suboptimizers(optimizer)
    saved_parameter_state = payload.pop("parameter_state", None)
    if not isinstance(saved_parameter_state, list) or len(
        saved_parameter_state
    ) != len(suboptimizers):
        raise RuntimeError("LoRA optimizer suboptimizer coverage drift")
    expected_manifest = payload.get("manifest")
    if (
        not isinstance(expected_manifest, Mapping)
        or not isinstance(expected_manifest.get("content"), Mapping)
        or expected_manifest.get("sha256")
        != _canonical_sha256(expected_manifest["content"])
    ):
        raise RuntimeError("i004 saved optimizer manifest integrity failed")
    expected_manifest_sha256 = expected_manifest["sha256"]
    expected_manifest_content = expected_manifest["content"]
    optimizer.load_state_dict(training_state["optimizer"])
    if scheduler is not None:
        scheduler_state = training_state.get("opt_param_scheduler")
        if scheduler_state is None:
            raise RuntimeError("LoRA optimizer checkpoint lacks scheduler state")
        scheduler.load_state_dict(scheduler_state)
    for index, (suboptimizer, saved) in enumerate(
        zip(suboptimizers, saved_parameter_state, strict=True)
    ):
        template = suboptimizer.get_parameter_state_dp_reshardable()
        live = _rehydrate_like(saved, template)
        saved_parameter_state[index] = None
        del saved, template
        suboptimizer.load_parameter_state_from_dp_reshardable(live)
        inner = suboptimizer.optimizer
        if hasattr(inner, "_sync_hdo_state_to_sub_optimizers"):
            inner._sync_hdo_state_to_sub_optimizers()
        del live
        gc.collect()
        torch.cuda.empty_cache()
    del saved_parameter_state, payload
    gc.collect()
    torch.cuda.empty_cache()
    restored = _optimizer_manifest(
        model, optimizer, scheduler, retain_raw_snapshots=False
    )
    restored.pop("raw_snapshots", None)
    manifest_representation_normalized = False
    diagnostic = None
    if restored["sha256"] != expected_manifest_sha256:
        diagnostic = _restore_manifest_diagnostic(
            expected_manifest_content, restored["content"]
        )
        if diagnostic["mismatched_categories"]:
            diagnostic.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "FAIL",
                    "phase": "fresh_process_same_topology_optimizer_restore_diagnostic",
                    "run_id": os.environ["MILES_EXPERIMENT_RUN_ID"],
                    "global_rank": rank,
                    "optimizer_step": EXPECTED_OPTIMIZER_STEP,
                    "expected_manifest_sha256": expected_manifest_sha256,
                    "restored_manifest_sha256": restored["sha256"],
                }
            )
            evidence_dir = os.environ.get("MILES_I004_RESTORE_EVIDENCE_DIR")
            if not evidence_dir:
                raise RuntimeError("i004 restore mismatch has no evidence directory")
            root = Path(evidence_dir)
            if root.name != "optimizer-restore-ranks":
                raise RuntimeError(f"i004 invalid restore evidence directory: {root}")
            _atomic_json(root / f"diagnostic-rank-{rank}.json", diagnostic)
            raise RuntimeError(
                f"LoRA optimizer restore differs on rank {rank}: "
                f"categories={diagnostic['mismatched_categories']} "
                f"{restored['sha256']} != {expected_manifest_sha256}"
            )
        manifest_representation_normalized = True
    pre_sync_optimizer_manifest_sha256 = restored["sha256"]
    model_sync = _restore_model_parameters_from_optimizer(suboptimizers)
    model_source = _verify_restored_model_against_native_adapter(
        adapter_dir, model
    )
    optimizer_after_model_sync = _optimizer_manifest(
        model, optimizer, scheduler, retain_raw_snapshots=False
    )
    optimizer_after_model_sync.pop("raw_snapshots", None)
    if optimizer_after_model_sync["sha256"] != pre_sync_optimizer_manifest_sha256:
        raise RuntimeError(
            "i004 model parameter synchronization changed optimizer state"
        )
    restored = optimizer_after_model_sync
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "phase": "fresh_process_same_topology_optimizer_restore",
        "run_id": os.environ["MILES_EXPERIMENT_RUN_ID"],
        "global_rank": rank,
        "optimizer_step": EXPECTED_OPTIMIZER_STEP,
        "optimizer_snapshot_sha256": summary["sha256"],
        "optimizer_manifest_sha256": restored["sha256"],
        "expected_optimizer_manifest_sha256": expected_manifest_sha256,
        "manifest_representation_normalized": manifest_representation_normalized,
        "post_restore_model_parameter_sync": (
            "hdo_aware_master_copy_plus_synchronous_dp_all_gather"
        ),
        "post_restore_model_parameter_sync_summary": model_sync,
        "post_restore_model_source_summary": model_source,
        "semantic_category_sha256": {
            name: value["restored_sha256"]
            for name, value in (diagnostic or {"categories": {}})["categories"].items()
        },
        "named_owned_shards": restored["content"]["owned_shard_count"],
        "same_physical_rank_exact": True,
        "cross_topology_portability_claimed": False,
    }
    evidence_dir = os.environ.get("MILES_I004_RESTORE_EVIDENCE_DIR")
    if evidence_dir:
        root = Path(evidence_dir)
        if root.name != "optimizer-restore-ranks":
            raise RuntimeError(f"i004 invalid restore evidence directory: {root}")
        _atomic_json(root / f"rank-{rank}.json", receipt)
    return receipt
