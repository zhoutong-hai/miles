"""Engine-side RPCs for a weight-update session.

The session frame is: pause -> begin -> (transfer) -> set version -> end ->
resume. Callers gate driver-only calls (typically global rank 0) themselves.
"""

import re
from argparse import Namespace
from collections.abc import Mapping, Sequence

import ray
from ray.actor import ActorHandle

_LORA_CHECKSUM_ACK_PATTERN = re.compile(
    r"\[LORA-CHECK\]\s+rank=(\d+)\s+(PASS|FAIL)\b"
)


def pause_engines(args: Namespace, rollout_engines: Sequence[ActorHandle]) -> None:
    """Quiesce the engines for a weight write.

    in_place pausing freezes requests and resumes them against their existing
    KV cache, so flushing would discard exactly what that mode preserves.
    """
    mode = args.pause_generation_mode
    ray.get([engine.pause_generation.remote(mode=mode) for engine in rollout_engines])
    if mode != "in_place":
        ray.get([engine.flush_cache.remote() for engine in rollout_engines])


def resume_engines(rollout_engines: Sequence[ActorHandle]) -> None:
    ray.get([engine.continue_generation.remote() for engine in rollout_engines])


def begin_weight_update(
    rollout_engines: Sequence[ActorHandle], selector: str = "all", *, sync_base: bool = True
) -> None:
    """Open a weight-update session on the selected engines. ``sync_base=False``
    declares an adapter-only session: no quant unpack, base tensors rejected."""
    ray.get([engine.begin_weight_update.remote(selector=selector, sync_base=sync_base) for engine in rollout_engines])


def end_weight_update(
    rollout_engines: Sequence[ActorHandle],
    *,
    expected_lora_checksums: Mapping | None = None,
    expected_tp_ranks: int | None = None,
) -> None:
    """Close the session: re-finalize base weights (sync_base sessions) and apply
    the streamed LoRA stash (optionally verified against a sha256 manifest)."""
    if expected_lora_checksums is not None and (
        type(expected_tp_ranks) is not int or expected_tp_ranks <= 0
    ):
        raise ValueError(
            "expected_tp_ranks must be a positive integer when validating LoRA checksums"
        )
    if expected_lora_checksums is not None:
        assert expected_tp_ranks is not None
    results = ray.get(
        [
            engine.end_weight_update.remote(expected_lora_checksums=expected_lora_checksums)
            for engine in rollout_engines
        ]
    )
    if len(results) != len(rollout_engines):
        raise RuntimeError(
            "end_weight_update did not return exactly one result per rollout engine"
        )
    checksum_ack_engines = 0
    for engine_index, result in enumerate(results):
        if isinstance(result, Mapping) and result.get("success") is False:
            raise RuntimeError(f"end_weight_update failed on a rollout engine: {result.get('message')}")
        if expected_lora_checksums is not None:
            if not isinstance(result, Mapping) or result.get("success") is not True:
                raise RuntimeError(
                    f"end_weight_update engine {engine_index} omitted a successful checksum result"
                )
            message = result.get("message", "")
            checksum_acks = _LORA_CHECKSUM_ACK_PATTERN.findall(message)
            if message.count("[LORA-CHECK]") != len(checksum_acks):
                raise RuntimeError(
                    f"end_weight_update engine {engine_index} returned malformed rank ACKs"
                )
            observed_ranks = [int(rank) for rank, _ in checksum_acks]
            expected_ranks = list(range(expected_tp_ranks))
            if sorted(observed_ranks) != expected_ranks:
                raise RuntimeError(
                    f"end_weight_update engine {engine_index} TP-rank ACK coverage mismatch: "
                    f"expected={expected_ranks} observed={sorted(observed_ranks)}"
                )
            failed_ranks = [
                int(rank) for rank, status in checksum_acks if status != "PASS"
            ]
            if failed_ranks:
                raise RuntimeError(
                    f"end_weight_update engine {engine_index} returned failed checksum ACKs "
                    f"from ranks {failed_ranks}"
                )
            checksum_ack_engines += 1
            print(
                f"[E006-LORA-END-ACK] engine={engine_index} ranks={expected_ranks} {message}",
                flush=True,
            )
    if expected_lora_checksums is not None and checksum_ack_engines != len(
        rollout_engines
    ):
        raise RuntimeError(
            "end_weight_update did not receive complete LoRA checksum ACKs from every engine"
        )


def register_lora_adapter(
    rollout_engines: Sequence[ActorHandle], *, lora_name: str, lora_config: Mapping, pinned: bool = False
) -> None:
    """Create-or-refresh an adapter's identity and config on every engine
    (weights zeroed; the bytes follow in the update stream)."""
    refs = [
        engine.register_lora_adapter.remote(lora_name=lora_name, config_dict=dict(lora_config), pinned=pinned)
        for engine in rollout_engines
    ]
    check_weight_sync_results(ray.get(refs), is_lora=True)


def set_weight_version(rollout_engines: Sequence[ActorHandle], weight_version: int) -> None:
    ray.get([engine.update_weight_version.remote(weight_version=str(weight_version)) for engine in rollout_engines])


def check_weight_sync_results(results: list, *, is_lora: bool) -> None:
    """Raise if any engine reported a failed weight-sync RPC."""
    sync_type = "LoRA" if is_lora else "Base model"
    for result in results:
        if isinstance(result, Mapping):
            success = result.get("success")
            error_msg = result.get("error_message") or result.get("error") or "unknown error"
        elif hasattr(result, "success"):
            success = result.success
            error_msg = getattr(result, "error_message", "unknown error")
        else:
            continue

        if success is False:
            raise RuntimeError(
                f"{sync_type} weight sync failed on rollout engine: {error_msg}. "
                f"Check SGLang version compatibility."
            )
