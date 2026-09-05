"""Engine-side RPCs for a weight-update session.

The session frame is: pause -> begin -> (transfer) -> set version -> end ->
resume. Callers gate driver-only calls (typically global rank 0) themselves.
"""

from argparse import Namespace
from collections.abc import Mapping, Sequence

import ray
from ray.actor import ActorHandle


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
    rollout_engines: Sequence[ActorHandle], *, expected_lora_checksums: Mapping | None = None
) -> None:
    """Close the session: re-finalize base weights (sync_base sessions) and apply
    the streamed LoRA stash (optionally verified against a sha256 manifest)."""
    results = ray.get(
        [
            engine.end_weight_update.remote(expected_lora_checksums=expected_lora_checksums)
            for engine in rollout_engines
        ]
    )
    checksum_ack_engines = 0
    for engine_index, result in enumerate(results):
        if isinstance(result, Mapping) and result.get("success") is False:
            raise RuntimeError(f"end_weight_update failed on a rollout engine: {result.get('message')}")
        if expected_lora_checksums is not None and isinstance(result, Mapping):
            message = result.get("message", "")
            rank_tokens = {
                int(token.split("=", 1)[1])
                for token in message.split()
                if token.startswith("rank=") and token.split("=", 1)[1].isdigit()
            }
            if not rank_tokens or "[LORA-CHECK]" not in message or "PASS" not in message:
                raise RuntimeError(
                    f"end_weight_update engine {engine_index} omitted rank-addressed LoRA checksum ACKs"
                )
            if message.count("[LORA-CHECK]") != len(rank_tokens):
                raise RuntimeError(
                    f"end_weight_update engine {engine_index} returned duplicate or malformed rank ACKs"
                )
            checksum_ack_engines += 1
            print(
                f"[E006-LORA-END-ACK] engine={engine_index} ranks={sorted(rank_tokens)} {message}",
                flush=True,
            )
    if expected_lora_checksums is not None and checksum_ack_engines == 0:
        raise RuntimeError("end_weight_update received no rank-addressed LoRA checksum ACKs")


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
