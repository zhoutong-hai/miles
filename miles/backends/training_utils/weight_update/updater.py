"""Backend-neutral weight-update driver.

The updater owns the lifecycle: it builds the HF weight iterator against the
protocol's required placement, runs the engine session frame, streams base
buckets (senders transmit, other ranks join the gathers), and orchestrates
LoRA adapter pushes.
"""

from miles.backends.training_utils.k3_publication.publication_paths import publication_root_matches

from miles.backends.training_utils.k3_publication.publication import publication_mode
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.conn_status import ConnStatusManager
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.session import (
    begin_weight_update,
    end_weight_update,
    pause_engines,
    register_lora_adapter,
    resume_engines,
    set_weight_version,
)
from miles.backends.training_utils.weight_update.utils import record_lora_checksums
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME
from miles.utils.timer import timer

logger = logging.getLogger(__name__)


class WeightUpdater:
    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        *,
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        model_name: str,
        quantization_config: dict | None,
        iterator_factory: Callable,
        parallel_state: ParallelState,
        is_lora: bool,
        lora_sync_config: dict | None = None,
    ) -> None:
        self.args = args
        self.parallel_state = parallel_state
        self.protocol = get_weight_transfer_protocol(args)
        self.conn_status = ConnStatusManager()
        assert (
            not is_lora or self.protocol.supports_lora
        ), f"LoRA weight sync is not supported for {args.update_weight_transfer_mode!r} weight transfer."
        self._hf_weight_iterator = iterator_factory(
            args,
            model,
            required_placement=self.protocol.required_placement,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.is_lora = is_lora
        if is_lora:
            assert lora_sync_config is not None
        self._lora_sync_config = lora_sync_config
        self._registered_adapters: set[str] = set()

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.protocol.connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            self.parallel_state,
            self._hf_weight_iterator.placement,
            self._hf_weight_iterator.weight_update_selector,
        )
        assert self.protocol.is_sender is not None, "connect() must set is_sender"
        self._registered_adapters.clear()

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the protocol's metrics; the actor drains them onto the step log."""
        return self.protocol.pop_metrics()

    @torch.no_grad()
    def update_weights(self) -> None:
        """Run one weight sync: session frame + base-bucket stream + adapter pushes for LoRA."""
        protocol = self.protocol
        if not protocol.begin_sync(self.weight_version + 1, self._iter_base_buckets):
            return
        self.weight_version += 1
        e007_hash_mode = publication_mode(self.weight_version)

        e007_phase = {
            "session_prepare_s": 0.0, "pre_stream_barrier_s": 0.0,
            "weights_getter_s": 0.0, "iterator_next_s": 0.0,
            "sender_checksum_s": 0.0, "sender_copy_s": 0.0,
            "sender_wire_materialize_s": 0.0, "sender_sha_s": 0.0,
            "send_receiver_wait_s": 0.0, "after_base_s": 0.0,
            "manifest_s": 0.0, "stream_barrier_s": 0.0,
            "protocol_finalize_s": 0.0, "receiver_install_rpc_s": 0.0,
            "set_version_rpc_s": 0.0, "resume_rpc_s": 0.0,
            "final_barrier_s": 0.0, "buckets": 0,
        }
        sync_base = not self.is_lora or protocol.needs_base_resync_for_lora
        adapters = self._get_updated_adapters()

        driver = dist.get_rank() == 0
        t0 = time.perf_counter()
        if protocol.use_weight_update_session and driver:
            pause_engines(self.args, protocol.rollout_engines)
            self._register_new_lora_adapters(protocol.rollout_engines, adapters)
            begin_weight_update(
                protocol.rollout_engines, self._hf_weight_iterator.weight_update_selector, sync_base=sync_base
            )
        e007_phase["session_prepare_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dist.barrier(group=get_gloo_group())
        e007_phase["pre_stream_barrier_s"] = time.perf_counter() - t0

        checksums = {name: {} for name, _ in adapters} if self.is_lora and self.args.check_lora_weight_equal else None
        if checksums is not None:
            assert (
                self._hf_weight_iterator.placement.gather_pp
            ), "the LoRA checksum manifest is recorded on one rank, which must hold the full adapter"
        with timer("update_weights_implementation"):
            pbar = tqdm(desc=f"[{protocol.group_name}] Update weights", total=0) if protocol.is_sender else None
            t0 = time.perf_counter()
            source_weights = self.weights_getter()
            e007_phase["weights_getter_s"] = time.perf_counter() - t0
            stream = iter(self._hf_weight_iterator.iter_hf_weights(
                source_weights,
                include_base=sync_base,
                adapters=adapters,
                materialize=protocol.is_sender,
            ))
            while True:
                t0 = time.perf_counter()
                try:
                    bucket = next(stream)
                except StopIteration:
                    e007_phase["iterator_next_s"] += time.perf_counter() - t0
                    break
                e007_phase["iterator_next_s"] += time.perf_counter() - t0
                e007_phase["buckets"] += 1
                if protocol.is_sender:
                    if driver and checksums is not None:
                        t0 = time.perf_counter()
                        with timer(f"e007_publication_sender_checksum_v{self.weight_version}"):
                            record_lora_checksums(
                                bucket, checksums, mode=e007_hash_mode, profile=e007_phase
                            )
                        e007_phase["sender_checksum_s"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    with timer("e007_publication_send_and_receiver_wait"):
                        protocol.send_bucket(bucket)
                    e007_phase["send_receiver_wait_s"] += time.perf_counter() - t0
                    pbar.update(1)
            t0 = time.perf_counter()
            protocol.after_base_weights()
            e007_phase["after_base_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            if driver and checksums is not None:
                root = os.environ.get("E007_OPT_DIAG_ROOT")
                if not root or not publication_root_matches(root):
                    raise RuntimeError("publication manifest root missing")
                wire = json.dumps(checksums, sort_keys=True, separators=(",", ":")).encode()
                payload = {"status": "COMPLETE", "version": self.weight_version,
                           "mode": e007_hash_mode, "checksums": checksums,
                           "checksums_sha256": hashlib.sha256(wire).hexdigest()}
                serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                digest = hashlib.sha256(serialized).hexdigest()
                output = Path(root)
                output.mkdir(parents=True, exist_ok=True)
                target = output / f"sender-manifest-v{self.weight_version}.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_bytes(serialized)
                temporary.replace(target)
                logger.info("E007_SENDER_MANIFEST version=%d mode=%s sha256=%s",
                            self.weight_version, e007_hash_mode, digest)
            e007_phase["manifest_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            e007_phase["stream_barrier_s"] = time.perf_counter() - t0

        with timer("finalize_and_resume_engines"):
            t0 = time.perf_counter()
            protocol.finalize(self.weight_version)
            e007_phase["protocol_finalize_s"] = time.perf_counter() - t0
            if protocol.use_weight_update_session and driver:
                t0 = time.perf_counter()
                end_weight_update(protocol.rollout_engines, expected_lora_checksums=checksums)
                e007_phase["receiver_install_rpc_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                set_weight_version(protocol.rollout_engines, self.weight_version)
                e007_phase["set_version_rpc_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                resume_engines(protocol.rollout_engines)
                e007_phase["resume_rpc_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            dist.barrier(group=get_gloo_group())
            e007_phase["final_barrier_s"] = time.perf_counter() - t0
        protocol.after_engines_resumed()
        profile_root = os.environ.get("E007_OPT_DIAG_ROOT")
        if profile_root and publication_root_matches(profile_root):
            rank = dist.get_rank()
            target = Path(profile_root) / f"publication-phase-v{self.weight_version}-rank{rank}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "version": self.weight_version, "rank": rank, "phase_seconds": e007_phase,
            }, sort_keys=True))
            temporary.replace(target)
        logger.info("E007_PUBLICATION_PHASE version=%d rank=%d phases=%s",
                    self.weight_version, dist.get_rank(), json.dumps(e007_phase, sort_keys=True))

    def _iter_base_buckets(self, *, materialize: bool):
        return self._hf_weight_iterator.iter_hf_weights(self.weights_getter(), materialize=materialize)

    def _get_updated_adapters(self) -> list[tuple[str, object]]:
        """``(lora_name, adapter_or_None)`` pairs for this sync; the push set is
        identical on every rank so the iterator's collectives align."""
        if not self.is_lora:
            return []
        return [(LORA_ADAPTER_NAME, None)]

    def _register_new_lora_adapters(self, rollout_engines, adapters: list[tuple[str, object]]) -> None:
        """Register adapters the current engine set has not seen, with their
        per-adapter config; eager so the engine validates rank before any bytes move."""
        for lora_name, adapter in adapters:
            if lora_name in self._registered_adapters:
                continue
            config = self._lora_sync_config
            if adapter is not None:
                config = config | {"r": adapter.rank, "lora_alpha": adapter.alpha}
            register_lora_adapter(rollout_engines, lora_name=lora_name, lora_config=config)
            self._registered_adapters.add(lora_name)
