import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle
from sglang.srt.utils import MultiprocessingSerializer

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.utils.lora import lora_base_cpu_backup_enabled, lora_rollout_base_retained, lora_rollout_enabled

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

from .broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


# One stable exporter allocation per process/device/dtype.  The receiver's
# update RPC is a per-bucket copy-completion acknowledgement, so the same bytes
# may be refreshed only after send_bucket's ray.get returns.
_IPC_TRANSFER_BUFFERS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _copy_into_persistent_transfer_buffer(
    flattened: torch.Tensor,
    *,
    minimum_bytes: int,
) -> torch.Tensor:
    """Return a view containing ``flattened`` in stable CUDA-IPC storage."""
    key = (flattened.device, flattened.dtype)
    minimum_numel = (minimum_bytes + flattened.element_size() - 1) // flattened.element_size()
    required_numel = max(flattened.numel(), minimum_numel)
    transfer = _IPC_TRANSFER_BUFFERS.get(key)
    if transfer is None:
        transfer = torch.empty(
            required_numel,
            dtype=flattened.dtype,
            device=flattened.device,
        )
        _IPC_TRANSFER_BUFFERS[key] = transfer
    if transfer.numel() < flattened.numel():
        raise RuntimeError(
            "weight-update bucket exceeds the persistent CUDA-IPC transfer buffer; "
            f"buffer_numel={transfer.numel()} bucket_numel={flattened.numel()}"
        )
    view = transfer[: flattened.numel()]
    view.copy_(flattened)
    if view.is_cuda:
        # The first CUDA-IPC share owns the automatic event.  Reuse therefore
        # needs an explicit producer-stream completion before serialization.
        torch.cuda.current_stream(device=view.device).synchronize()
    return view


class UpdateWeightFromTensor(WeightTransferProtocol):
    """
    Colocated transfer: every training rank serializes its bucket to CUDA-IPC
    handles and gather_objects them to its engine group's src rank, which hands
    them to the engine over Ray. Hybrid deployments broadcast to the
    non-colocated engine tail over NCCL from the global source rank.
    """

    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        """
        Create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        super().__init__(args)
        self._model_update_groups = None
        # Overwritten with "miles" when connect finds a distributed engine tail.
        self.group_name = "miles-colocate"

        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = min(start_rank + self.args.rollout_num_gpus_per_engine, dist.get_world_size())
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self._selector = selector

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums
        if self.use_distribute and lora_rollout_enabled(self.args):
            raise NotImplementedError("LoRA weight sync is not supported for hybrid colocated+distributed deployments")

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0 and parallel_state.pp.rank == 0
            )
            self.group_name = "miles"
            if self._is_distributed_src_rank:
                if (g := self._model_update_groups) is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self.group_name, g, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self.group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Determine whether this rank is covered by any colocated engine.
        all_colocated_ranks = set()
        for offset, count in zip(colocate_gpu_offsets, colocate_gpu_counts, strict=True):
            all_colocated_ranks.update(range(offset, offset + count))
        rank_has_engine = dist.get_rank() in all_colocated_ranks

        # Create IPC Gloo gather groups matching actual engine layout.
        # Re-create on first call or when engine layout changes (placeholder ranks
        # that had a group from __init__ but no actual engine need to be reset).
        if rank_has_engine:
            if self._ipc_gather_group is None:
                for i in range(colocate_engine_nums):
                    group_ranks = list(
                        range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i])
                    )
                    new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                    if dist.get_rank() in group_ranks:
                        self._ipc_gather_group = new_group
                        self._ipc_gather_src = colocate_gpu_offsets[i]
        else:
            # Ranks not covered by any engine (e.g. placeholder GPU slots)
            self._ipc_gather_group = None
            self._ipc_gather_src = None

        # Map training ranks to colocated engine actors.
        self._ipc_engine = None
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # Every engine-covered rank sends: the per-engine gather_object is
        # collective among the group's members.
        self.is_sender = self._ipc_gather_group is not None

        # A LoRA sync must re-push the frozen base unless the engines keep it
        # across pauses (CPU backup, persistent GPU copy, or remote engines);
        # the weight checker always needs the full rewrite.
        base_persists = (
            self.use_distribute
            or lora_base_cpu_backup_enabled(self.args)
            or (self.args.colocate and lora_rollout_base_retained(self.args))
        )
        self.needs_base_resync_for_lora = self.args.check_weight_update_equal or not base_persists

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        refs, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors=bucket,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            selector=self._selector,
            persistent_transfer_bytes=getattr(self.args, "update_weight_buffer_size", 0),
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.distributed_rollout_engines,
                bucket,
                selector=self._selector,
            )
            if refs_distributed:
                refs = (refs or []) + refs_distributed
        # This is the transfer-buffer reuse boundary: every receiver has copied
        # the bucket into memory it owns before its RPC can acknowledge here.
        # Only the gather source owns the Ray ref, but every rank contributed a
        # process-local persistent CUDA-IPC buffer.  Broadcast the ACK before
        # any contributor can return and refresh that buffer for the next
        # bucket.  The source also broadcasts failures so peers do not hang or
        # race ahead after a rejected transfer.
        _wait_for_colocated_transfer_ack(
            refs=refs,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
        )
        del long_lived_tensors


def _wait_for_colocated_transfer_ack(
    *,
    refs: list[ObjectRef] | None,
    ipc_gather_src: int,
    ipc_gather_group,
) -> None:
    """Fence group-wide CUDA-IPC buffer reuse on the receiver's copy ACK."""
    is_gather_src = dist.get_rank() == ipc_gather_src
    status = [None]
    if is_gather_src:
        try:
            check_weight_sync_results(ray.get(refs or []), is_lora=False)
            status[0] = {"success": True}
        except Exception as exc:
            status[0] = {
                "success": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    dist.broadcast_object_list(
        status,
        src=ipc_gather_src,
        group=ipc_gather_group,
    )
    result = status[0]
    if not isinstance(result, dict) or type(result.get("success")) is not bool:
        raise RuntimeError(
            "colocated weight-sync source broadcast a malformed transfer ACK"
        )
    if not result["success"]:
        raise RuntimeError(
            "colocated weight-sync receiver rejected the transfer: "
            f"{result.get('error_type', 'Error')}: {result.get('message', '')}"
        )


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    selector: str = "all",
    persistent_transfer_bytes: int = 0,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    is_gather_src = dist.get_rank() == ipc_gather_src
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors: list = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor_data = {
            "flattened_tensor": _copy_into_persistent_transfer_buffer(
                flattened_tensor_bucket.get_flattened_tensor(),
                minimum_bytes=persistent_transfer_bytes,
            ),
            "metadata": flattened_tensor_bucket.get_metadata(),
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = [None] * dist.get_world_size(ipc_gather_group) if is_gather_src else None
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    refs = []
    if is_gather_src:
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "selector": selector,
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    return refs, long_live_tensors
