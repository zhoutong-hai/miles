"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

import json
import logging
import math
import os
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)
from miles.utils.lora.utils import is_lora_weight_name

logger = logging.getLogger(__name__)


class MegatronHfWeightIteratorBase(HfWeightIteratorBase):
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packed_weight_basenames = (
            get_packed_weight_basenames(self.args.hf_checkpoint)
            if self.quantization_config is not None
            and self.quantization_config["quant_method"] == "compressed-tensors"
            else None
        )

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.backend.q_lora_rank)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        """Both megatron exporters are PP-local after gathering TP/EP; the PP
        gather streams one owner stage at a time, avoiding a full MoE adapter
        on every rank. Non-materializing ranks still join every collective."""
        named_tensors = self._export_pp_local_lora(adapter)
        num_tensors, num_lora_weights = _count_full_adapter_tensors(named_tensors, is_pp_local=True)
        if materialize and num_tensors == 0:
            raise RuntimeError("LoRA weight sync failed: the adapter export produced zero tensors")
        if materialize and num_lora_weights == 0:
            raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B names.")
        stages = (
            _iter_pp_adapter_stages(named_tensors, _gather_pp_adapter_layout(named_tensors))
            if self.placement.gather_pp
            else iter([named_tensors])
        )
        del named_tensors
        for stage in stages:
            while stage:
                hf_name, tensor = stage.pop(0)
                if materialize:
                    yield [(hf_name, tensor)]
                del tensor

    @abstractmethod
    def _export_pp_local_lora(self, adapter) -> list[tuple[str, torch.Tensor]]:
        """Backend hook: the adapter's HF-named tensors, TP/EP gathered, PP-local."""


def get_hf_weight_iterator(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    cls = {
        "raw": HfWeightIteratorDirect,
        "bridge": HfWeightIteratorBridge,
    }[args.megatron_to_hf_mode]

    return cls.build(
        args,
        model,
        required_placement=required_placement,
        model_name=model_name,
        quantization_config=quantization_config,
    )


def get_packed_weight_basenames(hf_checkpoint: str) -> set[str]:
    """Base names the checkpoint stores as compressed-tensors `weight_packed`; the quantizer
    re-quantizes exactly these, since the published `ignore` list is written for loaders and
    leaves out BF16 weights such as routers, residual projections and the vision tower."""
    with open(os.path.join(hf_checkpoint, "model.safetensors.index.json")) as index_file:
        names = json.load(index_file)["weight_map"]
    return {n.removesuffix(".weight_packed") for n in names if n.endswith(".weight_packed")}


def _count_full_adapter_tensors(
    named_tensors: Sequence[tuple[str, torch.Tensor]], *, is_pp_local: bool
) -> tuple[int, int]:
    """(tensors, lora_A/lora_B weights) in the whole adapter; a PP-local export is
    summed over PP since a stage may hold no adapted layer."""
    counts = [len(named_tensors), sum(is_lora_weight_name(name) for name, _ in named_tensors)]
    pp = get_parallel_state().pp
    if is_pp_local and pp.size > 1:
        counts_tensor = torch.tensor(counts, device=torch.cuda.current_device())
        dist.all_reduce(counts_tensor, group=pp.group)
        counts = counts_tensor.tolist()
    return counts[0], counts[1]


def _gather_pp_adapter_layout(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[list[tuple[str, tuple[int, ...], torch.dtype]]]:
    """Exchange only adapter names, shapes and dtypes across pipeline stages."""
    pp = get_parallel_state().pp
    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    if pp.size == 1:
        return [local_meta]
    all_meta: list = [None] * pp.size
    dist.all_gather_object(all_meta, local_meta, group=pp.group)
    return all_meta


def _iter_pp_adapter_stages(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
    layout: list[list[tuple[str, tuple[int, ...], torch.dtype]]],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Stream a PP owner's adapter before allocating the next owner's buffers.

    Adapted from upstream PR #4127, retaining the current PP-local count reduction.
    """
    pp = get_parallel_state().pp
    if pp.size == 1:
        yield hf_named_tensors if isinstance(hf_named_tensors, list) else list(hf_named_tensors)
        return
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()

    local_by_name = {n: t for n, t in hf_named_tensors}
    for src, meta in enumerate(layout):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        stage = []
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src == pp.rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src], group=pp.group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                stage.append((n, flat[off : off + k].view(shape)))
                off += k
            del flat
        stage.sort(key=lambda item: item[0])
        yield stage
        del stage


_FROZEN_TOWER_PROVIDERS = ("inkling_mm_model_provider", "qwen3_5_mm_model_provider")


def _iter_mm_tower_units(args, *, materialize):
    # The trainer never owns these frozen towers, so they come from the boot checkpoint on every sync, like any
    # weight an engine drops across offload.
    provider = args.custom_model_provider_path
    if provider is None or not any(name in provider.path for name in _FROZEN_TOWER_PROVIDERS):
        return
    yield from iter_checkpoint_tower_units(args.hf_checkpoint, materialize=materialize)
