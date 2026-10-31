"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

from miles.backends.training_utils.k3_publication.publication_paths import publication_root_matches

import json
import logging
import math
import os
import time
from pathlib import Path
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core.utils import unwrap_model

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
    resolve_placement,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.utils.lora import is_lora_weight_name

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
        trainer_has_mtp = bool(unwrap_model(self.model)[0].config.mtp_num_layers)
        if self.args.sglang_speculative_algorithm and not trainer_has_mtp:
            self.weight_update_selector = "target"

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.q_lora_rank)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        """Both megatron exporters are PP-local after gathering TP/EP; the PP
        gather runs only where the resolved placement asks for it."""
        profile = {"export_s": 0.0, "pp_metadata_s": 0.0,
                   "pp_allocate_s": 0.0, "pp_broadcast_s": 0.0,
                   "exported_tensors": 0, "streamed_tensors": 0}
        t0 = time.perf_counter()
        named_tensors = self._export_pp_local_lora(adapter)
        profile["export_s"] = time.perf_counter() - t0
        profile["exported_tensors"] = len(named_tensors)
        self._e007_pub_calls = getattr(self, "_e007_pub_calls", 0) + 1
        # All PP ranks must join the same collectives even when this rank does
        # not materialize transport buckets.
        if self.placement.gather_pp:
            units = _gather_pp_full_adapter(named_tensors, profile=profile)
        else:
            if materialize:
                if not named_tensors:
                    raise RuntimeError("LoRA weight sync failed: the adapter export produced zero tensors")
                if not any(is_lora_weight_name(name) for name, _tensor in named_tensors):
                    raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B names.")
            units = iter(sorted(named_tensors, key=lambda item: item[0]))
        for hf_name, tensor in units:
            profile["streamed_tensors"] += 1
            if materialize:
                yield [(hf_name, tensor)]
        rank = dist.get_rank()
        logger.info("E007_PP_EXPORT_PROFILE call=%d rank=%d phases=%s",
                    self._e007_pub_calls, rank, json.dumps(profile, sort_keys=True))
        profile_root = os.environ.get("E007_OPT_DIAG_ROOT")
        if profile_root and publication_root_matches(profile_root):
            target = Path(profile_root) / f"pp-export-call{self._e007_pub_calls}-rank{rank}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({
                "call": self._e007_pub_calls, "rank": rank, "phase_seconds": profile,
            }, sort_keys=True))
            temporary.replace(target)

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

    return cls(
        args,
        model,
        placement=resolve_placement(required_placement, cls.forced_placement),
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


def _gather_pp_full_adapter(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    profile: dict | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream one HF adapter tensor at a time in globally sorted key order.

    Every PP rank joins the same ordered broadcasts. The consumer may send a
    completed bucket before the next tensor is allocated, rather than keeping
    all PP stages' flat buffers live through the entire publication.
    """
    pp = get_parallel_state().pp
    if pp.size == 1:
        ordered = sorted(hf_named_tensors, key=lambda item: item[0])
        if not ordered or not any(is_lora_weight_name(name) for name, _ in ordered):
            raise RuntimeError("LoRA weight sync failed: missing adapter tensors")
        yield from ordered
        return
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp.size
    t0 = time.perf_counter()
    dist.all_gather_object(all_meta, local_meta, group=pp.group)
    if profile is not None:
        profile["pp_metadata_s"] += time.perf_counter() - t0
    ordered = sorted((
        (name, shape, dtype, src)
        for src, meta in enumerate(all_meta)
        for name, shape, dtype in meta
    ), key=lambda item: item[0])
    names = [name for name, _shape, _dtype, _src in ordered]
    if not names or len(names) != len(set(names)) or not any(is_lora_weight_name(name) for name in names):
        raise RuntimeError("LoRA weight sync failed: missing or duplicate adapter tensors")

    local_by_name = {name: tensor for name, tensor in hf_named_tensors}
    for name, shape, dtype, src in ordered:
        t0 = time.perf_counter()
        if src == pp.rank:
            tensor = local_by_name[name].contiguous()
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device)
        if profile is not None:
            profile["pp_allocate_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        dist.broadcast(tensor, src=global_ranks[src], group=pp.group)
        if profile is not None:
            profile["pp_broadcast_s"] += time.perf_counter() - t0
        yield name, tensor


_MM_TOWER_CACHE: list[tuple[str, "torch.Tensor"]] | None = None


def _iter_mm_tower_units(args, *, materialize):
    """Transitional: Inkling MM trains only the language model; the frozen
    vision/audio towers are deliberately unregistered trainer-side (ckpt
    compat) and exist only on pre_process ranks, so the boot checkpoint is the
    uniform source every rank can re-send from (loads are idempotent).
    Goes away when the towers become real megatron params (Kimi-style) or the
    engine keeps them across offload."""
    global _MM_TOWER_CACHE
    if "inkling_mm_model_provider" not in (args.custom_model_provider_path or ""):
        return
    if not materialize:
        return
    if _MM_TOWER_CACHE is None:
        import json
        import os

        from safetensors import safe_open

        ckpt_dir = args.hf_checkpoint
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        tower_keys = sorted(
            k
            for k in weight_map
            if ".visual." in f".{k}" or ".audio." in f".{k}" or k.startswith(("visual.", "audio."))
        )
        by_shard: dict[str, list[str]] = {}
        for k in tower_keys:
            by_shard.setdefault(weight_map[k], []).append(k)
        cache = []
        for shard, keys in by_shard.items():
            with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
                for k in keys:
                    cache.append((k, f.get_tensor(k)))
        logger.info("mm tower sync: caching %d tower tensors from %s", len(cache), ckpt_dir)
        _MM_TOWER_CACHE = cache
    for name, tensor in _MM_TOWER_CACHE:
        yield [(name, tensor.to(torch.cuda.current_device()))]
