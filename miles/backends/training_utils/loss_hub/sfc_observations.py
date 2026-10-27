"""Private aligned pre-update numeric records; one writer per final-stage CP shard."""

import hashlib
import io
import json
import os
from pathlib import Path
import struct
import time

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.sfc_logit_audit import component_gradients, component_statistics
from miles.backends.training_utils.parallel import get_parallel_state

_CONTEXT = None


def enabled(args):
    return getattr(args, "loss_type", None) == "score_centering" and bool(getattr(args, "sfc_diagnostics_dir", None))


def configure_rollout(args, rollout_id, trainer_version, data):
    """The trainer lifecycle owns this observational context, never the sampler."""
    global _CONTEXT
    if not enabled(args):
        return
    root = Path(args.sfc_diagnostics_dir)
    if not root.is_absolute() or root != root.resolve() or trainer_version is None:
        raise ValueError("SFC observations need an ordinary absolute output and actual trainer weight version")
    indices = data.get("sample_indices")
    if indices is None or len(set(indices)) != len(indices):
        raise ValueError("SFC observations require unique original sample identities")
    metadata = data["sfc_sample_metadata"]
    live = [(int(index), int(position)) for index, sample in zip(indices, metadata, strict=True)
            for position in sample["first_valid_positions"]]
    data["sfc_sample_metadata"] = [{**sample, "advantage": float(reward)}
                                   for sample, reward in zip(metadata, data["rewards"], strict=True)]
    _CONTEXT = {"root": root, "rollout": int(rollout_id), "completed": int(rollout_id)+1,
                "trainer_version": str(trainer_version), "expected_tokens": sum(sample["valid_tokens"] for sample in metadata),
                "expected_samples": set(map(int, indices)), "audit_positions": set(sorted(live)[:32]),
                "source": os.environ.get("MILES_SOURCE_COMMIT", "UNBOUND"),
                "run": getattr(args, "wandb_run_id", None), "args": args,
                "seen": set(), "frames": 0, "tokens": 0, "capture_seconds": 0.0, "audit_seconds": 0.0}
    if _CONTEXT["source"] == "UNBOUND":
        raise ValueError("SFC numeric sidecars require the effective source commit binding")
    parallel = get_parallel_state()
    if parallel.effective_dp.size != 1:
        raise ValueError("Selected GLM observations currently bind dense DP1")
    directory = root / "paired_scores" / f"update-{rollout_id+1:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    _CONTEXT["directory"] = directory
    _CONTEXT["file"] = directory / f"cp-{parallel.cp.rank}.bin"
    if parallel.is_pp_last_stage and parallel.tp.rank == 0 and _CONTEXT["file"].exists():
        raise ValueError("Existing numeric shard; use a new immutable attempt or reconcile exact resume")


def _writer():
    parallel = get_parallel_state()
    return _CONTEXT is not None and parallel.is_pp_last_stage and parallel.tp.rank == 0


def _response_layout(args, batch):
    positions, tokens, sample_ids, sample_slots = [], [], [], []
    for slot, (total, response, index) in enumerate(zip(batch["total_lengths"], batch["response_lengths"], batch["sample_indices"], strict=True)):
        maximum = batch["max_seq_lens"][slot] if batch.get("max_seq_lens") else None
        positions_i = slice_log_prob_with_cp(torch.arange(response), total, response, args.qkv_format, maximum)
        response_tokens = batch["unconcat_tokens"][slot][-response:] if response else batch["unconcat_tokens"][slot][:0]
        tokens_i = slice_log_prob_with_cp(response_tokens, total, response, args.qkv_format, maximum)
        positions.append(positions_i.cpu())
        tokens.append(tokens_i.detach().cpu())
        sample_ids.append(torch.full((len(positions_i),), int(index), dtype=torch.long))
        sample_slots.append(torch.full((len(positions_i),), slot, dtype=torch.long))
    return {"token_position": torch.cat(positions), "token_id": torch.cat(tokens),
            "sample_id": torch.cat(sample_ids), "sample_slot": torch.cat(sample_slots)}


def capture(args, batch, selected, rollout, advantages, active, ids, head, metrics, entropy, token_loss, sampler_kl):
    if not enabled(args) or not _writer():
        return
    start = time.perf_counter()
    context = _CONTEXT
    layout = _response_layout(args, batch)
    key = tuple(map(int, batch["sample_indices"]))
    if key in context["seen"]:
        return  # Loss checkpoint recomputation is the same pre-update observation.
    context["seen"].add(key)
    logp, logq = selected[:, 0].detach(), rollout.detach()
    if not torch.isfinite(logp[active]).all() or not torch.isfinite(logq[active]).all():
        raise ValueError("Nonfinite active sampler/trainer scores in paired diagnostics")
    if not torch.isfinite(advantages[active]).all() or not torch.isfinite((logp-logq)[active]).all():
        raise ValueError("Nonfinite active advantages or raw log ratios")
    records = {**layout, "mask": active.detach().cpu(), "logp": logp.cpu(), "logq": logq.cpu(),
               "advantage": advantages.detach().cpu(),
               "weight": ((logp-logq).clamp_max(0.6931471805599453).exp() if args.score_centering_is == "tis"
                          else torch.ones_like(logp)).cpu(),
               "entropy": entropy.detach().cpu(), "pg_loss": token_loss.detach().cpu(),
               "sampler_kl": sampler_kl.detach().cpu(), "correction": metrics.correction.cpu(),
               "p_head_mass": selected[:, 1:].detach().exp().masked_fill(ids < 0, 0).sum(-1).cpu(),
               "q_head_mass": head.detach().exp().masked_fill(ids < 0, 0).sum(-1).cpu(),
               "candidate_count": (ids >= 0).sum(-1).cpu(),
               "sample_in_head": ((ids.cpu() == layout["token_id"][:, None]) & (ids.cpu() >= 0)).any(-1)}
    metadata = batch.get("sfc_sample_metadata")
    if metadata is None or batch.get("weight_versions") is None:
        raise ValueError("SFC observations require reward/theme/group and actual sampler-version metadata")
    expected_advantage = torch.tensor([m["advantage"] for m in metadata])[layout["sample_slot"]]
    torch.testing.assert_close(records["advantage"][records["mask"]],
                               expected_advantage[records["mask"]], rtol=1e-5, atol=1e-6)
    candidate_hash = hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes() + head.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    packet = {"schema_version": 1, "run": context["run"], "source": context["source"],
              "rollout": context["rollout"], "native_pre_update": context["rollout"],
              "completed_update": context["completed"], "trainer_version": context["trainer_version"],
              "sample_indices": list(map(int, batch["sample_indices"])), "samples": metadata,
              "sampler_versions": batch["weight_versions"], "candidate_payload_sha256": candidate_hash,
              "rollout_packet": f"rollout_data/{context['rollout']}.pt", "data": records}
    buffer = io.BytesIO()
    torch.save(packet, buffer)
    payload = buffer.getvalue()
    with context["file"].open("ab") as stream:
        stream.write(struct.pack("<Q", len(payload)))
        stream.write(payload)
    context["frames"] += 1
    context["tokens"] += int(active.sum())
    context["capture_seconds"] += time.perf_counter() - start


def logit_audit(args, batch, sample, chunk, tokens, indices):
    context = _CONTEXT
    if not enabled(args) or context is None or context["completed"] not in (8, 32, 64):
        return
    index = int(batch["sample_indices"][sample])
    chosen = [i for i, position in enumerate(indices) if (index, int(position)) in context["audit_positions"]]
    if not chosen:
        return
    start = time.perf_counter()
    positions = torch.as_tensor(list(indices), dtype=torch.long)[chosen]
    keys = tuple((index, int(position)) for position in positions)
    # Every TP member must enter the same audit collectives even on recompute;
    # the writer below prevents duplicate persisted records.
    head_ids = torch.as_tensor(batch["rollout_topk_token_ids"][sample])[positions].to(chunk.device).long()
    head_logq = torch.as_tensor(batch["rollout_topk_log_probs"][sample])[positions].to(chunk.device)
    q = torch.as_tensor(batch["sfc_generation_log_probs"][sample])[positions].to(chunk.device)
    full_advantage = batch["sfc_sample_metadata"][sample]["advantage"]
    advantage = torch.full((len(chosen),), full_advantage, device=chunk.device)
    parallel = get_parallel_state()
    group = parallel.tp.group if parallel.tp.size > 1 else None
    components = component_gradients(chunk[chosen], head_ids, head_logq, tokens[chosen], q, advantage,
                                    mode=args.score_centering_is, beta=args.sampler_kl_coef,
                                    vocab_size=args.vocab_size, group=group, denominator=context["expected_tokens"])
    # chunk has already been temperature-scaled by the retained response iterator.
    components = {k: value / args.rollout_temperature for k, value in components.items()}
    statistics = component_statistics(components, group=group)
    if parallel.tp.rank == 0:
        destination = context["root"] / "logit_audit" / f"update-{context['completed']:06d}"
        destination.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(repr(keys).encode()).hexdigest()[:20]
        path = destination / f"cp-{parallel.cp.rank}-{digest}.pt"
        if not path.exists():
            torch.save({"positions": keys, "source": context["source"], "trainer_version": context["trainer_version"],
                        "completed_update": context["completed"], **statistics}, path)
            context["audit_seconds"] += time.perf_counter()-start


def finish(args, outcome):
    if not enabled(args) or not _writer():
        return {}
    context = _CONTEXT
    if not context["file"].exists():
        context["file"].touch()
    with context["file"].open("ab") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    seal = {"source": context["source"], "trainer_version": context["trainer_version"],
            "frames": context["frames"], "valid_tokens": context["tokens"], "outcome": str(outcome),
            "capture_seconds": context["capture_seconds"], "audit_seconds": context["audit_seconds"],
            "sha256": hashlib.sha256(context["file"].read_bytes()).hexdigest()}
    seal_path = context["file"].with_suffix(".complete.json")
    with seal_path.open("x") as stream:
        json.dump(seal, stream, sort_keys=True)
    parallel = get_parallel_state()
    if parallel.cp.size > 1:
        dist.barrier(group=parallel.cp.group)
    if parallel.cp.rank != 0:
        return {}
    # Import here to keep the aggregation-only CPU dependencies out of the loss
    # import path and avoid a cycle with the frame reader.
    from miles.backends.training_utils.loss_hub.sfc_summary import summarize_update
    return summarize_update(context["directory"], expected_tokens=context["expected_tokens"],
                            expected_samples=context["expected_samples"], cp_size=parallel.cp.size,
                            mode=args.score_centering_is, beta=args.sampler_kl_coef)
