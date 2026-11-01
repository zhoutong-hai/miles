"""Lifecycle-owned, detached REINFORCE/TIS observations for K3 DP4/CP1."""

import os
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from miles.backends.training_utils.loss_hub import policy_metrics
from miles.backends.training_utils.loss_hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss_hub.policy_audit import component_statistics
from miles.backends.training_utils.parallel import get_parallel_state

_CONTEXT = None


def enabled(args):
    return bool(getattr(args, "policy_diagnostics_dir", None))


def sample_metadata(sample):
    metadata = sample.metadata or {}
    live = [i for i, active in enumerate(sample.loss_mask) if active]
    return {
        "group_index": sample.group_index,
        "theme": metadata.get("theme", "UNBOUND"),
        "scenario": metadata.get("scenario", "UNBOUND"),
        "valid_tokens": len(live),
        "first_valid_positions": live[:32],
        "valid_positions": live,
        "rewards": {k: v for k, v in sample.reward.items() if type(v) in (int, float)}
        if isinstance(sample.reward, dict)
        else {"reward": sample.reward},
    }


def configure(args, rollout_id, trainer_version, data):
    global _CONTEXT
    if not enabled(args) or not get_parallel_state().is_pp_last_stage:
        return
    parallel = get_parallel_state()
    if parallel.cp.size != 1:
        raise ValueError("K3 policy diagnostics currently qualify CP1; DP4 is supported")
    root = Path(args.policy_diagnostics_dir)
    source = os.environ.get("MILES_SOURCE_COMMIT")
    if not root.is_absolute() or root != root.resolve() or not source or trainer_version is None:
        raise ValueError("Diagnostics require an absolute private path and source/trainer bindings")
    samples = {}
    for identity, metadata, advantage, mask, versions in zip(
        data["sample_indices"],
        data["policy_sample_metadata"],
        data["advantages"],
        data["loss_masks"],
        data["weight_versions"],
        strict=True,
    ):
        if int(identity) in samples or not versions:
            raise ValueError("Missing sampler version or repeated original sample")
        active = advantage[mask.to(device=advantage.device).bool()]
        value = float(active[0]) if active.numel() else 0.0
        if active.numel() and not (active == value).all():
            raise ValueError("Selected GRPO diagnostic requires a constant response advantage")
        samples[int(identity)] = {**metadata, "advantage": value, "sampler_versions": versions}
    gathered = [None] * parallel.effective_dp.size
    if parallel.effective_dp.size > 1:
        dist.all_gather_object(gathered, samples, group=parallel.effective_dp.group)
    else:
        gathered = [samples]
    combined = {}
    for part in gathered:
        if set(part) & set(combined):
            raise ValueError("DP sample ownership overlaps")
        combined.update(part)
    if sum(s["valid_tokens"] for s in combined.values()) == 0:
        raise ValueError("No original valid response tokens")
    provenance = {
        name: os.environ[name]
        for name in ("MILES_INPUT_SHA256", "MILES_TOKENIZER_REVISION", "MILES_RESOLVED_CONFIG_SHA256")
    }
    binding = {
        "run": args.wandb_run_id,
        "source": source,
        "trainer_version": str(trainer_version),
        "rollout_id": int(rollout_id),
        "completed_update": int(rollout_id) + 1,
        "dp_size": parallel.effective_dp.size,
        "cp_size": 1,
        "provenance": provenance,
        "score_source": "detached_training_forward",
        "loss": "reinforce",
        "tis": [0.5, 2.0],
    }
    directory = root / "paired_scores" / f"update-{rollout_id + 1:06d}"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"dp-{parallel.effective_dp.rank}.bin"
    if parallel.tp.rank == 0 and (path.exists() or path.with_suffix(".complete.json").exists()):
        raise ValueError("Existing diagnostic shard; use a new immutable attempt")
    _CONTEXT = {
        "binding": binding,
        "path": path,
        "root": root,
        "samples": samples,
        "global_samples": combined,
        "audit_positions": set(policy_metrics.select_positions(combined)),
        "seen": set(),
        "frames": 0,
        "valid_tokens": 0,
        "capture_seconds": 0.0,
        "audit_seconds": 0.0,
    }


@torch.no_grad()
def observe(args, batch, logits, *, logp, logq, ratio, weight, advantage, active, entropy):
    if not enabled(args):
        return
    context, parallel = _CONTEXT, get_parallel_state()
    if context is None:
        raise ValueError("Policy diagnostic context was not configured")
    start = time.perf_counter()
    ids, positions, tokens = [], [], []
    for identity, response, unconcat in zip(
        batch["sample_indices"], batch["response_lengths"], batch["unconcat_tokens"], strict=True
    ):
        ids.extend([int(identity)] * response)
        positions.extend(range(response))
        tokens.extend(unconcat[-response:].tolist() if response else [])
    data = {
        "sample_id": torch.tensor(ids),
        "position": torch.tensor(positions),
        "token_id": torch.tensor(tokens),
        "logp": logp.detach().float().cpu(),
        "logq": logq.detach().float().cpu(),
        "ratio": ratio.detach().cpu(),
        "weight": weight.detach().cpu(),
        "advantage": advantage.detach().cpu(),
        "entropy": entropy.detach().cpu(),
        "mask": active.cpu(),
    }
    if any(len(value) != len(logp) for value in data.values()):
        raise ValueError("Misaligned paired policy observations")
    if batch.get("policy_generation_log_probs") is not None:
        original_q = torch.cat(
            [torch.as_tensor(value, dtype=torch.float32) for value in batch["policy_generation_log_probs"]]
        )
        torch.testing.assert_close(data["logq"][data["mask"]], original_q[data["mask"]], rtol=0, atol=0)

    keys = list(zip(ids, positions, strict=True))
    fresh = torch.tensor([key not in context["seen"] for key in keys], dtype=torch.bool)
    if parallel.tp.rank == 0 and fresh.any():
        policy_metrics.append_frame(
            context["path"],
            {
                "binding": context["binding"],
                "dp_rank": parallel.effective_dp.rank,
                "data": {key: value[fresh] for key, value in data.items()},
            },
        )
        context["frames"] += 1
        context["valid_tokens"] += int(data["mask"][fresh].sum())
    if not fresh.all():
        raise ValueError("Repeated original tokens in the selected single-step observation")
    context["seen"].update(keys)
    context["capture_seconds"] += time.perf_counter() - start
    if context["binding"]["completed_update"] in args.policy_audit_updates:
        _audit(args, batch, logits, data)


def _audit(args, batch, logits, data):
    context, parallel = _CONTEXT, get_parallel_state()
    start, offset = time.perf_counter(), 0
    chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
        include_response_indices=True,
    )
    for sample, (chunk, tokens, positions) in enumerate(chunks):
        identity = int(batch["sample_indices"][sample])
        selected = [
            i for i, position in enumerate(positions) if (identity, int(position)) in context["audit_positions"]
        ]
        if selected:
            at = torch.tensor(selected, device=chunk.device)
            rows = torch.tensor([offset + i for i in selected])
            stats = component_statistics(
                chunk[at],
                tokens[at],
                data["advantage"][rows].to(chunk.device),
                data["ratio"][rows].to(chunk.device),
                data["weight"][rows].to(chunk.device),
                vocab_size=args.vocab_size,
                temperature=args.rollout_temperature,
                denominator=sum(s["valid_tokens"] for s in context["global_samples"].values()),
                group=parallel.tp.group if parallel.tp.size > 1 else None,
            )
            if parallel.tp.rank == 0:
                destination = context["root"] / "logit_audit" / f"update-{context['binding']['completed_update']:06d}"
                destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                path = destination / f"sample-{identity}.pt"
                if not path.exists():
                    torch.save(
                        {
                            "binding": context["binding"],
                            "positions": [(identity, int(positions[i])) for i in selected],
                            **stats,
                        },
                        path,
                    )
                    path.chmod(0o600)
        offset += len(chunk)
    context["audit_seconds"] += time.perf_counter() - start


def finish(args, outcome):
    if not enabled(args) or not get_parallel_state().is_pp_last_stage:
        return
    context, parallel = _CONTEXT, get_parallel_state()
    if parallel.tp.rank == 0:
        context["path"].touch(exist_ok=True)
        with context["path"].open("ab") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        policy_metrics.write_json(
            context["path"].with_suffix(".complete.json"),
            {
                "binding": context["binding"],
                "sha256": policy_metrics.sha256(context["path"]),
                "frames": context["frames"],
                "valid_tokens": context["valid_tokens"],
                "samples": context["samples"],
                "capture_seconds": context["capture_seconds"],
                "audit_seconds": context["audit_seconds"],
                "outcome": getattr(outcome, "name", str(outcome).split(".")[-1]),
            },
        )
        if parallel.effective_dp.size > 1:
            dist.barrier(group=parallel.effective_dp.group)
        if parallel.effective_dp.rank == 0:
            _summarize(args, context)


def _summarize(args, context):
    summary_start = time.perf_counter()
    binding = context["binding"]
    artifact_seals = {}
    data, samples = policy_metrics.load_shards(
        context["path"].parent, dp_size=binding["dp_size"], expected_samples=context["global_samples"], binding=binding
    )
    metrics = policy_metrics.summarize_tokens(data, samples)
    seals = [
        json.loads(context["path"].with_name(f"dp-{rank}.complete.json").read_text())
        for rank in range(binding["dp_size"])
    ]
    metrics["policy/overhead/capture_max_rank_seconds"] = max(seal["capture_seconds"] for seal in seals)
    metrics["policy/overhead/audit_max_rank_seconds"] = max(seal["audit_seconds"] for seal in seals)
    completed = binding["completed_update"]
    if completed in args.policy_audit_updates:
        packets = [
            torch.load(p, weights_only=False)
            for p in (context["root"] / "logit_audit" / f"update-{completed:06d}").glob("*.pt")
        ]
        if any(
            packet["binding"] != binding
            or not torch.isfinite(packet["norm2"]).all()
            or not torch.isfinite(packet["entropy_direction"]).all()
            for packet in packets
        ):
            raise ValueError("Mixed or nonfinite sparse audit bindings")
        positions = [tuple(position) for packet in packets for position in packet["positions"]]
        if set(positions) != context["audit_positions"] or len(positions) != len(set(positions)):
            raise ValueError("Sparse audit global coverage differs")
        metrics["policy/audit/positions"] = len(positions)
        metrics["policy/audit/nonzero_advantage_positions"] = sum(
            samples[identity]["advantage"] != 0 for identity, _ in positions
        )
        metrics["policy/audit/informative"] = int(metrics["policy/audit/nonzero_advantage_positions"] > 0)

        metrics["policy/audit/both_signs_available"] = int(
            any(s["advantage"] > 0 for s in samples.values()) and any(s["advantage"] < 0 for s in samples.values())
        )
        for column, name in enumerate(packets[0]["components"]):
            metrics[f"policy/audit/{name}/entropy_direction"] = sum(
                float(p["entropy_direction"][:, column].sum()) for p in packets
            )
        histogram = {
            key: torch.histogram(value.double(), bins=40)
            for key, value in data.items()
            if key in ("ratio", "weight", "advantage", "entropy")
        }
        histogram["native_interval"] = {
            "low": 0.5,
            "high": 2.0,
            "inclusive": True,
            "equal_low_tokens": int((data["ratio"] == 0.5).sum()),
            "equal_high_tokens": int((data["ratio"] == 2.0).sum()),
            "strict_lower_tokens": int((data["ratio"] < 0.5).sum()),
            "strict_upper_tokens": int((data["ratio"] > 2.0).sum()),
        }
        histogram_path = context["path"].parent / "histograms.pt"
        torch.save(histogram, histogram_path)
        histogram_path.chmod(0o600)
    if completed in args.policy_audit_updates:
        audit_root = context["root"] / "logit_audit" / f"update-{completed:06d}"
        for artifact in [context["path"].parent / "histograms.pt", *sorted(audit_root.glob("*.pt"))]:
            artifact_seals[str(artifact.relative_to(context["root"]))] = policy_metrics.sha256(artifact)
    metrics["policy/overhead/summary_seconds"] = time.perf_counter() - summary_start
    metrics.update(
        {"policy/step": completed, "policy/optimizer_step": completed, "policy/rollout_id": binding["rollout_id"]}
    )
    policy_metrics.write_json(
        context["path"].parent / "complete.json",
        {
            "binding": binding,
            "metrics": metrics,
            "artifact_sha256": artifact_seals,
            "categories": {
                str(identity): {k: sample[k] for k in ("theme", "scenario")} for identity, sample in samples.items()
            },
        },
    )
    policy_metrics.publish_metrics(args, metrics, context["path"].parent / "tracking.json")
