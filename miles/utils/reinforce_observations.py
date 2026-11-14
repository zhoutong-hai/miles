"""Private numeric CP1 REINFORCE receipts; no model payload or clinical text is read."""

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import torch

_context = None


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(data)


def _bias_digest(model):
    digest = hashlib.sha256()
    trainable = 0
    for chunk in model:
        for name, parameter in chunk.named_parameters():
            if parameter.requires_grad:
                if "lora" not in name.lower():
                    raise ValueError("Native baseline has a trainable non-LoRA parameter")
                trainable += parameter.numel()
            if "e_score_correction_bias" in name or "expert_bias" in name:
                if parameter.requires_grad:
                    raise ValueError("Router correction bias is trainable")
                digest.update(name.encode())
                digest.update(parameter.detach().float().cpu().numpy().tobytes())
        for name, value in chunk.named_buffers():
            if "e_score_correction_bias" in name or "expert_bias" in name:
                digest.update(name.encode())
                digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest(), trainable


def configure(args, rollout_id, rollout_data, model, optimizer, attempt):
    global _context
    from miles.backends.training_utils.parallel import get_parallel_state

    state = get_parallel_state()
    rank = torch.distributed.get_rank()
    root = Path(args.reinforce_diagnostics_dir) / f"update-{rollout_id + 1:06d}" / f"attempt-{attempt}"
    root.mkdir(parents=True, exist_ok=True)
    if (root / f"rank-{rank}-complete.json").exists():
        raise ValueError("Unresolved REINFORCE receipt identity cannot be reused")
    bias, numel = _bias_digest(model)
    from miles.utils.reinforce_audits import attach_routers, optimizer_membership
    from miles.utils.replay_base import all_replay_managers

    membership = optimizer_membership(model, optimizer, state.intra_dp)
    _context = {"root": root, "rank": rank, "writer": state.tp.rank == 0, "rollout": rollout_id, "bias": bias, "trainable_numel": numel}
    interval = args.reinforce_router_interval
    if interval and (rollout_id == 0 or (rollout_id + 1) % interval == 0):
        _context["router_counts"], _context["router_handles"] = attach_routers(model)
    _write_once(
        root / f"rank-{rank}-binding.json",
        {
            "rank": rank,
            "host": os.uname().nodename,
            "source": os.environ["MILES_SOURCE_COMMIT"],
            "LoRA_trainable_numel": numel,
            "optimizer_excludes_all_frozen_parameters": True,
            "optimizer_membership": membership,
            "loaded_bias_before": bias,
            "use_rollout_routing_replay": args.use_rollout_routing_replay,
            "replay_layers": {manager.name: len(manager.replays) for manager in all_replay_managers if manager.enabled},
        },
    )
    if state.tp.rank == 0:
        rows = []
        for i, sample in enumerate(rollout_data["sample_indices"]):
            length = rollout_data["response_lengths"][i]
            rows.append(
                {
                    "sample": int(sample),
                    "group": int(rollout_data["prompt_group_indices"][i]),
                    "group_tokens": int(rollout_data["prompt_group_tokens"][i]),
                    "factor": float(rollout_data["prompt_loss_scales"][i]),
                    "tags": rollout_data["prompt_group_tags"][i],
                    "tokens": rollout_data["tokens"][i][-length:].tolist(),
                    "mask": rollout_data["loss_masks"][i].tolist(),
                    "q": rollout_data["rollout_log_probs"][i].float().tolist(),
                    "advantage": rollout_data["advantages"][i].float().tolist(),
                }
            )
        _write_once(root / f"rank-{rank}-input.json", {"rollout": rollout_id, "rows": rows, "original_weight_versions": rollout_data.get("weight_versions")})


def capture(args, batch, layout, logits, logp, q, advantage, masks, factors, weight, ratio, entropy):
    if _context is None:
        raise ValueError("REINFORCE observation was not configured by the actual trainer")
    interval = args.reinforce_sparse_interval
    if interval and (_context["rollout"] == 0 or (_context["rollout"] + 1) % interval == 0):
        from miles.utils.reinforce_audits import sparse_output_audit

        sparse_output_audit(_context, args, batch, layout, logits, advantage, factors, weight, ratio)
    if not _context["writer"]:
        return
    columns = {"p": logp.detach().float(), "q": q.detach().float(), "A": advantage.detach().float(), "mask": masks, "factor": factors, "weight": weight, "ratio": ratio, "entropy": entropy.detach()}
    columns = {name: [x.cpu().tolist() for x in layout.per_sample(value)] for name, value in columns.items()}
    rows = [{"sample": int(sid), "group": int(batch["prompt_group_indices"][i]), **{key: values[i] for key, values in columns.items()}} for i, sid in enumerate(batch["sample_indices"])]
    path = _context["root"] / f"rank-{_context['rank']}-frames.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"rows": rows}, allow_nan=False) + "\n")


def finish(model, outcome):
    if _context is None:
        raise ValueError("Missing REINFORCE trainer context")
    bias, numel = _bias_digest(model)
    if bias != _context["bias"] or numel != _context["trainable_numel"]:
        raise ValueError("Frozen loaded router bias or trainable inventory changed")
    for handle in _context.get("router_handles", []):
        handle.remove()
    if "router_counts" in _context:
        _write_once(_context["root"] / f"rank-{_context['rank']}-router.json", {"rank": _context["rank"], "scope": "executed routing including padding/recompute; no unique-token or timing claim", "counts": {name: x.cpu().tolist() for name, x in _context["router_counts"].items()}})
    _write_once(_context["root"] / f"rank-{_context['rank']}-complete.json", {"rank": _context["rank"], "completed_update": _context["rollout"] + 1, "outcome": outcome.name, "bias_before_after": bias, "LoRA_numel": numel})


def finalize(args, rollout_id):
    """Seal only after native training and learned publication both returned successfully."""
    root = Path(args.reinforce_diagnostics_dir) / f"update-{rollout_id + 1:06d}" / "attempt-0"
    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    receipts = [json.loads((root / f"rank-{rank}-complete.json").read_text()) for rank in range(world)]
    if any(x["outcome"] != "NORMAL" or x["completed_update"] != rollout_id + 1 for x in receipts):
        raise ValueError("REINFORCE requires every rank's successful ordinary update")
    original, actual = {}, {}
    for path in root.glob("rank-*-input.json"):
        for row in json.loads(path.read_text())["rows"]:
            if row["sample"] in original:
                raise ValueError("Duplicate data-parallel input response")
            original[row["sample"]] = row
    for path in root.glob("rank-*-frames.jsonl"):
        for line in path.read_text().splitlines():
            for row in json.loads(line)["rows"]:
                if row["sample"] in actual:
                    raise ValueError("Duplicate microbatch response")
                actual[row["sample"]] = row
    if set(original) != set(actual) or len(actual) != args.global_batch_size:
        raise ValueError("Original full optimizer batch coverage differs")
    totals = defaultdict(int)
    counts = Counter()
    for row in original.values():
        totals[row["group"]] += sum(row["mask"])
        counts[row["group"]] += 1
    if len(counts) != args.rollout_batch_size or set(counts.values()) != {args.n_samples_per_prompt}:
        raise ValueError("Original complete prompt/sibling population differs")
    tokens, pg, entropy_sum, response_H, mass, square = 0, 0.0, 0.0, 0.0, 0.0, 0.0
    groups_H, eligible_groups, category_H = defaultdict(list), set(), defaultdict(list)
    for sid, row in actual.items():
        source = original[sid]
        expected_factor = float(torch.tensor(sum(totals.values()) / (len(totals) * totals[source["group"]]), dtype=torch.float32))
        if source["group_tokens"] != totals[source["group"]] or row["group"] != source["group"]:
            raise ValueError("Original prompt identity or denominator changed")
        if row["q"] != source["q"] or row["A"] != source["advantage"] or row["mask"] != source["mask"]:
            raise ValueError("Original q/advantages/mask changed before loss")
        for j, valid in enumerate(row["mask"]):
            if not valid:
                continue
            if row["factor"][j] != source["factor"]:
                # Device factors are FP32, compared at their actual precision.
                if row["factor"][j] != float(torch.tensor(source["factor"], dtype=torch.float32)):
                    raise ValueError("Original prompt factor differs")
            w = row["weight"][j]
            a = row["A"][j]
            r = row["ratio"][j]
            reference_r = math.exp(row["p"][j] - row["q"][j])
            bounds = (args.mis_positive_low, args.mis_positive_high) if a > 0 else (args.mis_negative_low, args.mis_negative_high)
            if not math.isclose(r, reference_r, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Native raw ratio does not replay from original q/current p")
            if w != (r if bounds[0] <= r <= bounds[1] else 0.0) or row["factor"][j] != expected_factor:
                raise ValueError("Native fixed MIS weight or original prompt denominator differs")
            tokens += 1
            pg -= row["p"][j] * w * a * row["factor"][j]
            entropy_sum += row["entropy"][j]
            mass += w
            square += w * w
        h = sum(h * m for h, m in zip(row["entropy"], row["mask"], strict=True)) / sum(row["mask"])
        response_H += h
        groups_H[row["group"]].append(h)
        if any(a != 0 for a, m in zip(row["A"], row["mask"], strict=True) if m):
            eligible_groups.add(row["group"])
        for key, value in source["tags"].items():
            if value is not None:
                category_H[(key, str(value))].append(h)
    metrics = {
        "train/step": rollout_id,
        "train/completed_optimizer_updates": rollout_id + 1,
        "train/sfc/pg_prompt_mean_replay": pg / tokens,
        "train/sfc/entropy_pooled_token_mean": entropy_sum / tokens,
        "train/sfc/entropy_equal_response_mean": response_H / len(actual),
        "train/sfc/raw_original_tokens": tokens,
        "train/sfc/applied_weight_ess_fraction": mass * mass / (tokens * square) if square else 0.0,
    }
    metrics["train/sfc/original_nonzero_advantage_group_count"] = len(eligible_groups)
    if eligible_groups:
        metrics["train/sfc/entropy_original_nonzero_advantage_groups"] = sum(sum(groups_H[g]) / len(groups_H[g]) for g in eligible_groups) / len(eligible_groups)
    metrics.update({f"train/sfc/entropy_by_{key}/{quote(value, safe='')}/equal_response_mean": sum(values) / len(values) for (key, value), values in category_H.items()})
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    _write_once(root / "complete.json", {"completed_update": rollout_id + 1, "native_NORMAL_ranks": world, "learned_publication_returned": True, "files": files, "metrics": metrics})
    from miles.utils.tracking_utils import tracking

    tracking.log(args, metrics, step_key="train/step")
