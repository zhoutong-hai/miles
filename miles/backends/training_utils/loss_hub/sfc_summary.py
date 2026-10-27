"""Original-token pooled summaries of sealed private SFC numeric shards."""

import hashlib
import io
import json
from pathlib import Path
import struct

import torch


def read_frames(path):
    with Path(path).open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError("Truncated paired-score frame header")
            size, = struct.unpack("<Q", header)
            if not 0 < size <= 1024**3:
                raise ValueError("Invalid paired-score frame size")
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError("Truncated paired-score frame")
            yield torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def _quantiles(output, prefix, values, quantiles):
    result = torch.quantile(values.double(), torch.tensor(quantiles, dtype=torch.float64))
    output.update({f"sfc/{prefix}_q{q:g}": float(v) for q, v in zip(quantiles, result, strict=True)})


def _group_rates(output, label, groups, keys, tails, observations):
    for key in sorted(set(keys)):
        chosen = torch.tensor([value == key for value in groups], dtype=torch.bool)
        count = int(chosen.sum())
        # Only configured category identifiers enter tracking, never prompt text.
        safe = hashlib.sha256(str(key).encode()).hexdigest()[:12]
        output[f"sfc/slices/{label}/{safe}/tokens"] = count
        for name, tail in tails.items():
            output[f"sfc/slices/{label}/{safe}/{name}"] = float(tail[chosen].double().mean())
        for name, values in observations.items():
            output[f"sfc/slices/{label}/{safe}/{name}_mean"] = float(values[chosen].double().mean())


def _load_update(directory, *, expected_tokens, expected_samples, cp_size):
    directory = Path(directory)
    frames, seals = [], []
    for rank in range(cp_size):
        path = directory / f"cp-{rank}.bin"
        seal = json.loads(path.with_suffix(".complete.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != seal["sha256"]:
            raise ValueError("Paired-score shard differs from its completion seal")
        if not seal["outcome"].endswith("NORMAL"):
            raise ValueError("Paired scores do not belong to a normal optimizer update")
        rank_frames = list(read_frames(path))
        if len(rank_frames) != seal["frames"]:
            raise ValueError("Paired-score frame count differs")
        frames.extend(rank_frames)
        seals.append(seal)
    if not frames or len({(f["source"], f["trainer_version"], f["rollout"], f["run"]) for f in frames}) != 1:
        raise ValueError("Mixed or absent paired-score source/model/update identities")
    sample_metadata = {}
    for frame in frames:
        for identity, metadata, versions in zip(frame["sample_indices"], frame["samples"], frame["sampler_versions"], strict=True):
            value = {**metadata, "sampler_versions": versions}
            if identity in sample_metadata and sample_metadata[identity] != value:
                raise ValueError("CP shards disagree on sample/reward/version metadata")
            sample_metadata[identity] = value
    if set(sample_metadata) != expected_samples:
        raise ValueError("Missing or extra original response identities")
    columns = {key: torch.cat([frame["data"][key] for frame in frames]) for key in frames[0]["data"]}
    mask = columns["mask"].bool()
    data = {key: value[mask] for key, value in columns.items()}
    n = int(mask.sum())
    identities = list(zip(data["sample_id"].tolist(), data["token_position"].tolist(), strict=True))
    if n != expected_tokens or len(set(identities)) != n or sum(s["valid_tokens"] for s in seals) != n:
        raise ValueError("Original valid-token coverage or CP/recompute deduplication differs")
    if not n:
        raise ValueError("Selected ordinary update has no valid response tokens")
    for key in ("logp", "logq", "advantage", "weight", "entropy", "pg_loss", "sampler_kl", "correction", "p_head_mass", "q_head_mass"):
        if not torch.isfinite(data[key]).all():
            raise ValueError("Nonfinite original-token diagnostic: " + key)
    return data, frames, seals, sample_metadata, n


def _token_statistics(data, *, n, expected_samples, seals, mode, beta):
    log_ratio = (data["logp"]-data["logq"]).double()
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("Nonfinite active raw importance ratios")
    advantage, weight = data["advantage"].double(), data["weight"].double()
    expected_weight = ratio.clamp_max(2) if mode == "tis" else torch.ones_like(ratio)
    torch.testing.assert_close(weight, expected_weight, atol=1e-6, rtol=2e-5)
    tails = {"ratio_lt01": ratio < .1, "ratio_lt05": ratio < .5,
             "ratio_gt2": ratio > 2, "ratio_gt5": ratio > 5}
    output = {"sfc/original_valid_tokens": n, "sfc/original_responses": len(expected_samples),
              "sfc/sampler_kl_coefficient_actual": float(beta),
              "sfc/sampler_kl_raw_token_mean": float(data["sampler_kl"].double().mean()),
              "sfc/pg_token_mean": float(data["pg_loss"].double().mean()),
              "sfc/entropy_pooled_token_mean": float(data["entropy"].double().mean()),
              "sfc/applied_floor_fraction": 0.0,
              "sfc/applied_cap_fraction": float((ratio > 2).double().mean()) if mode == "tis" else 0.0,
              "sfc/applied_weight_sum": float(weight.sum()), "sfc/applied_weight_square_sum": float(weight.square().sum()),
              "sfc/raw_weight_sum": float(ratio.sum()), "sfc/raw_weight_square_sum": float(ratio.square().sum()),
              "sfc/applied_ess": float(weight.sum().square()/weight.square().sum()),
              "sfc/raw_ess": float(ratio.sum().square()/ratio.square().sum()),
              "sfc/zero_advantage_token_fraction": float((advantage == 0).double().mean()),
              "sfc/sample_in_head_fraction": float(data["sample_in_head"].double().mean()),
              "sfc/candidate_count_mean": float(data["candidate_count"].double().mean()),
              "sfc/centering_correction_abs_mean": float(data["correction"].double().abs().mean()),
              "sfc/paired_capture_max_rank_seconds": max(s["capture_seconds"] for s in seals),
              "sfc/logit_audit_sum_rank_seconds": sum(s["audit_seconds"] for s in seals)}
    output["sfc/applied_ess_fraction"] = output["sfc/applied_ess"]/n
    output["sfc/raw_ess_fraction"] = output["sfc/raw_ess"]/n
    output["sfc/sampler_kl_weighted_token_mean"] = beta*output["sfc/sampler_kl_raw_token_mean"]
    output["sfc/total_loss_token_mean"] = output["sfc/pg_token_mean"]+output["sfc/sampler_kl_weighted_token_mean"]
    _quantiles(output, "log_ratio", log_ratio, [.001, .01, .5, .99, .999])
    _quantiles(output, "centering_correction_abs", data["correction"].abs(), [.95])
    for prefix, masses in (("trainer", data["p_head_mass"]), ("sampler", data["q_head_mass"])):
        if (masses < -1e-6).any() or (masses > 1+1e-5).any():
            raise ValueError("Candidate head mass is outside probability bounds")
        output[f"sfc/{prefix}_head_mass_mean"] = float(masses.double().mean())
        tail = (1-masses).clamp_min(0)
        output[f"sfc/{prefix}_tail_mass_mean"] = float(tail.double().mean())
        _quantiles(output, prefix+"_tail_mass", tail, [.5, .95, .99])
    signs = {"positive": advantage > 0, "negative": advantage < 0, "zero": advantage == 0}
    delta = (weight-1)*advantage
    floor_delta = torch.where(ratio < .5, (.5-ratio)*advantage, 0.0)
    for label, selected in signs.items():
        for name, tail in tails.items():
            output[f"sfc/{name}_adv_{label}_fraction"] = float((tail & selected).double().mean())
        output[f"sfc/weight_advantage_delta_{label}_mean"] = float((delta*selected).mean())
        output[f"sfc/weight_advantage_delta_{label}_abs_mean"] = float((delta*selected).abs().mean())
        output[f"sfc/hypothetical_floor_delta_{label}_mean"] = float((floor_delta*selected).mean())
    for name, tail in tails.items():
        output[f"sfc/{name}_fraction"] = float(tail.double().mean())
        output[f"sfc/{name}_tokens"] = int(tail.sum())
    return output, tails


def _slice_statistics(output, data, sample_metadata, expected_samples, tails):
    ids = data["sample_id"].tolist()
    observations = {"p_head_mass": data["p_head_mass"], "q_head_mass": data["q_head_mass"],
                    "correction_abs": data["correction"].abs(), "sample_in_head": data["sample_in_head"]}
    for field in ("theme", "scenario"):
        groups = [sample_metadata[i][field] for i in ids]
        _group_rates(output, field, groups, groups, tails, observations)
    positions = data["token_position"]
    buckets = torch.where(positions < 32, 0, torch.where(positions < 128, 1, torch.where(positions < 256, 2, 3))).tolist()
    _group_rates(output, "position_bucket", buckets, buckets, tails, observations)
    sign_groups = torch.sign(data["advantage"]).tolist()
    _group_rates(output, "advantage_sign", sign_groups, sign_groups, tails, observations)
    unique_ids, inverse, counts = torch.unique(data["sample_id"], return_inverse=True, return_counts=True)
    entropy_sums = torch.zeros(len(unique_ids), dtype=torch.float64)
    entropy_sums.scatter_add_(0, inverse, data["entropy"].double())
    output["sfc/entropy_equal_response_mean"] = float((entropy_sums/counts).sum()/len(expected_samples))
    categories = {field: {hashlib.sha256(str(m[field]).encode()).hexdigest()[:12]: m[field]
                          for m in sample_metadata.values()} for field in ("theme", "scenario")}
    return categories


def _audit_integrity(directory, data, frame):
    completed = frame["completed_update"]
    if completed not in (8, 32, 64):
        return None
    root = directory.parent.parent/"logit_audit"/f"update-{completed:06d}"
    positions, packets = [], []
    for path in sorted(root.glob("cp-*.pt")):
        packet = torch.load(path, map_location="cpu", weights_only=False)
        if packet["source"] != frame["source"] or packet["trainer_version"] != frame["trainer_version"]:
            raise ValueError("Partial logit audit source/model differs")
        for key in ("norms", "dot_products", "cosines", "component_to_pg_norm_ratio",
                    "entropy_change_per_unit_logit_gradient_step"):
            if not torch.isfinite(packet[key]).all():
                raise ValueError("Nonfinite partial logit audit")
        positions.extend(map(tuple, packet["positions"]))
        packets.append(packet)
    expected = sorted(zip(data["sample_id"].tolist(), data["token_position"].tolist(), strict=True))[:32]
    if len(positions) != 32 or sorted(positions) != expected:
        raise ValueError("Partial logit audit must cover exactly32 selected original positions")
    packet = {"state": "COMPLETE32OUTPUT_LOGIT_POSITIONS", "positions": positions,
              "source": frame["source"], "trainer_version": frame["trainer_version"],
              "scope": "partial output-logit gradients only; not parameter-gradient directions"}
    with (root/"complete.json").open("x") as stream:
        json.dump(packet, stream, allow_nan=False)
    return packet


def summarize_update(directory, *, expected_tokens, expected_samples, cp_size, mode, beta):
    directory = Path(directory)
    data, frames, seals, sample_metadata, n = _load_update(
        directory, expected_tokens=expected_tokens, expected_samples=expected_samples, cp_size=cp_size)
    output, tails = _token_statistics(data, n=n, expected_samples=expected_samples, seals=seals, mode=mode, beta=beta)
    categories = _slice_statistics(output, data, sample_metadata, expected_samples, tails)
    audit = _audit_integrity(directory, data, frames[0])
    complete = {"schema_version": 1, "state": "COMPLETE_ORIGINAL_TOKEN_DIAGNOSTICS", "metrics": output,
                "categories": categories, "source": frames[0]["source"], "trainer_version": frames[0]["trainer_version"],
                "completed_update": frames[0]["completed_update"], "ranks": cp_size,
                "valid_tokens": n, "responses": len(expected_samples),
                "logit_audit": audit,
                "scope": "pooled original response tokens; sample-mean entropy includes zero-live responses as zero"}
    with (directory / "complete.json").open("x") as stream:
        json.dump(complete, stream, sort_keys=True, allow_nan=False)
    return output
