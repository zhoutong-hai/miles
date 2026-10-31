"""Original-token policy diagnostics, independent of distributed/model imports."""

import hashlib
import io
import json
import os
from pathlib import Path
import struct

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def publish_metrics(args, metrics, receipt_path):
    """The numeric spool precedes transport; a tracking failure remains explicit."""
    from miles.utils.tracking_utils import tracking

    state = "SENT_ACCEPTANCE_PENDING_API_READBACK"
    error_type = None
    try:
        tracking.define_step_key_metric_group("policy", "policy/step")
        tracking.log(args, metrics, step_key="policy/step")
    except Exception as error:
        state, error_type = "DEGRADED_DURABLE_METRICS_PRESERVED", type(error).__name__
    write_json(receipt_path, {"state": state, "exception_type": error_type, "step": metrics["policy/step"]})


def append_frame(path, frame):
    buffer = io.BytesIO()
    torch.save(frame, buffer)
    payload = buffer.getvalue()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab") as stream:
        stream.write(struct.pack("<Q", len(payload)))
        stream.write(payload)


def read_frames(path):
    with Path(path).open("rb") as stream:
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError("Truncated policy frame header")
            (size,) = struct.unpack("<Q", header)
            if not 0 < size <= 1 << 30:
                raise ValueError("Invalid policy frame size")
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError("Truncated policy frame payload")
            yield torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


def select_positions(samples, count=32):
    by_sign = {1: [], -1: [], 0: []}
    for identity, sample in sorted(samples.items()):
        sign = int(sample["advantage"] > 0) - int(sample["advantage"] < 0)
        by_sign[sign].extend((identity, p) for p in sample["first_valid_positions"])
    selected = sorted(by_sign[1])[: count // 2] + sorted(by_sign[-1])[: count // 2]
    chosen = set(selected)
    selected += [p for p in sorted(sum(by_sign.values(), [])) if p not in chosen]
    return sorted(selected[:count])


def _quantiles(output, name, values):
    for quantile, value in zip(
        (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999),
        torch.quantile(
            values.double(), torch.tensor([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999], dtype=torch.float64)
        ),
        strict=True,
    ):
        output[f"policy/{name}/q{quantile:g}"] = float(value)
    output[f"policy/{name}/min"] = float(values.min())
    output[f"policy/{name}/max"] = float(values.max())


def _slice_statistics(output, data, chosen, label, *, low, high):
    count, total = int(chosen.sum()), len(chosen)
    prefix = f"policy/{label}"
    output[prefix + "/tokens"] = count
    output[prefix + "/token_fraction"] = count / total
    ratio, weight, advantage = (data[key].double() for key in ("ratio", "weight", "advantage"))
    for name, tail in (("floor", ratio < low), ("cap", ratio > high), ("total", (ratio < low) | (ratio > high))):
        selected = chosen & tail
        output[f"{prefix}/tis_{name}_tokens"] = int(selected.sum())
        output[f"{prefix}/tis_{name}_global_fraction"] = float(selected.double().mean())
        if count:
            output[f"{prefix}/tis_{name}_conditional_fraction"] = float(tail[chosen].double().mean())
    if not count:
        return  # no fabricated zero for an undefined conditional statistic
    coefficient = weight * advantage
    mis = torch.where((ratio >= low) & (ratio <= high), ratio, 0.0)
    values = {
        "entropy": data["entropy"],
        "advantage": advantage,
        "weight": weight,
        "pg_coefficient": coefficient,
        "unweighted_pg_coefficient": advantage,
        "tis_minus_unweighted_coefficient": (weight - 1) * advantage,
        "counterfactual_mis_minus_tis_coefficient": (mis - weight) * advantage,
    }
    for name, value in values.items():
        value = value[chosen].double()
        output[f"{prefix}/{name}_mean"] = float(value.mean())
        output[f"{prefix}/{name}_abs_mean"] = float(value.abs().mean())
        output[f"{prefix}/{name}_std"] = float(value.std(unbiased=False))


def summarize_tokens(data, samples, *, low=0.5, high=2.0):
    """No rank-mean averaging, mask renormalization, or FP64 boundary recomputation."""
    size = len(data["ratio"])
    if not size or any(not torch.isfinite(value).all() for value in data.values()):
        raise ValueError("Missing or nonfinite original-token observations")
    ratio, weight = data["ratio"].double(), data["weight"].double()
    if (ratio <= 0).any():
        raise ValueError("Native ratio underflow; withhold raw-ratio statistics")
    torch.testing.assert_close(weight, ratio.clamp(low, high), rtol=0, atol=0)
    log_ratio = (data["logp"] - data["logq"]).double()
    torch.testing.assert_close(ratio, log_ratio.exp(), rtol=2e-5, atol=1e-6)
    output = {
        "policy/valid_tokens": size,
        "policy/responses": len(samples),
        "policy/entropy/token_mean": float(data["entropy"].double().mean()),
        "policy/entropy/coefficient": 0.0,
        "policy/entropy/loss_contribution": 0.0,
        "policy/surprise/sampler_mean": float(-data["logq"].double().mean()),
        "policy/surprise/trainer_mean": float(-data["logp"].double().mean()),
        "policy/logp_abs_difference_mean": float(log_ratio.abs().mean()),
        "policy/sampled_kl_q_to_p_k3": float((log_ratio.exp() - 1 - log_ratio).mean()),
        "policy/sampled_kl_q_to_p_k3/log_ratio_outside_native_clamp_fraction": float(
            (log_ratio.abs() > 10).double().mean()
        ),
    }
    clamped = log_ratio.clamp(-10, 10)
    output["policy/sampled_kl_q_to_p_native_clamped"] = float((clamped.exp() - 1 - clamped).clamp(-10, 10).mean())
    for name, values in (("log_ratio", log_ratio), ("ratio", ratio), ("weight", weight)):
        _quantiles(output, name, values)
    for name, values in (("raw", ratio), ("applied", weight)):
        scaled = values / values.max()  # stable ESS without changing native ratios
        ess = float(scaled.sum().square() / scaled.square().sum())
        output[f"policy/ess/{name}"] = ess
        output[f"policy/ess/{name}_fraction"] = ess / size
    advantage = data["advantage"]
    slices = {
        "all": torch.ones(size, dtype=torch.bool),
        "adv_positive": advantage > 0,
        "adv_negative": advantage < 0,
        "adv_zero": advantage == 0,
    }
    for label, chosen in slices.items():
        _slice_statistics(output, data, chosen, label, low=low, high=high)
        for tail_name, tail in (
            ("lt0.1", ratio < 0.1),
            ("lt0.5", ratio < 0.5),
            ("gt2", ratio > 2),
            ("gt5", ratio > 5),
        ):
            output[f"policy/{label}/raw_ratio_{tail_name}_tokens"] = int((tail & chosen).sum())
            output[f"policy/{label}/raw_ratio_{tail_name}_global_fraction"] = float((tail & chosen).double().mean())
            if chosen.any():
                output[f"policy/{label}/raw_ratio_{tail_name}_conditional_fraction"] = float(
                    tail[chosen].double().mean()
                )

    response_entropies = []
    for identity, sample in samples.items():
        chosen = data["sample_id"] == identity
        if int(chosen.sum()) != sample["valid_tokens"]:
            raise ValueError("Per-response token coverage differs")
        expected_positions = sample.get("valid_positions", sample["first_valid_positions"])
        if set(data["position"][chosen].tolist()) != set(expected_positions):
            raise ValueError("Original valid-token positions differ")
        if (data["advantage"][chosen] != sample["advantage"]).any():
            raise ValueError("Paired advantages disagree with the original response")

        if chosen.any():
            response_entropies.append(float(data["entropy"][chosen].double().mean()))
    output["policy/entropy/equal_nonempty_response_mean"] = sum(response_entropies) / len(response_entropies)
    output["policy/empty_responses"] = len(samples) - len(response_entropies)
    for field in ("theme", "scenario"):
        for value in sorted({str(sample[field]) for sample in samples.values()}):
            ids = [identity for identity, sample in samples.items() if str(sample[field]) == value]
            chosen = torch.tensor([int(identity) in ids for identity in data["sample_id"]])
            safe = hashlib.sha256(value.encode()).hexdigest()[:12]
            _slice_statistics(output, data, chosen, f"{field}/{safe}", low=low, high=high)
    return output


def load_shards(directory, *, dp_size, expected_samples, binding):
    frames, samples, token_count = [], {}, 0
    for rank in range(dp_size):
        path = Path(directory) / f"dp-{rank}.bin"
        seal = json.loads(path.with_suffix(".complete.json").read_text())
        if seal["binding"] != binding or seal["sha256"] != sha256(path) or seal["outcome"] != "NORMAL":
            raise ValueError("Invalid policy shard identity, seal or optimizer outcome")
        part = list(read_frames(path))
        if len(part) != seal["frames"]:
            raise ValueError("Policy frame count differs")
        for frame in part:
            if frame["binding"] != binding or frame["dp_rank"] != rank:
                raise ValueError("Mixed policy frame identities")
        overlap = set(samples) & set(map(int, seal["samples"]))
        if overlap:
            raise ValueError("DP sample ownership overlaps")
        samples.update({int(k): v for k, v in seal["samples"].items()})
        frames += part
        token_count += seal["valid_tokens"]
    if set(samples) != set(expected_samples) or not frames:
        raise ValueError("Missing global sample coverage")
    columns = {key: torch.cat([frame["data"][key] for frame in frames]) for key in frames[0]["data"]}
    data = {key: value[columns["mask"].bool()] for key, value in columns.items() if key != "mask"}
    identities = list(zip(data["sample_id"].tolist(), data["position"].tolist(), strict=True))
    if len(identities) != token_count or len(set(identities)) != token_count:
        raise ValueError("Missing or duplicated original tokens")
    return data, samples
