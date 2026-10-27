"""Generation-time candidate transport; never reconstruct stale sampler scores."""

import numpy as np

from miles.utils.score_centering import validate_candidate_sampling


def configure_rollout_topk_logprobs_request(args, request):
    k = getattr(args, "rollout_top_logprobs_num", 0)
    if not k:
        return
    sampling = request["sampling_params"]
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    validate_candidate_sampling(sampling, temperature=args.rollout_temperature)
    request["top_logprobs_num"] = k


def append_rollout_topk_logprobs(sample, meta, k):
    if not k:
        return
    n = len(meta.get("output_token_logprobs") or [])
    rows = meta.get("output_top_logprobs")
    if n and (rows is None or len(rows) != n):
        raise ValueError("SFC requires generation-time output_top_logprobs for every generated token")
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows or []):
        for j, entry in enumerate(entries[:k]):
            logps[i, j], ids[i, j] = entry[:2]
    for name, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, name)
        setattr(sample, name, values if previous is None else np.concatenate((previous, values)))


def validate_rollout_topk_logprobs_sample(sample, k):
    """Validate once at conversion, including custom producers and restored data."""
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError("SFC requires sampled-token and candidate logprobs on every sample")
    if ids.shape[1] != k or ids.dtype.kind not in "iu":
        raise ValueError("SFC requires integer candidate IDs with the configured width")
    valid = ids >= 0
    if (ids < -1).any() or not np.isneginf(logps[~valid]).all():
        raise ValueError("SFC candidate padding requires ID -1 and logprob -inf")
    if not np.isfinite(logps[valid]).all() or (logps[valid] > 1e-6).any():
        raise ValueError("SFC candidate logprobs must be finite and nonpositive")
    ordered = np.sort(ids, axis=-1)
    if ((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])).any():
        raise ValueError("Duplicate SFC candidate token IDs")
    active = np.asarray(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool)
    if (active & ~valid.any(-1)).any() or (np.exp(logps).sum(-1) > 1.0001).any():
        raise ValueError("SFC candidates need nonempty trained heads with probability mass <= 1")
    sampled = np.asarray(sample.rollout_log_probs)
    if not np.isfinite(sampled[active]).all() or (sampled[active] > 1e-6).any():
        raise ValueError("SFC sampled-token logprobs must be finite and nonpositive on trained tokens")
    tokens = np.asarray(sample.tokens[-sample.response_length:] if sample.response_length else [], dtype=np.int64)
    matches = (ids == tokens[:, None]) & valid & active[:, None]
    if not np.allclose(np.broadcast_to(sampled[:, None], logps.shape)[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate logprobs describe different sampler distributions")


def pad_rollout_topk_logprobs(sample, count):
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def merge_rollout_topk_logprobs_field(first, second, field, gap):
    a, b = getattr(first, field), getattr(second, field)
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field}")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
