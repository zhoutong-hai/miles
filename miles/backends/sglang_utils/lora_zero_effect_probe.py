"""Deterministic runtime probes for a freshly published zero-effect LoRA."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any


def build_probe_payload(*, batch_size: int, lora_path: str | None, stochastic: bool) -> dict[str, Any]:
    """Build a small greedy equality or production stochastic-smoke request."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompt = "A box holds 12 pencils. Five pencils are removed and three are added. How many pencils are in the box? Show the calculation, then give the answer."
    payload: dict[str, Any] = {
        "text": [prompt] * batch_size,
        "sampling_params": [
            {
                "temperature": 1.0 if stochastic else 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_new_tokens": 4,
                "max_new_tokens": 32,
            }
            for _ in range(batch_size)
        ],
        "return_logprob": True,
        "return_routed_experts": True,
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    return payload


def summarize_probe_response(response: Any, *, expected_batch_size: int) -> dict[str, Any]:
    """Validate one /generate response and retain compact deterministic evidence."""
    outputs = response if isinstance(response, list) else [response]
    if len(outputs) != expected_batch_size:
        raise RuntimeError(f"probe response batch mismatch: expected={expected_batch_size} observed={len(outputs)}")

    summaries = []
    for output_index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            raise RuntimeError(f"probe output {output_index} is not a mapping")
        meta = output.get("meta_info")
        if not isinstance(meta, Mapping):
            raise RuntimeError(f"probe output {output_index} omitted meta_info")
        raw_logprobs = meta.get("output_token_logprobs")
        if not isinstance(raw_logprobs, Sequence) or not raw_logprobs:
            raise RuntimeError(f"probe output {output_index} omitted output token logprobs")

        token_ids: list[int] = []
        logprobs: list[float] = []
        for token_index, item in enumerate(raw_logprobs):
            if not isinstance(item, Sequence) or len(item) < 2:
                raise RuntimeError(f"probe output {output_index} token {token_index} has malformed logprob data")
            logprob = float(item[0])
            token_id = int(item[1])
            if not math.isfinite(logprob):
                raise RuntimeError(f"probe output {output_index} token {token_index} has non-finite logprob")
            logprobs.append(logprob)
            token_ids.append(token_id)

        routed = meta.get("routed_experts")
        if not isinstance(routed, str) or not routed:
            raise RuntimeError(f"probe output {output_index} omitted routed-expert evidence")
        summaries.append(
            {
                "text": str(output.get("text", "")),
                "token_ids": token_ids,
                "logprobs": logprobs,
                "routed_experts_sha256": hashlib.sha256(routed.encode()).hexdigest(),
            }
        )
    return {"batch_size": expected_batch_size, "outputs": summaries}


def _compare_case(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> float:
    if left.get("batch_size") != right.get("batch_size"):
        raise RuntimeError(f"{label}: batch size changed")
    left_outputs = left.get("outputs")
    right_outputs = right.get("outputs")
    if not isinstance(left_outputs, list) or not isinstance(right_outputs, list):
        raise RuntimeError(f"{label}: malformed output summaries")
    if len(left_outputs) != len(right_outputs):
        raise RuntimeError(f"{label}: output count changed")

    max_abs_logprob_diff = 0.0
    for output_index, (before, after) in enumerate(zip(left_outputs, right_outputs, strict=True)):
        if before["token_ids"] != after["token_ids"]:
            raise RuntimeError(f"{label}: output {output_index} token IDs changed")
        if before["text"] != after["text"]:
            raise RuntimeError(f"{label}: output {output_index} text changed")
        if before["routed_experts_sha256"] != after["routed_experts_sha256"]:
            raise RuntimeError(f"{label}: output {output_index} routed experts changed")
        left_logprobs = before["logprobs"]
        right_logprobs = after["logprobs"]
        if len(left_logprobs) != len(right_logprobs):
            raise RuntimeError(f"{label}: output {output_index} logprob length changed")
        for a, b in zip(left_logprobs, right_logprobs, strict=True):
            max_abs_logprob_diff = max(max_abs_logprob_diff, abs(a - b))
    if max_abs_logprob_diff > 1e-5:
        raise RuntimeError(f"{label}: max logprob difference {max_abs_logprob_diff} exceeded 1e-5")
    return max_abs_logprob_diff


def validate_initial_zero_effect_probes(
    before_results: Sequence[Mapping[str, Any]],
    after_results: Sequence[Mapping[str, Any]],
    *,
    expected_engine_ranks: Sequence[int],
) -> dict[str, Any]:
    """Require base stability and base/zero-LoRA equality on every engine."""
    expected_ranks = {int(rank) for rank in expected_engine_ranks}
    if not expected_ranks or len(expected_ranks) != len(expected_engine_ranks):
        raise RuntimeError("zero-effect probe expected engine ranks are empty or duplicated")
    for expected_phase, results in (("before", before_results), ("after", after_results)):
        for item in results:
            if item.get("schema_version") != 1 or item.get("phase") != expected_phase:
                raise RuntimeError(
                    f"zero-effect probe invalid {expected_phase} receipt: "
                    f"schema={item.get('schema_version')} phase={item.get('phase')!r}"
                )
    before_by_engine = {int(item["engine_rank"]): item for item in before_results}
    after_by_engine = {int(item["engine_rank"]): item for item in after_results}
    if len(before_by_engine) != len(before_results) or len(after_by_engine) != len(after_results):
        raise RuntimeError("duplicate engine rank in zero-effect probe")
    if set(before_by_engine) != expected_ranks or set(after_by_engine) != expected_ranks:
        raise RuntimeError(
            "zero-effect probe engine coverage mismatch: "
            f"expected={sorted(expected_ranks)} before={sorted(before_by_engine)} "
            f"after={sorted(after_by_engine)}"
        )

    comparisons = []
    for engine_rank in sorted(before_by_engine):
        before = before_by_engine[engine_rank]
        after = after_by_engine[engine_rank]
        for batch_size in (1, 8):
            case = f"greedy_batch{batch_size}"
            baseline = before["cases"][case]["base"]
            post_base = after["cases"][case]["base"]
            post_lora = after["cases"][case]["lora"]
            base_diff = _compare_case(
                baseline,
                post_base,
                label=f"engine={engine_rank} {case} pre-base/post-base",
            )
            lora_diff = _compare_case(
                post_base,
                post_lora,
                label=f"engine={engine_rank} {case} post-base/zero-lora",
            )
            comparisons.append(
                {
                    "engine_rank": engine_rank,
                    "batch_size": batch_size,
                    "mode": "greedy_equality",
                    "base_max_abs_logprob_diff": base_diff,
                    "lora_max_abs_logprob_diff": lora_diff,
                }
            )
            stochastic_case = f"stochastic_batch{batch_size}"
            # summarize_probe_response already proved every returned token
            # logprob finite. Presence of each result is the production
            # FlashInfer smoke gate; stochastic outputs are not compared.
            for phase_name, result in (
                ("pre_base", before["cases"][stochastic_case]["base"]),
                ("post_base", after["cases"][stochastic_case]["base"]),
                ("post_lora", after["cases"][stochastic_case]["lora"]),
            ):
                if result.get("batch_size") != batch_size or not result.get("outputs"):
                    raise RuntimeError(f"engine={engine_rank} {stochastic_case} {phase_name}: incomplete smoke result")
                comparisons.append(
                    {
                        "engine_rank": engine_rank,
                        "batch_size": batch_size,
                        "mode": "production_stochastic_smoke",
                        "phase": phase_name,
                        "finite": True,
                    }
                )
    return {
        "status": "PASS",
        "engine_count": len(before_by_engine),
        "greedy_comparison_count": sum(item["mode"] == "greedy_equality" for item in comparisons),
        "stochastic_smoke_count": sum(item["mode"] == "production_stochastic_smoke" for item in comparisons),
        "comparisons": comparisons,
    }
