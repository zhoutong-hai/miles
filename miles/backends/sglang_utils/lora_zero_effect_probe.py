"""Runtime integrity probes for a freshly published zero-effect LoRA."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def build_probe_payload(*, batch_size: int, lora_path: str | None, stochastic: bool) -> dict[str, Any]:
    """Build a small greedy equality or production stochastic-smoke request."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompt = "A box holds 12 pencils. Five pencils are removed and three are added. How many pencils are in the box? Show the calculation, then give the answer."
    prompts = [f"{prompt} Fixed probe item {index + 1}." for index in range(batch_size)]
    payload: dict[str, Any] = {
        "text": prompts,
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


def build_dynamic_shrink_probe_payload(*, repeat: int) -> dict[str, Any]:
    """Build a deterministic production-style batch that drains 8 -> 1 rows."""
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    payload = build_probe_payload(batch_size=8, lora_path=None, stochastic=True)
    for index, sampling_params in enumerate(payload["sampling_params"]):
        sampling_params["ignore_eos"] = True
        sampling_params["max_new_tokens"] = 32 * (index + 1)
        sampling_params["sampling_seed"] = 4300 + repeat * 100 + index
    return payload


def summarize_probe_response(
    response: Any,
    *,
    expected_batch_size: int,
    context: str | None = None,
    input_texts: Sequence[str] | None = None,
    num_layers: int | None = None,
) -> dict[str, Any]:
    """Validate one /generate response and retain compact operational evidence."""
    prefix = f"{context}: " if context else ""
    outputs = response if isinstance(response, list) else [response]
    if len(outputs) != expected_batch_size:
        raise RuntimeError(f"{prefix}probe response batch mismatch: expected={expected_batch_size} observed={len(outputs)}")
    if input_texts is not None and len(input_texts) != expected_batch_size:
        raise RuntimeError(f"{prefix}probe input batch mismatch: expected={expected_batch_size} observed={len(input_texts)}")

    summaries = []
    for output_index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            raise RuntimeError(f"{prefix}probe output {output_index} is not a mapping")
        if not isinstance(output.get("text"), str):
            raise RuntimeError(f"{prefix}probe output {output_index} has malformed text")
        meta = output.get("meta_info")
        if not isinstance(meta, Mapping):
            raise RuntimeError(f"{prefix}probe output {output_index} omitted meta_info")
        raw_logprobs = meta.get("output_token_logprobs")
        if not isinstance(raw_logprobs, Sequence) or not raw_logprobs:
            raise RuntimeError(f"{prefix}probe output {output_index} omitted output token logprobs")

        token_ids: list[int] = []
        logprobs: list[float] = []
        for token_index, item in enumerate(raw_logprobs):
            if not isinstance(item, Sequence) or len(item) < 2:
                raise RuntimeError(f"{prefix}probe output {output_index} token {token_index} has malformed logprob data")
            raw_logprob, raw_token_id = item[0], item[1]
            if raw_logprob is None:
                raise RuntimeError(f"{prefix}probe output {output_index} token {token_index} has null output logprob; the server likely serialized a non-finite value")
            if isinstance(raw_logprob, bool):
                # Legacy immutable preflight marker: has non-numeric logprob/token data
                raise RuntimeError(f"{prefix}probe output {output_index} token {token_index} has non-numeric logprob data")
            try:
                logprob = float(raw_logprob)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{prefix}probe output {output_index} token {token_index} has non-numeric logprob data") from exc
            if not math.isfinite(logprob):
                raise RuntimeError(f"{prefix}probe output {output_index} token {token_index} has non-finite logprob")
            if type(raw_token_id) is not int or raw_token_id < 0:
                raise RuntimeError(f"{prefix}probe output {output_index} token {token_index} has invalid token ID")
            logprobs.append(logprob)
            token_ids.append(raw_token_id)

        routed = meta.get("routed_experts")
        if not isinstance(routed, str) or not routed:
            raise RuntimeError(f"{prefix}probe output {output_index} omitted routed-expert evidence")
        prompt_tokens = meta.get("prompt_tokens")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise RuntimeError(f"{prefix}probe output {output_index} omitted positive prompt token count")
        try:
            raw_routed = base64.b64decode(routed.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise RuntimeError(f"{prefix}probe output {output_index} has malformed routed-expert evidence") from exc
        position_count = prompt_tokens + len(token_ids) - 1
        if position_count <= 0 or len(raw_routed) % 4:
            raise RuntimeError(f"{prefix}probe output {output_index} routed-expert rows are not position aligned")
        if num_layers is not None:
            routed_elements = len(raw_routed) // 4
            topk = routed_elements // max(1, position_count * num_layers)
            row_elements = num_layers * max(1, topk)
            if routed_elements == (position_count + 1) * row_elements:
                raw_routed = raw_routed[: position_count * row_elements * 4]
            elif routed_elements != position_count * row_elements:
                raise RuntimeError(f"{prefix}probe output {output_index} routed-expert elements are not model aligned")
        if len(raw_routed) % position_count:
            raise RuntimeError(f"{prefix}probe output {output_index} routed-expert bytes are not position aligned")
        row_bytes = len(raw_routed) // position_count
        routed_rows = [hashlib.sha256(raw_routed[start : start + row_bytes]).hexdigest() for start in range(0, len(raw_routed), row_bytes)]
        summaries.append(
            {
                "text": output["text"],
                "token_ids": token_ids,
                "logprobs": logprobs,
                "routed_experts_sha256": hashlib.sha256(routed.encode()).hexdigest(),
                "routed_experts_row_sha256": routed_rows,
                "input_text_sha256": (hashlib.sha256(input_texts[output_index].encode()).hexdigest() if input_texts is not None else None),
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


def _compare_case_diagnostics(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Collect every semantic and routed-row delta before making a gate decision."""
    if left.get("batch_size") != right.get("batch_size"):
        raise RuntimeError(f"{label}: batch size changed")
    left_outputs = left.get("outputs")
    right_outputs = right.get("outputs")
    if not isinstance(left_outputs, list) or not isinstance(right_outputs, list):
        raise RuntimeError(f"{label}: malformed output summaries")
    if len(left_outputs) != len(right_outputs):
        raise RuntimeError(f"{label}: output count changed")

    max_abs_logprob_diff = 0.0
    routed_row_mismatches = 0
    routed_rows = 0
    issues = []
    per_output = []
    for output_index, (before, after) in enumerate(zip(left_outputs, right_outputs, strict=True)):
        output_issues = []
        if before.get("input_text_sha256") != after.get("input_text_sha256"):
            output_issues.append("input prompt identity changed")
        if before.get("token_ids") != after.get("token_ids"):
            output_issues.append("token IDs changed")
        if before.get("text") != after.get("text"):
            output_issues.append("text changed")
        left_logprobs = before.get("logprobs")
        right_logprobs = after.get("logprobs")
        if not isinstance(left_logprobs, list) or not isinstance(right_logprobs, list):
            output_issues.append("malformed logprobs")
        elif len(left_logprobs) != len(right_logprobs):
            output_issues.append("logprob length changed")
        else:
            for a, b in zip(left_logprobs, right_logprobs, strict=True):
                max_abs_logprob_diff = max(max_abs_logprob_diff, abs(float(a) - float(b)))

        left_rows = before.get("routed_experts_row_sha256")
        right_rows = after.get("routed_experts_row_sha256")
        output_row_mismatches = 0
        if not isinstance(left_rows, list) or not isinstance(right_rows, list):
            output_issues.append("malformed routed-expert rows")
        elif len(left_rows) != len(right_rows):
            output_issues.append("routed-expert row count changed")
        else:
            output_row_mismatches = sum(a != b for a, b in zip(left_rows, right_rows, strict=True))
            routed_row_mismatches += output_row_mismatches
            routed_rows += len(left_rows)
        issues.extend(f"output {output_index} {issue}" for issue in output_issues)
        per_output.append(
            {
                "output_index": output_index,
                "routed_row_mismatches": output_row_mismatches,
                "routed_rows": len(left_rows) if isinstance(left_rows, list) else 0,
                "issues": output_issues,
            }
        )
    return {
        "label": label,
        "max_abs_logprob_diff": max_abs_logprob_diff,
        "routed_row_mismatches": routed_row_mismatches,
        "routed_rows": routed_rows,
        "issues": issues,
        "per_output": per_output,
    }


def _case_repeats(case: Mapping[str, Any], variant: str) -> list[Mapping[str, Any]]:
    repeats = case.get(f"{variant}_repeats")
    if repeats is None:
        return [case[variant]]
    if not isinstance(repeats, list) or not repeats:
        raise RuntimeError(f"zero-effect probe has invalid {variant} repeats")
    return repeats


def validate_initial_zero_effect_probes(
    before_results: Sequence[Mapping[str, Any]],
    after_results: Sequence[Mapping[str, Any]],
    *,
    expected_engine_ranks: Sequence[int],
) -> dict[str, Any]:
    """Require operationally valid probes and characterize nonblocking decode drift."""
    expected_ranks = {int(rank) for rank in expected_engine_ranks}
    if not expected_ranks or len(expected_ranks) != len(expected_engine_ranks):
        raise RuntimeError("zero-effect probe expected engine ranks are empty or duplicated")
    for expected_phase, results in (("before", before_results), ("after", after_results)):
        for item in results:
            if item.get("schema_version") != 1 or item.get("phase") != expected_phase:
                raise RuntimeError(f"zero-effect probe invalid {expected_phase} receipt: schema={item.get('schema_version')} phase={item.get('phase')!r}")
    before_by_engine = {int(item["engine_rank"]): item for item in before_results}
    after_by_engine = {int(item["engine_rank"]): item for item in after_results}
    if len(before_by_engine) != len(before_results) or len(after_by_engine) != len(after_results):
        raise RuntimeError("duplicate engine rank in zero-effect probe")
    if set(before_by_engine) != expected_ranks or set(after_by_engine) != expected_ranks:
        raise RuntimeError(f"zero-effect probe engine coverage mismatch: expected={sorted(expected_ranks)} before={sorted(before_by_engine)} after={sorted(after_by_engine)}")

    comparisons = []
    nonblocking_diagnostics = []
    for engine_rank in sorted(before_by_engine):
        before = before_by_engine[engine_rank]
        after = after_by_engine[engine_rank]
        for batch_size in (1, 8):
            case = f"greedy_batch{batch_size}"
            before_case = before["cases"][case]
            after_case = after["cases"][case]
            baseline_repeats = _case_repeats(before_case, "base")
            post_base_repeats = _case_repeats(after_case, "base")
            post_lora_repeats = _case_repeats(after_case, "lora")
            if len(post_base_repeats) != len(post_lora_repeats):
                raise RuntimeError(f"engine={engine_rank} {case}: base/LoRA repeat count differs")

            baseline_deltas = [
                _compare_case_diagnostics(
                    baseline_repeats[0],
                    repeat,
                    label=f"engine={engine_rank} {case} pre-base repeat={index}",
                )
                for index, repeat in enumerate(baseline_repeats[1:], start=1)
            ]
            base_envelope_logprob = max([1e-5, *(item["max_abs_logprob_diff"] for item in baseline_deltas)])
            base_envelope_routed_rows = max([0, *(item["routed_row_mismatches"] for item in baseline_deltas)])
            base_deltas = [
                _compare_case_diagnostics(
                    baseline_repeats[0],
                    repeat,
                    label=f"engine={engine_rank} {case} pre-base/post-base repeat={index}",
                )
                for index, repeat in enumerate(post_base_repeats)
            ]
            lora_deltas = [
                _compare_case_diagnostics(
                    post_base,
                    post_lora,
                    label=f"engine={engine_rank} {case} post-base/zero-lora repeat={index}",
                )
                for index, (post_base, post_lora) in enumerate(zip(post_base_repeats, post_lora_repeats, strict=True))
            ]
            for delta in [*baseline_deltas, *base_deltas, *lora_deltas]:
                reasons = list(delta["issues"])
                if delta["max_abs_logprob_diff"] > base_envelope_logprob:
                    reasons.append(f"max logprob difference {delta['max_abs_logprob_diff']} exceeded base/base envelope {base_envelope_logprob}")
                if delta["routed_row_mismatches"] > base_envelope_routed_rows:
                    reasons.append(f"routed row mismatches {delta['routed_row_mismatches']} exceeded base/base envelope {base_envelope_routed_rows}")
                if reasons:
                    nonblocking_diagnostics.append({"label": delta["label"], "reasons": reasons, "delta": delta})

            base_diff = max(item["max_abs_logprob_diff"] for item in base_deltas)
            lora_diff = max(item["max_abs_logprob_diff"] for item in lora_deltas)
            comparisons.append(
                {
                    "engine_rank": engine_rank,
                    "batch_size": batch_size,
                    "mode": "greedy_drift_characterization",
                    "base_max_abs_logprob_diff": base_diff,
                    "lora_max_abs_logprob_diff": lora_diff,
                    "base_base_logprob_envelope": base_envelope_logprob,
                    "base_base_routed_row_mismatch_envelope": base_envelope_routed_rows,
                    "baseline_repeat_deltas": baseline_deltas,
                    "post_publication_base_deltas": base_deltas,
                    "paired_zero_lora_deltas": lora_deltas,
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
    result = {
        "status": "PASS",
        "interpretation": "operational_finite_smoke_with_nonblocking_drift_characterization",
        "semantic_or_routing_equivalence_claim": False,
        "engine_count": len(before_by_engine),
        "greedy_comparison_count": sum(item["mode"] == "greedy_drift_characterization" for item in comparisons),
        "stochastic_smoke_count": sum(item["mode"] == "production_stochastic_smoke" for item in comparisons),
        "nonblocking_diagnostic_count": len(nonblocking_diagnostics),
        "nonblocking_diagnostics": nonblocking_diagnostics,
        "comparisons": comparisons,
    }
    return result


def validate_operational_probe_results(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_engine_ranks: Sequence[int],
    phase: str,
) -> dict[str, Any]:
    """Validate full base/LoRA batch-1 and batch-8 smoke coverage."""

    expected_ranks = {int(rank) for rank in expected_engine_ranks}
    if not expected_ranks or len(expected_ranks) != len(expected_engine_ranks):
        raise RuntimeError("operational probe expected engine ranks are empty or duplicated")
    indexed = {int(item["engine_rank"]): item for item in results}
    if len(indexed) != len(results) or set(indexed) != expected_ranks:
        raise RuntimeError(f"operational probe engine coverage mismatch: expected={sorted(expected_ranks)} observed={sorted(indexed)}")

    expected_cases = {
        "greedy_batch1": 1,
        "greedy_batch8": 8,
        "stochastic_batch1": 1,
        "stochastic_batch8": 8,
    }
    request_count = 0
    output_count = 0
    token_count = 0
    for engine_rank in sorted(indexed):
        result = indexed[engine_rank]
        cases = result.get("cases")
        if result.get("schema_version") != 1 or result.get("phase") != phase or not isinstance(cases, Mapping) or set(cases) != set(expected_cases):
            raise RuntimeError(f"engine={engine_rank} operational probe receipt is malformed")
        for case_name, batch_size in expected_cases.items():
            case = cases[case_name]
            if not isinstance(case, Mapping) or set(case) < {"base", "lora"}:
                raise RuntimeError(f"engine={engine_rank} {case_name} omitted base or LoRA evidence")
            for variant in ("base", "lora"):
                summary = case[variant]
                if not isinstance(summary, Mapping):
                    raise RuntimeError(f"engine={engine_rank} {case_name} {variant} has invalid summary")
                outputs = summary.get("outputs")
                if summary.get("batch_size") != batch_size or not isinstance(outputs, list) or len(outputs) != batch_size:
                    raise RuntimeError(f"engine={engine_rank} {case_name} {variant} has invalid batch coverage")
                for output in outputs:
                    ids = output.get("token_ids") if isinstance(output, Mapping) else None
                    logprobs = output.get("logprobs") if isinstance(output, Mapping) else None
                    hashes = (
                        output.get("routed_experts_sha256") if isinstance(output, Mapping) else None,
                        output.get("input_text_sha256") if isinstance(output, Mapping) else None,
                    )
                    if (
                        not isinstance(output, Mapping)
                        or not isinstance(output.get("text"), str)
                        or not isinstance(ids, list)
                        or not ids
                        or any(type(token_id) is not int or token_id < 0 for token_id in ids)
                        or not isinstance(logprobs, list)
                        or len(logprobs) != len(ids)
                        or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in logprobs)
                        or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes)
                    ):
                        raise RuntimeError(f"engine={engine_rank} {case_name} {variant} has invalid output evidence")
                    token_count += len(ids)
                request_count += 1
                output_count += len(outputs)

    evidence_sha256 = hashlib.sha256(json.dumps(results, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1,
        "status": "PASS",
        "phase": phase,
        "engine_count": len(indexed),
        "request_count": request_count,
        "output_count": output_count,
        "token_count": token_count,
        "evidence_sha256": evidence_sha256,
        "interpretation": "finite_structurally_valid_base_and_lora_operational_smoke",
    }


def validate_initial_lora_slot_audits(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_engine_ranks: Sequence[int],
    expected_tp_size: int,
) -> dict[str, Any]:
    """Require complete post-probe slot-integrity evidence from every TP rank."""

    expected_engines = {int(rank) for rank in expected_engine_ranks}
    if not expected_engines or len(expected_engines) != len(expected_engine_ranks):
        raise RuntimeError("LoRA slot audit expected engine ranks are empty or duplicated")
    if type(expected_tp_size) is not int or expected_tp_size <= 0:
        raise RuntimeError("LoRA slot audit expected TP size must be positive")
    by_engine = {int(item["engine_rank"]): item for item in results}
    if len(by_engine) != len(results) or set(by_engine) != expected_engines:
        raise RuntimeError(f"LoRA slot audit engine coverage mismatch: expected={sorted(expected_engines)} observed={sorted(by_engine)}")

    hash_pattern = re.compile(r"[0-9a-f]{64}")
    expected_tp_ranks = list(range(expected_tp_size))
    normalized = []
    for engine_rank in sorted(by_engine):
        result = by_engine[engine_rank]
        if result.get("success") is not True:
            raise RuntimeError(f"engine={engine_rank} LoRA slot audit failed: {result.get('error_message')}")
        reports = result.get("rank_reports")
        if not isinstance(reports, list) or len(reports) != expected_tp_size:
            raise RuntimeError(f"engine={engine_rank} LoRA slot audit report count mismatch")
        observed_tp_ranks = [report.get("tp_rank") for report in reports]
        report_by_rank = {report.get("tp_rank"): report for report in reports}
        if any(type(rank) is not int for rank in observed_tp_ranks) or len(report_by_rank) != len(reports) or sorted(observed_tp_ranks) != expected_tp_ranks:
            raise RuntimeError(f"engine={engine_rank} LoRA slot audit TP coverage mismatch: expected={expected_tp_ranks} observed={observed_tp_ranks}")
        for tp_rank, report in sorted(report_by_rank.items()):
            base_slot = report.get("base_slot_id")
            adapter_slot = report.get("adapter_slot_id")
            hashes = (
                report.get("manifest_sha256"),
                report.get("slot_schema_sha256"),
                report.get("base_schema_sha256"),
                report.get("adapter_b_schema_sha256"),
            )
            positive_counts = (
                report.get("base_buffer_tensors"),
                report.get("base_buffer_elements"),
                report.get("adapter_b_buffer_tensors"),
                report.get("adapter_b_buffer_elements"),
            )
            invalid = (
                report.get("schema_version") != 1
                or report.get("distinct_live_slots") is not True
                or report.get("checksum_verified") is not True
                or type(base_slot) is not int
                or type(adapter_slot) is not int
                or base_slot == adapter_slot
                or any(type(value) is not int or value <= 0 for value in positive_counts)
                or report.get("base_nonzero_elements") != 0
                or report.get("adapter_b_nonzero_elements") != 0
                or any(not isinstance(value, str) or hash_pattern.fullmatch(value) is None for value in hashes)
            )
            if invalid:
                raise RuntimeError(f"engine={engine_rank} tp_rank={tp_rank} invalid LoRA slot audit: {report!r}")
        normalized.append(
            {
                "engine_rank": engine_rank,
                "rank_reports": [report_by_rank[rank] for rank in expected_tp_ranks],
            }
        )
    manifest_hashes = {report["manifest_sha256"] for engine in normalized for report in engine["rank_reports"]}
    if len(manifest_hashes) != 1:
        raise RuntimeError(f"LoRA slot audit manifest disagreement: {sorted(manifest_hashes)}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "interpretation": "inference-independent resident-slot integrity",
        "engine_count": len(normalized),
        "tp_ranks_per_engine": expected_tp_size,
        "rank_report_count": len(normalized) * expected_tp_size,
        "manifest_sha256": next(iter(manifest_hashes)),
        "engines": normalized,
    }


def validate_integrated_base_probes(
    pre_trainer_results: Sequence[Mapping[str, Any]],
    post_trainer_results: Sequence[Mapping[str, Any]],
    *,
    expected_engine_ranks: Sequence[int],
) -> dict[str, Any]:
    """Require stable, finite stock-base probes across trainer load/offload."""
    expected_ranks = {int(rank) for rank in expected_engine_ranks}
    if not expected_ranks or len(expected_ranks) != len(expected_engine_ranks):
        raise RuntimeError("integrated base probe expected engine ranks are empty or duplicated")

    def index_results(label: str, results: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
        for item in results:
            if item.get("schema_version") != 1 or item.get("phase") != "before":
                raise RuntimeError(f"integrated base probe invalid {label} receipt: schema={item.get('schema_version')} phase={item.get('phase')!r}")
        indexed = {int(item["engine_rank"]): item for item in results}
        if len(indexed) != len(results) or set(indexed) != expected_ranks:
            raise RuntimeError(f"integrated base probe engine coverage mismatch for {label}: expected={sorted(expected_ranks)} observed={sorted(indexed)}")
        return indexed

    before_by_engine = index_results("pre_trainer", pre_trainer_results)
    after_by_engine = index_results("post_trainer", post_trainer_results)
    comparisons = []
    for engine_rank in sorted(expected_ranks):
        before = before_by_engine[engine_rank]
        after = after_by_engine[engine_rank]
        for batch_size in (1, 8):
            case = f"greedy_batch{batch_size}"
            max_abs_logprob_diff = _compare_case(
                before["cases"][case]["base"],
                after["cases"][case]["base"],
                label=f"engine={engine_rank} {case} pre-trainer/post-trainer",
            )
            comparisons.append(
                {
                    "engine_rank": engine_rank,
                    "batch_size": batch_size,
                    "max_abs_logprob_diff": max_abs_logprob_diff,
                }
            )
            for stage, receipt in (
                ("pre_trainer", before),
                ("post_trainer", after),
            ):
                smoke = receipt["cases"][f"stochastic_batch{batch_size}"]["base"]
                if smoke.get("batch_size") != batch_size or not smoke.get("outputs"):
                    raise RuntimeError(f"engine={engine_rank} stochastic_batch{batch_size} {stage}: incomplete smoke result")
    return {
        "status": "PASS",
        "engine_count": len(expected_ranks),
        "greedy_comparison_count": len(comparisons),
        "stochastic_smoke_count": len(expected_ranks) * 2 * 2,
        "comparisons": comparisons,
    }
