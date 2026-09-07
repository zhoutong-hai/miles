from copy import deepcopy

import pytest

from miles.backends.sglang_utils.lora_zero_effect_probe import (
    build_probe_payload,
    summarize_probe_response,
    validate_initial_zero_effect_probes,
    validate_integrated_base_probes,
)


def _response(batch_size):
    return [
        {
            "text": f"answer-{index}",
            "meta_info": {
                "output_token_logprobs": [[-0.5, 10 + index], [-0.25, 20 + index]],
                "routed_experts": f"routing-{index}",
            },
        }
        for index in range(batch_size)
    ]


def _engine_receipt(engine_rank, phase):
    cases = {}
    for mode in ("greedy", "stochastic"):
        for batch_size in (1, 8):
            summary = summarize_probe_response(_response(batch_size), expected_batch_size=batch_size)
            cases[f"{mode}_batch{batch_size}"] = {"base": summary}
            if phase == "after":
                cases[f"{mode}_batch{batch_size}"]["lora"] = deepcopy(summary)
    return {
        "schema_version": 1,
        "engine_rank": engine_rank,
        "phase": phase,
        "cases": cases,
    }


def test_probe_payload_exercises_production_stochastic_batch_and_lora():
    payload = build_probe_payload(batch_size=8, lora_path="miles_lora", stochastic=True)
    assert payload["lora_path"] == "miles_lora"
    assert payload["return_logprob"] is True
    assert payload["return_routed_experts"] is True
    assert all(item["temperature"] == 1.0 for item in payload["sampling_params"])
    assert len(set(payload["text"])) == 8


def test_probe_payload_builds_greedy_equality_request():
    payload = build_probe_payload(batch_size=1, lora_path=None, stochastic=False)
    assert "lora_path" not in payload
    assert payload["sampling_params"][0]["temperature"] == 0.0


def test_probe_response_rejects_nonfinite_logprob():
    response = _response(1)
    response[0]["meta_info"]["output_token_logprobs"][0][0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        summarize_probe_response(response, expected_batch_size=1)


def test_probe_response_rejects_null_output_logprob_with_context():
    response = _response(1)
    response[0]["meta_info"]["output_token_logprobs"][0][0] = None
    with pytest.raises(
        RuntimeError,
        match=r"engine_rank=12 phase=before mode=greedy batch_size=8 variant=base: .*null output logprob",
    ):
        summarize_probe_response(
            response,
            expected_batch_size=1,
            context="engine_rank=12 phase=before mode=greedy batch_size=8 variant=base",
        )


def test_probe_response_rejects_non_numeric_logprob():
    response = _response(1)
    response[0]["meta_info"]["output_token_logprobs"][0][0] = "not-a-number"
    with pytest.raises(RuntimeError, match="non-numeric"):
        summarize_probe_response(response, expected_batch_size=1)


def test_initial_zero_effect_probe_accepts_exact_four_engine_match():
    expected_ranks = [0, 4, 8, 12]
    before = [_engine_receipt(rank, "before") for rank in expected_ranks]
    after = [_engine_receipt(rank, "after") for rank in expected_ranks]
    result = validate_initial_zero_effect_probes(before, after, expected_engine_ranks=expected_ranks)
    assert result["status"] == "PASS"
    assert result["engine_count"] == 4
    assert result["greedy_comparison_count"] == 8
    assert result["stochastic_smoke_count"] == 24


def test_initial_zero_effect_probe_rejects_lora_token_change():
    before = [_engine_receipt(0, "before")]
    after = [_engine_receipt(0, "after")]
    after[0]["cases"]["greedy_batch8"]["lora"]["outputs"][0]["token_ids"][0] += 1
    with pytest.raises(RuntimeError, match="token IDs changed"):
        validate_initial_zero_effect_probes(before, after, expected_engine_ranks=[0])


def test_initial_zero_effect_probe_accepts_routing_within_measured_base_repeat_envelope():
    before = [_engine_receipt(0, "before")]
    after = [_engine_receipt(0, "after")]
    for batch_size in (1, 8):
        case = f"greedy_batch{batch_size}"
        baseline = before[0]["cases"][case]["base"]
        repeat = deepcopy(baseline)
        repeat["outputs"][0]["routed_experts_row_sha256"][0] = "natural-base-variance"
        before[0]["cases"][case]["base_repeats"] = [baseline, repeat]
        post_base = after[0]["cases"][case]["base"]
        post_lora = after[0]["cases"][case]["lora"]
        post_base["outputs"][0]["routed_experts_row_sha256"][0] = "post-base-variance"
        post_lora["outputs"][0]["routed_experts_row_sha256"][0] = "post-lora-variance"

    result = validate_initial_zero_effect_probes(before, after, expected_engine_ranks=[0])
    assert result["status"] == "PASS"
    assert all(
        item["base_base_routed_row_mismatch_envelope"] == 1
        for item in result["comparisons"]
        if item["mode"] == "greedy_equality"
    )


def test_initial_zero_effect_probe_rejects_routing_beyond_measured_base_repeat_envelope():
    before = [_engine_receipt(0, "before")]
    after = [_engine_receipt(0, "after")]
    after[0]["cases"]["greedy_batch8"]["lora"]["outputs"][0][
        "routed_experts_row_sha256"
    ][0] = "unexpected-lora-route"
    with pytest.raises(RuntimeError, match="routed row mismatches 1 exceeded base/base envelope 0"):
        validate_initial_zero_effect_probes(before, after, expected_engine_ranks=[0])


def test_initial_zero_effect_probe_rejects_missing_engine():
    before = [_engine_receipt(rank, "before") for rank in range(2)]
    after = [_engine_receipt(0, "after")]
    with pytest.raises(RuntimeError, match="engine coverage mismatch"):
        validate_initial_zero_effect_probes(before, after, expected_engine_ranks=[0, 1])


def test_initial_zero_effect_probe_rejects_equal_but_incomplete_engine_set():
    before = [_engine_receipt(0, "before")]
    after = [_engine_receipt(0, "after")]
    with pytest.raises(RuntimeError, match="engine coverage mismatch"):
        validate_initial_zero_effect_probes(before, after, expected_engine_ranks=[0, 4, 8, 12])


def test_initial_zero_effect_probe_rejects_invalid_phase():
    before = [_engine_receipt(0, "after")]
    after = [_engine_receipt(0, "after")]
    with pytest.raises(RuntimeError, match="invalid before receipt"):
        validate_initial_zero_effect_probes(before, after, expected_engine_ranks=[0])


def test_integrated_base_probe_accepts_exact_four_engine_match():
    expected_ranks = [0, 4, 8, 12]
    before = [_engine_receipt(rank, "before") for rank in expected_ranks]
    after = deepcopy(before)
    result = validate_integrated_base_probes(before, after, expected_engine_ranks=expected_ranks)
    assert result["status"] == "PASS"
    assert result["engine_count"] == 4
    assert result["greedy_comparison_count"] == 8
    assert result["stochastic_smoke_count"] == 16


def test_integrated_base_probe_rejects_post_trainer_token_change():
    before = [_engine_receipt(0, "before")]
    after = deepcopy(before)
    after[0]["cases"]["greedy_batch8"]["base"]["outputs"][0]["token_ids"][0] += 1
    with pytest.raises(RuntimeError, match="token IDs changed"):
        validate_integrated_base_probes(before, after, expected_engine_ranks=[0])


def test_integrated_base_probe_rejects_incomplete_engine_set():
    before = [_engine_receipt(0, "before")]
    after = [_engine_receipt(0, "before")]
    with pytest.raises(RuntimeError, match="engine coverage mismatch"):
        validate_integrated_base_probes(before, after, expected_engine_ranks=[0, 4])
