"""Retained-API candidate transport and native loss parity; no model/Ray/GPU."""

import ast
import copy
import math
import subprocess
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean, slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils import parallel
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    merge_rollout_topk_logprobs_field,
    pad_rollout_topk_logprobs,
    validate_rollout_topk_logprobs_sample,
)
from miles.utils.score_centering import validate_score_centering_args
from miles.utils.types import Sample

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def reset_parallel():
    previous = parallel._parallel_state
    parallel.set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, group=None)))
    yield
    parallel.set_parallel_state(previous)


def arguments(**overrides):
    args = dict(
        loss_type="score_centering", rollout_top_logprobs_num=3,
        score_centering_is="none", score_centering_tis_clip=2.0,
        score_centering_mis_low=0.5, score_centering_mis_high=5.0,
        sampler_kl_coef=0.03, rollout_top_p=1.0, rollout_top_k=-1,
        rollout_temperature=1.0, advantage_estimator="grpo", qkv_format="thd",
        allgather_cp=False, true_on_policy_mode=False, vocab_size=5,
        log_probs_chunk_size=2, entropy_coef=0.0, observe_training_entropy=True,
        use_kl_loss=False,
    )
    return Namespace(**{**args, **overrides})


def _load_functions(path, names, namespace):
    """Exercise actual functions while omitting unavailable Ray/SGLang imports."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("overrides", [
    {"use_tis": True}, {"rollout_top_p": 0.9}, {"rollout_top_k": 8},
    {"sampler_kl_coef": -1}, {"sampler_kl_coef": math.nan},
    {"rollout_top_logprobs_num": 0}, {"score_centering_is": "mis"},
    {"qkv_format": "bshd"}, {"partial_rollout": True},
])
def test_unsupported_compositions_fail(overrides):
    with pytest.raises(ValueError):
        validate_score_centering_args(arguments(**overrides))


def test_default_off_and_actual_sampler_contract(monkeypatch):
    validate_score_centering_args(Namespace(loss_type="policy_loss"))
    validate_score_centering_args(arguments())
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    with pytest.raises(ValueError, match="actual sampling"):
        validate_score_centering_args(arguments())


def sample_with_candidates():
    sample = Sample(tokens=[4, 1, 0], response_length=2, rollout_log_probs=[math.log(0.5), math.log(0.4)])
    append_rollout_topk_logprobs(sample, {
        "output_token_logprobs": [[math.log(0.5), 1], [math.log(0.4), 0]],
        "output_top_logprobs": [
            [[math.log(0.5), 1], [math.log(0.3), 3]],
            [[math.log(0.4), 0], [math.log(0.2), 2]],
        ],
    }, 3)
    return sample


def test_transport_validation_and_sample_round_trip():
    sample = sample_with_candidates()
    validate_rollout_topk_logprobs_sample(sample, 3)
    restored = Sample.from_dict(sample.to_dict())
    np.testing.assert_array_equal(restored.rollout_topk_token_ids, sample.rollout_topk_token_ids)
    validate_rollout_topk_logprobs_sample(restored, 3)
    restored.strip_last_output_tokens(1, SimpleNamespace(decode=lambda _: ""))
    validate_rollout_topk_logprobs_sample(restored, 3)
    restored.reset_for_retry()
    assert restored.rollout_topk_log_probs is restored.rollout_topk_token_ids is None


@pytest.mark.parametrize("kind", ["missing", "duplicate", "different_q", "mass", "negative_id", "no_head"])
def test_bad_generation_candidates_fail(kind):
    sample = sample_with_candidates()
    if kind == "missing":
        sample.rollout_topk_log_probs = None
    elif kind == "duplicate":
        sample.rollout_topk_token_ids[0, 1] = 1
    elif kind == "different_q":
        sample.rollout_log_probs[0] -= 0.1
    elif kind == "mass":
        sample.rollout_topk_log_probs[0, :2] = math.log(0.9)
    elif kind == "negative_id":
        sample.rollout_topk_token_ids[0, 0] = -2
    else:
        sample.rollout_topk_token_ids[0] = -1
        sample.rollout_topk_log_probs[0] = -np.inf
    with pytest.raises((ValueError, AssertionError)):
        validate_rollout_topk_logprobs_sample(sample, 3)


def test_request_and_evaluation_boundaries():
    args = arguments(rollout_max_response_len=512, rollout_max_context_len=None,
                     use_rollout_routing_replay=False, use_rollout_indexer_replay=False)
    ns = _load_functions("miles/rollout/generate_utils/generate_endpoint_utils.py", {"compute_request_payload"}, {
        "Any": object, "Sample": Sample, "deepcopy": copy.deepcopy,
        "lora_rollout_enabled": lambda _: False,
        "configure_rollout_topk_logprobs_request": configure_rollout_topk_logprobs_request,
    })
    for evaluation in (False, True):
        payload, status = ns["compute_request_payload"](args, [1, 2], {"temperature": 0 if evaluation else 1}, evaluation=evaluation)
        assert status is None
        assert payload.get("top_logprobs_num") == (None if evaluation else 3)
    with pytest.raises(ValueError, match="unfiltered"):
        configure_rollout_topk_logprobs_request(args, {"sampling_params": {"top_p": 0.9}})
    with pytest.raises(ValueError, match="generation-time"):
        append_rollout_topk_logprobs(Sample(), {"output_token_logprobs": [[-1, 2]]}, 3)


def test_tool_and_merge_gap_padding():
    first, second = sample_with_candidates(), sample_with_candidates()
    merged_ids = merge_rollout_topk_logprobs_field(first, second, "rollout_topk_token_ids", 2)
    merged_logps = merge_rollout_topk_logprobs_field(first, second, "rollout_topk_log_probs", 2)
    assert (merged_ids[2:4] == -1).all() and np.isneginf(merged_logps[2:4]).all()
    pad_rollout_topk_logprobs(first, 2)
    assert first.rollout_topk_token_ids.shape == (4, 3)
    second.rollout_topk_log_probs = None
    with pytest.raises(ValueError):
        merge_rollout_topk_logprobs_field(first, second, "rollout_topk_log_probs", 1)


def test_native_conversion_and_reordered_microbatch_keep_candidate_axes():
    samples = [sample_with_candidates(), sample_with_candidates()]
    ns = _load_functions("miles/ray/rollout/train_data_conversion.py", {
        "convert_samples_to_train_data", "_compute_rollout_mask_sums",
    }, {"Any": object, "Sample": Sample, "validate_rollout_topk_logprobs_sample": validate_rollout_topk_logprobs_sample,
        "_post_process_rewards": lambda *a, **kw: ([0, 1], [0, 1])})
    data = ns["convert_samples_to_train_data"](Namespace(rollout_top_logprobs_num=3, use_dynamic_global_batch_size=False), samples, {}, None, None)
    assert data["rollout_topk_token_ids"][0].shape == (2, 3)
    ns = _load_functions("miles/backends/training_utils/data.py", {"DataIterator"}, {"RolloutBatch": dict, "Sequence": list})
    batch = ns["DataIterator"](data, micro_batch_indices=[[1, 0]]).get_next(["rollout_topk_token_ids", "rollout_topk_log_probs"])
    np.testing.assert_array_equal(batch["rollout_topk_log_probs"][0], samples[1].rollout_topk_log_probs)


def test_glm_kernels_router_sampler_and_lifecycle_are_unchanged():
    # These paths carry E010's accepted GLM/H200/frozen-gate/sampler evidence.
    # The loss, candidate-conversion and CP-width changes have separate tests.
    paths = ["miles_plugins/models/glm5", "miles/kernels/attention/dsa",
             "miles/backends/megatron_utils/router_load_observer.py",
             "miles/backends/megatron_utils/actor.py", "miles/rollout/adaptive_sampling",
             "miles/ray/placement_group.py", "miles/ray/actor_group.py",
             "miles/utils/tracking_utils", "miles/backends/megatron_utils/update_weight"]
    subprocess.run(["git", "diff", "--exit-code", "d951cc5765e2abf1f6d07815ff2ed2f92cd49c69", "--", *paths],
                   cwd=ROOT, check=True, capture_output=True)


@pytest.mark.parametrize("mode", ["none", "tis"])
@pytest.mark.parametrize("coefficient", [0.0, 0.03])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("cp_size", [1, 4])
def test_native_loss_cp_partition_matches_dense_gradient(mode, coefficient, temperature, cp_size):
    args = arguments(score_centering_is=mode, sampler_kl_coef=coefficient, rollout_temperature=temperature)
    generator = torch.Generator().manual_seed(35)
    totals, responses = [7, 10, 3], [4, 5, 0]
    parts = [torch.randn(n, 8, generator=generator, requires_grad=True) for n in totals]
    tokens = [torch.randint(5, (n,), generator=generator) for n in totals]
    masks = [torch.tensor([1., 0, 1, 1]), torch.tensor([1., 1, 1, 0, 1]), torch.empty(0)]
    advantages = [torch.tensor([1., 1, -0.5, 0]), torch.tensor([-0.7, 0, 1.2, 0, -0.2]), torch.empty(0)]
    qs = [torch.rand(n, 5, generator=generator).softmax(-1) for n in responses]
    heads = [q.topk(3, dim=-1) for q in qs]
    q_sample = [q.log().gather(-1, t[-n:, None]).squeeze(-1) if n else torch.empty(0) for q, t, n in zip(qs, tokens, responses)]
    loss = parts[0].sum() * 0
    logs = []
    from miles.backends.training_utils.cp_utils import slice_with_cp
    for rank in range(cp_size):
        parallel.set_parallel_state(SimpleNamespace(cp=SimpleNamespace(size=cp_size, rank=rank), tp=SimpleNamespace(size=1, group=None)))
        batch = dict(unconcat_tokens=tokens, total_lengths=totals, response_lengths=responses, loss_masks=masks,
                     rollout_topk_token_ids=[h.indices for h in heads], rollout_topk_log_probs=[h.values.log() for h in heads],
                     rollout_log_probs=[slice_log_prob_with_cp(q, t, n) for q, t, n in zip(q_sample, totals, responses)],
                     advantages=[slice_log_prob_with_cp(a, t, n) for a, t, n in zip(advantages, totals, responses)])
        local = torch.cat([slice_with_cp(p, 0) for p in parts]).unsqueeze(0) if cp_size > 1 else torch.cat(parts).unsqueeze(0)
        reduce = get_sum_of_sample_mean(totals, responses, masks, calculate_per_token_loss=True)
        actual, metrics = score_centering_loss_function(args, batch, local, reduce)
        loss = loss + actual
        logs.append(metrics)
    count = sum(m.sum() for m in masks)
    loss = loss / count
    expected = sum(part.sum() * 0 for part in parts)
    for part, token, n, mask, advantage, sampled_logq, head in zip(parts, tokens, responses, masks, advantages, q_sample, heads):
        if not n:
            continue
        p_log = (part[-n-1:-1, :5] / temperature).log_softmax(-1)
        p = p_log.detach().exp()
        rho = (1 - head.values.sum(-1)) / (1 - p.gather(-1, head.indices).sum(-1)).clamp_min(1e-6)
        qhat = p * rho[:, None]
        qhat.scatter_(-1, head.indices, head.values)
        weight = (p / qhat).clamp_max(2) if mode == "tis" else torch.ones_like(p)
        chosen = token[-n:]
        sampled_logp = p_log.gather(-1, chosen[:, None]).squeeze(-1)
        # The sampled weight uses actual q even when the sample is outside K;
        # only the centering integral reconstructs the omitted tail.
        sampled_weight = (sampled_logp.detach() - sampled_logq).exp().clamp_max(2) if mode == "tis" else 1
        pg = -advantage * (sampled_weight * sampled_logp
                           - (qhat * weight * p_log).sum(-1))
        kl = (qhat * (qhat.log() - p_log)).sum(-1)
        expected = expected + ((pg + coefficient * kl) * mask).sum() / count
    torch.testing.assert_close(torch.autograd.grad(loss, parts, retain_graph=True), torch.autograd.grad(expected, parts), atol=2e-6, rtol=2e-5)
    for row in logs:
        torch.testing.assert_close(row["loss"], row["pg_loss"] + row.get("sampler_kl_weighted_loss", 0))
        if mode == "none":
            assert row["sc_tis_clip_fraction"] == 0
        assert row["sc_tis_floor_fraction"] == 0
    if coefficient:
        torch.testing.assert_close(sum(row["sampler_kl_coef"] for row in logs) / count, torch.tensor(coefficient))
