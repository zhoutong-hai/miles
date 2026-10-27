"""Independent dense-autograd checks for observational output-logit gradients."""

from argparse import Namespace
import ast
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import parallel
from miles.backends.training_utils.loss_hub.score_centering import ScoreCenteringInputs, score_centering_loss
from miles.backends.training_utils.loss_hub.score_centering.sampler_kl import sampler_kl_loss
from miles.backends.training_utils.loss_hub.sfc_logit_audit import component_gradients, component_statistics
from miles.backends.training_utils.loss_hub.sfc_probe import distribution_statistics
from miles.backends.training_utils.loss_hub.sfc_summary import read_frames, summarize_update


def test_actor_changes_are_only_optional_observation_calls():
    path = "miles/backends/megatron_utils/actor.py"
    root = Path(__file__).resolve().parents[2]
    original = subprocess.check_output(["git", "show", "d951cc5765e2abf1f6d07815ff2ed2f92cd49c69:"+path], cwd=root, text=True)
    new = ast.parse((root/path).read_text())

    class RemoveObservationCalls(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module and node.module.endswith((".sfc_observations", ".sfc_probe")) else node

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "configure_rollout":
                return None
            return node

        def visit_If(self, node):
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_fixed_probe" for n in ast.walk(node)):
                return None
            return self.generic_visit(node)

    restored = RemoveObservationCalls().visit(new)
    assert ast.dump(restored) == ast.dump(ast.parse(original))


@pytest.mark.parametrize("mode", ["none", "tis"])
@pytest.mark.parametrize("head_size", [2, 7])
@pytest.mark.parametrize("temperature", [.7, 1.0])
def test_logit_components_match_actual_native_objective(mode, head_size, temperature):
    generator = torch.Generator().manual_seed(315)
    logits = torch.randn(4, 9, dtype=torch.float64, generator=generator, requires_grad=True)
    logp = (logits[:, :7]/temperature).log_softmax(-1)
    logq = torch.randn(4, 7, dtype=torch.float64, generator=generator).log_softmax(-1)
    head_logq, head_ids = logq.topk(head_size, dim=-1)
    # Force one sampled tail token when a tail exists.
    samples = torch.tensor([6, 1, 0, 2])
    sample_q = logq.gather(-1, samples[:, None]).squeeze(-1)
    advantage = torch.tensor([2., -.8, 0., 1.], dtype=torch.float64)
    selected = logp.gather(-1, samples[:, None]).squeeze(-1)
    head_p = logp.gather(-1, head_ids)
    common = dict(train_log_probs=selected, train_head_log_probs=head_p,
                  rollout_log_probs=sample_q, rollout_head_log_probs=head_logq,
                  head_mask=torch.ones_like(head_ids, dtype=torch.bool), advantages=advantage)
    pg_none, _ = score_centering_loss(ScoreCenteringInputs(**common, mode="none"))
    pg, _ = score_centering_loss(ScoreCenteringInputs(**common, mode=mode))
    kl = .01*sampler_kl_loss(head_p, head_logq, common["head_mask"], torch.ones(4, dtype=torch.bool))
    weights = ((selected-sample_q).detach().exp().clamp_max(2) if mode == "tis" else torch.ones_like(selected))
    uncentered = -advantage*weights*selected
    entropy = -(logp.exp()*logp).sum(-1)
    expected = {"unweighted_sfc_pg": pg_none.sum()/21,
                "importance_change": (pg-pg_none).sum()/21,
                "weighted_sampler_kl": kl.sum()/21,
                "sfc_change_from_uncentered": (pg-uncentered).sum()/21,
                "entropy": entropy.sum()}
    actual = component_gradients(logits, head_ids, head_logq, samples, sample_q, advantage,
                                 mode=mode, beta=.01, temperature=temperature, vocab_size=7, denominator=21)
    for name, objective in expected.items():
        dense = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        torch.testing.assert_close(actual[name], dense, atol=1e-12, rtol=1e-10)
    statistics = component_statistics(actual)
    assert not statistics["cosine_defined"][2, 0, 0]  # zero-A PG
    assert statistics["norms"][2, 2] > 0  # independent sampler KL


def test_probe_distribution_excludes_padded_vocab_and_matches_dense():
    logits = torch.tensor([[1., 2., -3., 999.], [-1., 0., 3., 999.]])
    entropy, mass = distribution_statistics(logits, vocab_size=3, topk=2)
    logp = logits[:, :3].log_softmax(-1)
    torch.testing.assert_close(entropy, -(logp.exp()*logp).sum(-1))
    torch.testing.assert_close(mass, logp.exp().topk(2).values.sum(-1))
    empty = distribution_statistics(logits[:0], vocab_size=3, topk=2)
    assert empty[0].shape == empty[1].shape == (0,)


def _write_shard(directory, rank, frame):
    path = directory/f"cp-{rank}.bin"
    payload = io.BytesIO()
    torch.save(frame, payload)
    raw = payload.getvalue()
    path.write_bytes(struct.pack("<Q", len(raw))+raw)
    seal = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frames": 1,
            "valid_tokens": int(frame["data"]["mask"].sum()), "outcome": "NORMAL",
            "capture_seconds": .1, "audit_seconds": .0}
    path.with_suffix(".complete.json").write_text(json.dumps(seal))


def _frame(ids, positions, log_ratios):
    n = len(ids)
    fields = {"sample_id": torch.tensor(ids), "token_position": torch.tensor(positions),
              "mask": torch.ones(n, dtype=torch.bool), "logp": torch.tensor(log_ratios)-5,
              "logq": torch.full((n,), -5.), "weight": torch.ones(n), "advantage": torch.tensor([1., 0.][:n]),
              "entropy": torch.tensor([1., 3.][:n]), "pg_loss": torch.zeros(n), "sampler_kl": torch.full((n,), .2),
              "correction": torch.zeros(n), "p_head_mass": torch.full((n,), .8),
              "q_head_mass": torch.full((n,), .9), "candidate_count": torch.full((n,), 128),
              "sample_in_head": torch.ones(n, dtype=torch.bool)}
    return {"source": "commit", "trainer_version": "1", "rollout": 0, "completed_update": 1, "run": "test",
            "sample_indices": [10, 11, 12], "sampler_versions": [["1"]]*3,
            "samples": [{"theme": "a", "scenario": "b"}]*3, "data": fields}


def test_pooled_summary_counts_original_tokens_and_zero_live_responses(tmp_path):
    _write_shard(tmp_path, 0, _frame([10, 11], [0, 0], [-3., 2.]))
    _write_shard(tmp_path, 1, _frame([11], [1], [0.]))
    output = summarize_update(tmp_path, expected_tokens=3, expected_samples={10, 11, 12}, cp_size=2, mode="none", beta=.01)
    assert output["sfc/ratio_lt05_fraction"] == pytest.approx(1/3)
    assert output["sfc/ratio_gt2_fraction"] == pytest.approx(1/3)
    assert output["sfc/applied_cap_fraction"] == 0
    assert output["sfc/applied_ess"] == 3
    assert output["sfc/entropy_pooled_token_mean"] == pytest.approx(5/3)
    assert output["sfc/entropy_equal_response_mean"] == 1  # (1 + (3+1)/2 + 0)/3
    assert output["sfc/sampler_kl_weighted_token_mean"] == pytest.approx(.002)
    assert (tmp_path/"complete.json").exists()


@pytest.mark.parametrize("failure", ["digest", "duplicate", "missing", "nonfinite"])
def test_paired_summary_rejects_invalid_coverage_or_values(tmp_path, failure):
    frame = _frame([10, 11], [0, 0], [0., 0.])
    if failure == "duplicate":
        frame["data"]["sample_id"][1] = 10
    if failure == "nonfinite":
        frame["data"]["logp"][0] = float("nan")
    _write_shard(tmp_path, 0, frame)
    if failure == "digest":
        with (tmp_path/"cp-0.bin").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        summarize_update(tmp_path, expected_tokens=3 if failure == "missing" else 2,
                         expected_samples={10, 11, 12}, cp_size=1, mode="none", beta=.01)
    assert not (tmp_path/"complete.json").exists()


def test_reader_rejects_truncated_frame(tmp_path):
    p = tmp_path/"broken.bin"
    p.write_bytes(struct.pack("<Q", 100)+b"short")
    with pytest.raises(ValueError, match="Truncated"):
        list(read_frames(p))
