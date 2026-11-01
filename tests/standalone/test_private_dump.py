"""Private output integrity, bounded backpressure, immutability and actor ordering."""

import ast
import asyncio
from argparse import Namespace
import base64
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
import sys

import numpy as np
import pytest
import torch

from miles.ray.rollout import private_dump
from miles.utils.types import Sample


def sample():
    return Sample(index=7, tokens=[1, 2, 3], response="private fixture", response_length=1,
                  reward={"score": .4, "scenario_score": .6},
                  metadata={"verifier_diagnostics": {"reasoning": "private judgment"}},
                  rollout_log_probs=[-.2],
                  rollout_routed_experts=np.arange(8000, dtype=np.int32).reshape(100, 80)[:, ::2])


def test_completion_last_immutable_private_fields_and_numpy_dtype(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original_write = private_dump._write_part

    def held_write(path, value):
        if path.name.endswith(".incomplete"):
            entered.set()
            assert release.wait(5)
        return original_write(path, value)

    monkeypatch.setattr(private_dump, "_write_part", held_write)
    value = sample()
    expected = value.rollout_routed_experts.copy()
    path = tmp_path / "point.pt"
    writer = private_dump.PrivateDumpWriter(chunk_bytes=65536)
    writer.submit(path, [value], rollout_id=9, evaluation=False, metadata={"cursor": 10})
    try:
        assert entered.wait(5) and not path.exists()
        value.rollout_routed_experts[:] = -1
        value.metadata["verifier_diagnostics"]["reasoning"] = "changed"
        value.rollout_log_probs[0] = -9
    finally:
        release.set()
        writer.close()
    payload = private_dump.load_rollout_payload(path)
    result = payload["samples"][0]
    assert result["metadata"]["verifier_diagnostics"]["reasoning"] == "private judgment"
    assert result["rollout_log_probs"] == [-.2] and payload["metadata"] == {"cursor": 10}
    assert result["rollout_routed_experts"].dtype == np.int32
    assert np.array_equal(result["rollout_routed_experts"], expected)
    assert Sample.from_dict(result).index == 7


def test_chunk_bytes_and_queue_backpressure(tmp_path, monkeypatch):
    writer = private_dump.PrivateDumpWriter(chunk_bytes=65536, queue_chunks=2)
    real_put = writer._put
    depths, chunk_sizes = [], []

    def observed_put(queue, item):
        if isinstance(item[1], torch.Tensor):
            chunk_sizes.append(item[1].numel() * item[1].element_size())
        real_put(queue, item)
        depths.append(queue.qsize())

    monkeypatch.setattr(writer, "_put", observed_put)
    value = sample()
    value.rollout_routed_experts = np.arange(1_000_000, dtype=np.int32)
    path = tmp_path / "point.pt"
    writer.submit(path, [value], rollout_id=0, evaluation=False, metadata={})
    writer.close()
    assert max(depths) <= 2 and max(chunk_sizes) <= 65536 and len(chunk_sizes) > 50
    assert np.array_equal(private_dump.load_rollout_payload(path)["samples"][0]["rollout_routed_experts"],
                          value.rollout_routed_experts)
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(tmp_path / "another.pt", [], rollout_id=1, evaluation=False, metadata={})


def test_writer_failure_propagates_and_partial_is_unaccepted(tmp_path, monkeypatch):
    def failed_write(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(private_dump, "_write_part", failed_write)
    writer = private_dump.PrivateDumpWriter(chunk_bytes=65536)
    path = tmp_path / "point.pt"
    with pytest.raises(OSError, match="simulated"):
        try:
            writer.submit(path, [sample()], rollout_id=0, evaluation=True, metadata={})
            writer.drain()
        finally:
            writer.close()
    assert not path.exists()


def test_integrity_and_immutable_destination(tmp_path):
    writer = private_dump.PrivateDumpWriter(chunk_bytes=65536)
    path = tmp_path / "point.pt"
    writer.submit(path, [sample()], rollout_id=0, evaluation=False, metadata={})
    writer.drain()
    with pytest.raises(FileExistsError):
        writer.submit(path, [], rollout_id=0, evaluation=False, metadata={})
    writer.close()
    part = next(path.with_name(path.name + ".parts").glob("array-*.pt"))
    with part.open("ab") as stream:
        stream.write(b"damage")
    with pytest.raises(ValueError, match="integrity"):
        private_dump.load_rollout_payload(path)


def test_legacy_and_empty_tensor_roundtrip(tmp_path):
    old = tmp_path / "legacy.pt"
    torch.save({"rollout_id": 1, "metadata": {}, "samples": [sample().to_dict()]}, old)
    assert private_dump.load_rollout_payload(old)["samples"][0]["index"] == 7
    value = sample()
    value.metadata = {"empty": torch.empty((0, 3), dtype=torch.bfloat16)}
    writer = private_dump.PrivateDumpWriter(chunk_bytes=65536)
    path = tmp_path / "new.pt"
    writer.submit(path, [value], rollout_id=1, evaluation=True, metadata={})
    writer.close()
    empty = private_dump.load_rollout_payload(path)["samples"][0]["metadata"]["empty"]
    assert empty.dtype == torch.bfloat16 and empty.shape == (0, 3)


def test_oversize_metadata_is_not_silently_dropped(tmp_path):
    writer = private_dump.PrivateDumpWriter(chunk_bytes=1024)
    path = tmp_path / "point.pt"
    with pytest.raises(ValueError, match="metadata record"):
        try:
            writer.submit(path, [], rollout_id=0, evaluation=True, metadata={"private": "x" * 2048})
        finally:
            writer.close()
    assert not path.exists()


def method(name, globals):
    source = Path(__file__).parents[2] / "miles/ray/rollout/rollout_executor.py"
    tree = ast.parse(source.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "RolloutExecutor")
    function = next(x for x in node.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), globals)
    return globals[name]


def test_eval_metrics_precede_private_snapshot_submission():
    events = []
    result = SimpleNamespace(data={}, metrics={})
    namespace = dict(asyncio=asyncio, timer=lambda _: nullcontext(),
                     call_rollout_fn=lambda *a, **k: result,
                     log_eval_rollout_data=lambda *a: events.append("numeric") or {"eval/step": 0},
                     save_debug_rollout_data=lambda *a, **k: events.append("snapshot"))
    owner = SimpleNamespace(args=SimpleNamespace(debug_train_only=False, eval_uses_snapshots=False),
                            use_legacy_rollout_v1=True, eval_generate_rollout=None, data_source=None,
                            _private_dump_writer=object(), _metric_checker=None)
    assert asyncio.run(method("eval", namespace)(owner, 0)) == {"eval/step": 0}
    assert events == ["numeric", "snapshot"]


@pytest.mark.parametrize("evaluation,replay,requested", [(False, True, True), (True, True, False),
                                                         (False, False, False), (True, False, False)])
def test_real_legacy_request_keeps_training_r3_only(evaluation, replay, requested):
    source = Path(__file__).parents[2] / "miles/rollout/sglang_rollout.py"
    tree = ast.parse(source.read_text())
    node = next(x for x in ast.walk(tree) if isinstance(x, ast.If)
                and any(isinstance(c, ast.Constant) and c.value == "return_routed_experts"
                        for child in x.body for c in ast.walk(child)))
    values = dict(args=SimpleNamespace(use_rollout_routing_replay=replay), evaluation=evaluation,
                  payload={"return_logprob": True})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), values)
    assert values["payload"].get("return_routed_experts", False) is requested
    assert values["payload"]["return_logprob"] is True


@pytest.mark.parametrize("evaluation", [False, True])
def test_actual_generation_keeps_q_versions_and_native_training_routes(evaluation):
    source = Path(__file__).parents[2] / "miles/rollout/sglang_rollout.py"
    tree = ast.parse(source.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.AsyncFunctionDef) and x.name == "generate")
    routes = np.arange(8, dtype=np.int32).reshape(2, 2, 2)

    async def post(url, payload, headers):
        assert payload["return_logprob"] is True
        info = dict(output_token_logprobs=[[-.25, 13, None]], prompt_tokens=2,
                    weight_version="1", finish_reason={"type": "stop"})
        if not evaluation:
            assert payload["return_routed_experts"] is True
            info["routed_experts"] = base64.b64encode(routes.tobytes()).decode("ascii")
        else:
            assert "return_routed_experts" not in payload
        return dict(text="fixture", meta_info=info)

    state = SimpleNamespace(processor=None, tokenizer=SimpleNamespace(encode=lambda *a, **k: [11, 12]))
    values = dict(Namespace=Namespace, Sample=Sample, Any=Any, GenerateState=lambda args: state,
                  should_return_sampling_mask=lambda *a, **k: False, lora_rollout_enabled=lambda args: False,
                  compute_routing_headers=lambda *a: {}, post=post, np=np, pybase64=base64)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), values)
    args = Namespace(ci_test=False, sglang_router_ip="127.0.0.1", sglang_router_port=1,
                     use_rollout_routing_replay=True, num_layers=2, sglang_speculative_algorithm=None)
    value = Sample(prompt="fixture", metadata={})
    got = asyncio.run(values["generate"](args, value, {"max_new_tokens": 1}, evaluation=evaluation))
    assert got.tokens == [11, 12, 13] and got.rollout_log_probs == [-.25]
    assert got.weight_versions[0].spans[0].version == "1"
    if evaluation:
        assert got.rollout_routed_experts is None
    else:
        assert np.array_equal(got.rollout_routed_experts, routes)


def test_actor_dispose_flushes_real_writer_before_tracking_finish(tmp_path, monkeypatch):
    events = []
    writer = private_dump.PrivateDumpWriter(chunk_bytes=65536)
    path = tmp_path / "point.pt"
    writer.submit(path, [sample()], rollout_id=40, evaluation=True, metadata={})

    def finish():
        assert path.exists() and writer._closed
        private_dump.load_rollout_payload(path)
        events.append("tracking-finished-after-dump")

    monkeypatch.setitem(sys.modules, "miles.utils.tracking_utils.tracking", SimpleNamespace(finish_tracking=finish))
    owner = SimpleNamespace(_private_dump_writer=writer, data_source=SimpleNamespace(close=lambda: None),
                            args=None, _metric_checker=None, eval_generate_rollout=None)
    values = dict(event_analyzer=SimpleNamespace(run_analysis_from_args=lambda _: None),
                  CheckpointEvalFn=type("CheckpointEvalFn", (), {}))
    method("dispose", values)(owner)
    assert events == ["tracking-finished-after-dump"]


@pytest.mark.parametrize("evaluation,replay", [(False, True), (True, True), (False, False), (True, False)])
def test_selected_single_turn_request_and_response_preserve_q_and_training_r3(evaluation, replay):
    root = Path(__file__).parents[2]
    values = dict(Any=Any, Sample=Sample, deepcopy=deepcopy, np=np, pybase64=base64,
                  should_return_sampling_mask=lambda *a, **k: False,
                  lora_rollout_enabled=lambda args: False,
                  GenerateFnInput=SimpleNamespace, GenerateFnOutput=SimpleNamespace)
    endpoint = root / "miles/rollout/generate_utils/generate_endpoint_utils.py"
    names = {"compute_request_payload", "update_sample_from_response", "get_routed_experts_from_response",
             "get_indexer_topk_from_response", "_decode_topk_buffer"}
    functions = [node for node in ast.parse(endpoint.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(endpoint), "exec"), values)
    routes = np.arange(8, dtype=np.int32).reshape(2, 2, 2)

    async def post(url, payload, headers):
        assert payload["return_logprob"] is True
        assert payload["return_routed_experts"] is (replay and not evaluation)
        info = dict(output_token_logprobs=[[-.25, 13, None]], prompt_tokens=2,
                    weight_version="1", finish_reason={"type": "stop"})
        if payload["return_routed_experts"]:
            info["routed_experts"] = base64.b64encode(routes.tobytes()).decode("ascii")
        return dict(text="fixture", meta_info=info)

    values.update(post=post, compute_prompt_ids_from_sample=lambda *a: [11, 12],
                  compute_routing_headers=lambda *a: None)
    source = root / "miles/rollout/generate_hub/single_turn.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), values)
    args = Namespace(rollout_max_response_len=512, rollout_max_context_len=65536,
                     use_rollout_routing_replay=replay, use_rollout_indexer_replay=False,
                     sglang_router_ip="127.0.0.1", sglang_router_port=1,
                     num_layers=2, sglang_speculative_algorithm=None)
    sample = Sample(prompt="fixture", metadata={})
    result = asyncio.run(values["generate"](SimpleNamespace(
        args=args, sample=sample, state=None, sampling_params={"max_new_tokens": 1}, evaluation=evaluation)))
    assert result.samples.tokens == [11, 12, 13]
    assert result.samples.rollout_log_probs == [-.25]
    assert result.samples.weight_versions[0].spans[0].version == "1"
    if replay and not evaluation:
        assert np.array_equal(result.samples.rollout_routed_experts, routes)
    else:
        assert result.samples.rollout_routed_experts is None
