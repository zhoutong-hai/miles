from __future__ import annotations

import pytest

from scripts import run_kimi_k3_lora


def _full(**kwargs):
    defaults = {
        "model_variant": "full",
        "num_nodes": 16,
        "num_gpus_per_node": 8,
        "pipeline_parallel_size": 1,
        "context_parallel_size": 1,
        "tp_size_override": 32,
        "ep_size_override": 128,
        "rollout_tp_size": 32,
        "rollout_ep_size": 32,
    }
    defaults.update(kwargs)
    return run_kimi_k3_lora.ScriptArgs(**defaults)


def test_e006_full_model_parallel_derivation():
    args = _full()
    model_parallel = args.tensor_parallel_size * args.context_parallel_size * args.pipeline_parallel_size
    data_parallel = (args.num_nodes * args.num_gpus_per_node) // model_parallel
    assert args.tensor_parallel_size == 32
    assert args.pipeline_parallel_size == 1
    assert args.context_parallel_size == 1
    assert args.expert_parallel_size == 128
    assert data_parallel == 4
    assert args.expert_parallel_size == args.tensor_parallel_size * args.context_parallel_size * data_parallel
    assert args.rollout_num_gpus_per_engine == 32
    assert args.rollout_expert_parallel_size == 32


@pytest.mark.parametrize(
    "changes",
    [
        {"num_nodes": 8},
        {"num_gpus_per_node": 4},
        {"pipeline_parallel_size": 2},
        {"context_parallel_size": 2},
        {"tp_size_override": 16},
        {"ep_size_override": 64},
        {"rollout_tp_size": 16},
        {"rollout_ep_size": 16},
    ],
)
def test_e006_rejects_non_selected_layout(changes):
    with pytest.raises((ValueError, NotImplementedError)):
        _full(**changes)


def test_four_layer_variant_is_unaffected():
    args = run_kimi_k3_lora.ScriptArgs(model_variant="4layer", num_nodes=1, num_gpus_per_node=8)
    assert args.tensor_parallel_size == 8
    assert args.expert_parallel_size == 8


def test_e006_full_model_uses_hopper_decode_backend(monkeypatch):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_kimi_k3_lora.U, "execute_train", capture)
    run_kimi_k3_lora._execute_train(_full())
    train_args = captured["train_args"]
    assert "--sglang-decode-attention-backend flashmla" in train_args
    assert "--sglang-decode-attention-backend trtllm_mla" not in train_args
    assert "--sglang-lora-no-cpu-backup" in train_args
    assert captured["extra_env_vars"]["CUDA_HOME"] == "/usr/local/cuda"
    assert captured["extra_env_vars"]["CUDA_PATH"] == "/usr/local/cuda"
    assert captured["extra_env_vars"]["MILES_KIMI_K3_KDA_BACKWARD_PREFLIGHT"] == "1"


def test_four_layer_variant_does_not_force_e006_cuda_preflight(monkeypatch):
    captured = {}

    monkeypatch.setattr(run_kimi_k3_lora.U, "execute_train", lambda **kwargs: captured.update(kwargs))
    run_kimi_k3_lora._execute_train(
        run_kimi_k3_lora.ScriptArgs(model_variant="4layer", num_nodes=1, num_gpus_per_node=8)
    )

    assert "CUDA_HOME" not in captured["extra_env_vars"]
    assert "CUDA_PATH" not in captured["extra_env_vars"]
    assert "MILES_KIMI_K3_KDA_BACKWARD_PREFLIGHT" not in captured["extra_env_vars"]
