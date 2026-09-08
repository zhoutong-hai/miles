from __future__ import annotations

import sys

import pytest

from miles_plugins.models.kimi_k3 import preflight


def test_expected_release_uses_cuda_runtime_encoding():
    assert preflight._expected_release(13000) == "13.0"
    assert preflight._expected_release(12080) == "12.8"


def test_preflight_rejects_incoherent_cuda_environment(monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("CUDA_PATH", "/different/toolkit")

    with pytest.raises(RuntimeError, match="requires CUDA_HOME=CUDA_PATH"):
        preflight.run_kda_backward_preflight(rank=0)


def test_cache_configuration_fails_if_tilelang_was_already_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "tilelang", object())

    with pytest.raises(RuntimeError, match="TileLang loaded before"):
        preflight._configure_isolated_cache(rank=7)
