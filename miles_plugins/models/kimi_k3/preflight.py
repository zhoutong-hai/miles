"""Fail-closed CUDA toolchain and deferred KDA backward admission."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import torch

from miles_plugins.models.kimi_k3.ops import kda

KDA_BACKWARD_PREFLIGHT_ENV = "MILES_KIMI_K3_KDA_BACKWARD_PREFLIGHT"
_EXPECTED_CUDA_HOME = "/usr/local/cuda"
_HEADS = 3
_HEAD_DIM = 128
_TOKENS = 64


def _cudart_version(cuda_home: Path) -> int:
    header = cuda_home / "include/cuda_runtime_api.h"
    match = re.search(r"^#define\s+CUDART_VERSION\s+(\d+)", header.read_text(), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing CUDART_VERSION in {header}")
    return int(match.group(1))


def _nvcc_release(nvcc: Path) -> str:
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if match is None:
        raise RuntimeError(f"cannot parse nvcc release from {output!r}")
    return match.group(1)


def _expected_release(cudart_version: int) -> str:
    return f"{cudart_version // 1000}.{cudart_version % 1000 // 10}"


def _configure_isolated_cache(rank: int) -> str:
    loaded = sorted(name for name in sys.modules if name == "tilelang" or name.startswith("tilelang."))
    if loaded:
        raise RuntimeError(f"TileLang loaded before KDA backward preflight: {loaded[:5]}")
    cache_dir = tempfile.mkdtemp(prefix=f"miles-kimi-k3-kda-rank{rank}-")
    os.environ["TILELANG_CACHE_DIR"] = cache_dir
    os.environ["TILELANG_TMP_DIR"] = os.path.join(cache_dir, "tmp")
    atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
    return cache_dir


def run_kda_backward_preflight(rank: int) -> dict[str, object]:
    """Compile and execute the exact TP32-local Kimi-K3 KDA backward once."""
    cuda_home_env = os.environ.get("CUDA_HOME")
    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_home_env != _EXPECTED_CUDA_HOME or cuda_path_env != _EXPECTED_CUDA_HOME:
        raise RuntimeError(f"Kimi-K3 KDA preflight requires CUDA_HOME=CUDA_PATH={_EXPECTED_CUDA_HOME}; got CUDA_HOME={cuda_home_env!r}, CUDA_PATH={cuda_path_env!r}")

    cache_dir = _configure_isolated_cache(rank)
    import tilelang.env as tilelang_env
    from tilelang.contrib.nvcc import get_nvcc_compiler

    cuda_home = Path(cuda_home_env).resolve()
    selected_home = Path(tilelang_env.CUDA_HOME).resolve()
    selected_nvcc = Path(get_nvcc_compiler()).resolve()
    expected_nvcc = (cuda_home / "bin/nvcc").resolve()
    if selected_home != cuda_home or selected_nvcc != expected_nvcc:
        raise RuntimeError(f"TileLang selected CUDA_HOME={selected_home}, nvcc={selected_nvcc}; expected CUDA_HOME={cuda_home}, nvcc={expected_nvcc}")

    cudart_version = _cudart_version(cuda_home)
    nvcc_release = _nvcc_release(selected_nvcc)
    if nvcc_release != _expected_release(cudart_version):
        raise RuntimeError(f"CUDA compiler/header mismatch: nvcc release {nvcc_release}, CUDART_VERSION {cudart_version}")
    if torch.version.cuda != nvcc_release:
        raise RuntimeError(f"Torch CUDA {torch.version.cuda} does not match nvcc release {nvcc_release}")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA H200":
        raise RuntimeError("Kimi-K3 KDA backward preflight requires one visible NVIDIA H200")

    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(31031)

        def activation() -> torch.Tensor:
            return torch.randn(
                1,
                _TOKENS,
                _HEADS,
                _HEAD_DIM,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )

        q, k, v, g = (activation() for _ in range(4))
        beta = torch.rand(
            1,
            _TOKENS,
            _HEADS,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        a_log = torch.zeros(_HEADS, device=device, dtype=torch.float32, requires_grad=True)
        dt_bias = torch.zeros(_HEADS * _HEAD_DIM, device=device, dtype=torch.float32, requires_grad=True)
        output = kda(q, k, v, g, beta, a_log, dt_bias, -5.0)
        if output.shape != (1, _TOKENS, _HEADS, _HEAD_DIM) or not torch.isfinite(output).all():
            raise RuntimeError(f"non-finite or invalid KDA preflight output shape: {tuple(output.shape)}")
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        gradients = {
            "q": q.grad,
            "k": k.grad,
            "v": v.grad,
            "g": g.grad,
            "beta": beta.grad,
            "A_log": a_log.grad,
            "dt_bias": dt_bias.grad,
        }
        invalid = [name for name, gradient in gradients.items() if gradient is None or not torch.isfinite(gradient).all()]
        if invalid:
            raise RuntimeError(f"KDA backward preflight produced invalid gradients: {invalid}")

    torch.cuda.empty_cache()
    return {
        "result": "PASS",
        "rank": rank,
        "cuda_home": str(cuda_home),
        "nvcc": str(selected_nvcc),
        "nvcc_release": nvcc_release,
        "cudart_version": cudart_version,
        "shape": [1, _TOKENS, _HEADS, _HEAD_DIM, _HEAD_DIM],
        "finite_gradients": sorted(gradients),
        "tilelang_cache_dir": cache_dir,
    }
