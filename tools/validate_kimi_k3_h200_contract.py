#!/usr/bin/env python3
"""Fail-closed static contract check for the e006 Kimi-K3 H200 runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


MILES_BASE = "e13758d9b40a3d164f87651c7ac513b844070b6a"
MILES_ORPHAN_ROOT = "93a7fa4f262f6f241aa3f00c42e066c9d022ee25"
MILES_UPSTREAM_TREE_SOURCE = "7e575f0549db9664277a26b7862a181c4faa0e5e"
SGLANG_COMMIT = "2380121e9b16fd3ce778bcc2b4717414b9c4d8a5"
SGLANG_K3_MERGE = "abddb1c7e9d61ddddeaf016d885c2f20aab426e8"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_text(path: Path, fragments: tuple[str, ...]) -> None:
    text = path.read_text()
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"{path} is missing required contract fragments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--miles", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sglang", type=Path, required=True)
    args = parser.parse_args()

    miles = args.miles.resolve()
    sglang = args.sglang.resolve()

    if git(miles, "rev-parse", f"{MILES_ORPHAN_ROOT}^{{tree}}") != git(
        miles, "rev-parse", f"{MILES_UPSTREAM_TREE_SOURCE}^{{tree}}"
    ):
        raise RuntimeError("K3 orphan-root tree does not equal the recorded upstream Miles tree")
    if git(miles, "rev-list", "--count", f"{MILES_ORPHAN_ROOT}..{MILES_BASE}") != "20":
        raise RuntimeError("K3 integration base is not the recorded linear 20-commit series")
    subprocess.check_call(["git", "-C", str(miles), "merge-base", "--is-ancestor", MILES_BASE, "HEAD"])

    if git(sglang, "rev-parse", "HEAD") != SGLANG_COMMIT:
        raise RuntimeError("SGLang worktree is not at the pinned public commit")
    subprocess.check_call(
        ["git", "-C", str(sglang), "merge-base", "--is-ancestor", SGLANG_K3_MERGE, SGLANG_COMMIT]
    )

    require_text(
        miles / "scripts/run_kimi_k3_lora.py",
        (
            "_K3_H200_FULL_NODES = 16",
            "_K3_H200_GPUS_PER_NODE = 8",
            "return 8 if self.model_variant == \"4layer\" else 32",
            "return 8 if self.model_variant == \"4layer\" else 128",
            "return self.total_actor_gpus // model_parallel",
            "return 8 if self.model_variant == \"4layer\" else 32",
            "--pipeline-model-parallel-size {args.pipeline_parallel_size}",
            "--expert-model-parallel-size {args.expert_parallel_size}",
            "--sglang-decode-attention-backend flashmla",
            "--sglang-enable-symm-mem",
            "--sglang-mamba-radix-cache-strategy extra_buffer_lazy",
            '"SGLANG_K3_ATTN_RES_MODE": "jit"',
            '"--no-gradient-accumulation-fusion "',
        ),
    )
    require_text(
        miles / "scripts/models/kimi-k3-lora-full-sglang.yaml",
        ("num_gpus_per_engine: 32", "worker_type: regular", "num_gpus: 128"),
    )
    require_text(
        miles / "miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py",
        (
            "_lora_ipc_live_tensors",
            "_prepare_lora_ipc_payload_for_reuse",
            "_refresh_flattened_lora_ipc_payload",
            "metadata changed across publications",
            "Do not collect them here",
        ),
    )
    require_text(
        miles / "miles/backends/megatron_utils/lora_utils.py",
        (
            "_lora_grad_sum_group",
            'gradient = getattr(parameter, "main_grad", None)',
            "dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)",
        ),
    )
    require_text(
        miles / "tools/audit_kimi_k3_dcp_capacity.py",
        (
            '"tensor_parallel": 32',
            '"expert_parallel": 128',
            '"ranks_completed") != 128',
            "rank_resident_dcp_bytes",
            "rank_runtime_extra_peak_bytes",
        ),
    )

    require_text(
        sglang / "python/sglang/srt/server_args.py",
        (
            "mem_fraction_static:",
            "decode_attention_backend:",
            "cuda_graph_backend_prefill:",
            "cuda_graph_bs_decode:",
            "enable_symm_mem:",
            "moe_runner_backend:",
            "mamba_radix_cache_strategy:",
        ),
    )
    require_text(
        sglang / "python/sglang/srt/managers/io_struct.py",
        ("class LoadLoRAAdapterFromTensorsReqInput", "expected_checksums:"),
    )
    require_text(
        sglang / "python/sglang/srt/managers/tp_worker.py",
        ("def load_lora_adapter_from_tensors", 'recv_req.load_format == "flattened_bucket"'),
    )
    require_text(
        sglang / "python/sglang/srt/lora/lora_manager.py",
        ("def load_lora_adapter_from_tensors", "experts_shared_outer_loras"),
    )
    require_text(
        sglang / "docs/src/snippets/configs/moonshotai/kimi-k3.jsx",
        (
            'match: { hw: "h200", pdMode: "unified", strategy: "high-throughput" }',
            '"--tp-size 32"',
            '"--ep-size 32"',
            '"--moe-runner-backend marlin"',
            '"--decode-attention-backend flashmla"',
            '"SGLANG_K3_ATTN_RES_MODE=jit"',
        ),
    )

    result = {
        "miles_head": git(miles, "rev-parse", "HEAD"),
        "miles_base": MILES_BASE,
        "miles_base_series_commits": 20,
        "miles_orphan_tree": git(miles, "rev-parse", f"{MILES_ORPHAN_ROOT}^{{tree}}"),
        "sglang_commit": SGLANG_COMMIT,
        "sglang_k3_merge_ancestor": SGLANG_K3_MERGE,
        "trainer": "TP32/PP1/CP1/EP128/ETP1/DP4",
        "rollout": "4x TP32/EP32",
        "status": "STATIC_COMPATIBILITY_OK",
    }
    print("KIMI_K3_H200_CONTRACT_OK " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
