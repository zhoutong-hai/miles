"""Fixed-input/index old-vs-new attention timing; not whole-training causality.

Run only after GPU admission. The legacy modules are unchanged test reference
code; GLM production imports only miles.kernels.attention.dsa.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from miles.kernels.attention.dsa import sparse_attention
from miles.kernels.attention.dsa.sparse_attention import _default_config
from miles_plugins.models.glm5.ops.sparse_mla import SparseMLA


def milliseconds(fn, iterations):
    fn()  # compile/warm up outside timing
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def fixed_indices(query_length, kv_length, topk):
    # Exact once-only top-k without duplicated KV positions. CP rank0, causal.
    rows = []
    for chunk in torch.arange(query_length, device="cuda").split(256):
        scores = torch.rand(len(chunk), kv_length, device="cuda")
        scores.masked_fill_(torch.arange(kv_length, device="cuda")[None] > chunk[:, None], -torch.inf)
        selected_scores, selected = scores.topk(topk, dim=-1)
        rows.append(selected.int().masked_fill(selected_scores == -torch.inf, -1))
    return torch.cat(rows)[:, None].contiguous()


def benchmark(kv_length, iterations):
    torch.manual_seed(20261013)
    query_length = kv_length // 4
    q = torch.randn(query_length, 16, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(kv_length, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = fixed_indices(query_length, kv_length, 2048)
    grad = torch.randn(query_length, 16, 512, device="cuda", dtype=torch.bfloat16)
    selected = _default_config(q[None], kv[None], 512)
    if selected.forward_backend != "flash_mla":
        raise RuntimeError("H200/16-head fast path did not select FlashMLA; do not report a fast-path benchmark")
    results, outputs = {}, {}
    for name in ("legacy_tilelang", "candidate_flashmla_tilelang"):
        qg, kvg = q.clone().requires_grad_(), kv.clone().requires_grad_()

        def forward():
            if name == "legacy_tilelang":
                return SparseMLA.apply(qg, kvg, indices, 576**-.5)[0]
            return sparse_attention(qg[None], kvg[None], indices[None], 576**-.5, d_v=512)[0]

        def forward_backward():
            qg.grad = kvg.grad = None
            forward().backward(grad)

        with torch.no_grad():
            forward_ms = milliseconds(forward, iterations)
        total_ms = milliseconds(forward_backward, iterations)
        forward_backward()
        output = forward().detach()
        assert all(torch.isfinite(value).all() for value in (output, qg.grad, kvg.grad))
        outputs[name] = (output, qg.grad.detach().clone(), kvg.grad.detach().clone())
        results[name] = {"forward_ms": forward_ms, "forward_backward_ms": total_ms}
    old, new = outputs.values()
    agreement = [float(((a.float() - b.float()).square().sum() / (a.float().square().sum() + 1e-12)).sqrt()) for a, b in zip(old, new)]
    assert max(agreement) < .03, agreement
    return {"q_shape": list(q.shape), "kv_shape": list(kv.shape), "topk": 2048,
            "identical_inputs_and_indices": True, "indexer_selection_excluded": True,
            "tuning": asdict(selected), "timing": results, "relative_l2_out_dq_dkv": agreement}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-lengths", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or any(n < 2048 or n % 4 for n in args.kv_lengths):
        parser.error("positive iterations and CP4-compatible KV lengths >=2048 required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Selected qualification is H200/Hopper only")
    packet = {"kind": "FIXED_INDEX_KERNEL_TIMING_NOT_WHOLE_RUN", "rows": [benchmark(n, args.iterations) for n in args.kv_lengths]}
    with args.output.open("x") as stream:
        json.dump(packet, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
