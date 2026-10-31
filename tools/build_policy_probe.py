"""Seal64 native K3 prompt-end positions; no generation or chat templating."""

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(input_path, tokenizer_path, output_path, *, tokenizer_revision):
    rows = pq.read_table(input_path, columns=["prompt", "extra_info"]).to_pylist()
    by_theme = defaultdict(deque)
    for index, row in enumerate(rows):
        by_theme[row["extra_info"]["theme"]].append(index)
    selected = []
    while len(selected) < 64:
        before = len(selected)
        for theme in sorted(by_theme):
            if by_theme[theme] and len(selected) < 64:
                selected.append(by_theme[theme].popleft())
        if len(selected) == before:
            raise ValueError("Need64 original native prompts")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    target = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    tokens, identities = [], []
    for index in selected:
        prompt = tokenizer.encode(rows[index]["prompt"], add_special_tokens=False)
        if not prompt or len(prompt) >= 65536:
            raise ValueError("Probe requires a nonempty, untruncated native K3 prefix")
        # This dummy target is never scored. Its preceding causal logit row is
        # the first response distribution after the complete original prompt.
        tokens.append(torch.tensor(prompt + [target], dtype=torch.long))
        identities.append((index, len(prompt)))
    output = Path(output_path)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Fixed probes are immutable")
    packet = {
        "tokens": tokens,
        "identities": identities,
        "tokenizer_revision": tokenizer_revision,
        "input_sha256": sha256(input_path),
        "selection": "sorted_theme_round_robin_native_source_order",
        "scope": "64 first-response distributions on sealed native prompts; no sampling or update",
    }
    torch.save(packet, output)
    output.chmod(0o600)
    summary = {k: v for k, v in packet.items() if k != "tokens"}
    summary.update(
        {
            "probe_sha256": sha256(output),
            "positions": 64,
            "tokenizer_files": {p.name: sha256(p) for p in Path(tokenizer_path).iterdir() if p.is_file()},
            "prefix_token_lengths": [len(t) - 1 for t in tokens],
        }
    )
    with output.with_suffix(".json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    output.with_suffix(".json").chmod(0o600)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.input, args.tokenizer, args.output, tokenizer_revision=args.tokenizer_revision)
    print(json.dumps({key: result[key] for key in ("probe_sha256", "positions", "tokenizer_revision")}))
