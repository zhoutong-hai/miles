"""Bounded EP32 routed-factor manifest for the e007 receiver diagnostic.

The wire retains its original tensor name and a sender-side full SHA-256. Each
of the 32 serving ranks independently verifies only the 28 experts it owns.
This format is diagnostic-only until four-cell and corruption gates pass.
"""

from __future__ import annotations

import hashlib
import re

import torch


EXPERTS = 896
EP_SIZE = 32
LOCAL_EXPERTS = EXPERTS // EP_SIZE
PREFIX = "e007-ep32-sha256-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ROUTED = re.compile(
    r"language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\."
    r"(?:w1\.lora_B|w3\.lora_B|w2\.lora_A)\.weight\Z"
)


def is_routed_inner(name: str) -> bool:
    return bool(_ROUTED.fullmatch(name))


def sha256_owned_cpu(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("routed shard digest requires owned contiguous CPU tensor")
    return hashlib.sha256(
        memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
    ).hexdigest()


def encode_sender_shards(name: str, snapshot: torch.Tensor, full_digest: str) -> str:
    if not is_routed_inner(name) or not _HEX.fullmatch(full_digest):
        raise ValueError("invalid K3 routed sender manifest identity")
    if (snapshot.device.type != "cpu" or not snapshot.is_contiguous()
            or snapshot.ndim != 3 or snapshot.shape[0] != EXPERTS):
        raise ValueError("K3 routed sender manifest needs contiguous 896-expert CPU wire")
    shards = [sha256_owned_cpu(snapshot.narrow(0, rank * LOCAL_EXPERTS, LOCAL_EXPERTS))
              for rank in range(EP_SIZE)]
    return f"{PREFIX}:{full_digest}:{','.join(shards)}"


def expected_owned_digest(name: str, manifest_value: str, ep_rank: int) -> str:
    if not is_routed_inner(name) or not 0 <= ep_rank < EP_SIZE:
        raise ValueError("invalid K3 routed receiver ownership")
    pieces = manifest_value.split(":")
    if len(pieces) != 3 or pieces[0] != PREFIX or not _HEX.fullmatch(pieces[1]):
        raise ValueError("invalid K3 routed manifest header")
    shards = pieces[2].split(",")
    if len(shards) != EP_SIZE or any(not _HEX.fullmatch(value) for value in shards):
        raise ValueError("K3 routed manifest lacks all 32 shard digests")
    return shards[ep_rank]


def receiver_receipt(ep_rank: int, digest: str) -> str:
    if not 0 <= ep_rank < EP_SIZE or not _HEX.fullmatch(digest):
        raise ValueError("invalid K3 routed receiver receipt")
    return f"{PREFIX}:{ep_rank}:{digest}"


def verify_receiver_receipt(name: str, receipt: str, expected: str,
                            ep_rank: int) -> bool:
    if not is_routed_inner(name):
        return bool(_HEX.fullmatch(receipt) and receipt == expected)
    return (receipt == receiver_receipt(ep_rank,
            expected_owned_digest(name, expected, ep_rank)))
