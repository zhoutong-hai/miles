import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.kernels.attention.dsa.backend_observer import BackendObserver, safe_direct_url

sparse = importlib.import_module("miles.kernels.attention.dsa.sparse_attention")


def test_source_metadata_does_not_capture_url_credentials_or_queries():
    value = safe_direct_url({"url": "https://user:secret@example.test/pkg?token=secret#secret",
                             "vcs_info": {"commit_id": "a" * 40, "requested_revision": "secret"}})
    assert value == {"url": "https://example.test/pkg", "commit_id": "a" * 40}
    assert "secret" not in json.dumps(value)


@pytest.mark.parametrize("available,arch,heads,groups,d_v,dim,backend", [
    (True, 9, 16, 1, 512, 576, "flash_mla"),
    (False, 9, 16, 1, 512, 576, "tilelang"),
    (True, 8, 16, 1, 512, 576, "tilelang"),
    (True, 9, 16, 2, 512, 576, "tilelang"),
    (True, 9, 16, 1, 128, 192, "tilelang"),
])
def test_actual_dispatch_table_and_missing_import(monkeypatch, available, arch, heads, groups, d_v, dim, backend):
    monkeypatch.setattr(sparse, "flash_mla_sparse_fwd", (lambda: None) if available else None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (arch, 0))
    q, kv = SimpleNamespace(shape=(1, 4096, heads, dim), device="cuda:0"), SimpleNamespace(shape=(1, 16384, groups, dim))
    config = sparse._default_config(q, kv, d_v)
    assert config.forward_backend == backend
    if arch == 9:
        assert config.backward_stages == 1
    if backend != "flash_mla":
        with pytest.raises(RuntimeError, match="silently fall back"):
            BackendObserver(require_forward="flash_mla").require(config)


def test_receipt_is_private_once_per_actual_phase_not_each_shape(tmp_path, monkeypatch):
    observer = BackendObserver(tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock-H200-NOT-GPU-EVIDENCE")
    q = torch.zeros(1, 4, 16, 576)
    kv = torch.zeros(1, 8, 1, 576)
    indices = torch.zeros(1, 4, 1, 64, dtype=torch.int32)
    config = sparse.SparseAttentionConfig(backward_stages=1)
    for phase in ("forward", "forward", "backward"):
        observer.record(phase, q, kv, indices, 512, config)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2 and all(p.stat().st_mode & 0o777 == 0o600 for p in files)
    packets = [json.loads(p.read_text()) for p in files]
    assert {p["backend"] for p in packets} == {"tilelang", "flash_mla"}
    assert all(p["status"] == "KERNEL_RETURNED_NOT_NUMERIC_ACCEPTANCE" for p in packets)
    assert all(p["packages"]["torch"]["version"] for p in packets)


def test_equal_pid_cross_node_owners_and_actual_extension_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "getpid", lambda: 123)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "MOCK_NOT_GPU_EVIDENCE")
    monkeypatch.setitem(sys.modules, "flash_mla.cuda", SimpleNamespace(__file__="/opt/mock-flashmla.so"))
    build = tmp_path / "mock-build.json"
    build.write_text(json.dumps({"binary_file": "/opt/mock-flashmla.so", "binary_sha256": "0" * 64}))
    owners = [BackendObserver(tmp_path / "receipts", build_receipt=build) for _ in range(2)]
    q, kv, indices = torch.zeros(1,4,16,576), torch.zeros(1,8,1,576), torch.zeros(1,4,1,64,dtype=torch.int32)
    for owner in owners:
        owner.record("forward", q, kv, indices, 512, sparse.SparseAttentionConfig())
    packets = [json.loads(p.read_text()) for p in (tmp_path / "receipts").glob("*.json")]
    assert len(packets) == 2 and len({p["process_identity"] for p in packets}) == 2
    assert all(p["imports"]["flash_mla.cuda"]["file"] == "/opt/mock-flashmla.so" for p in packets)
    assert all(p["binary_provenance"]["binary_sha256"] == "0" * 64 for p in packets)


def test_failed_binary_identity_is_not_cached_as_accepted(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_mla.cuda", SimpleNamespace(__file__="/opt/other.so"))
    build = tmp_path / "wrong-build.json"
    build.write_text(json.dumps({"binary_file": "/opt/selected.so"}))
    owner = BackendObserver(tmp_path / "receipts", build_receipt=build)
    q, kv, indices = torch.zeros(1,4,16,576), torch.zeros(1,8,1,576), torch.zeros(1,4,1,64,dtype=torch.int32)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="differs from selected image"):
            owner.record("forward", q, kv, indices, 512, sparse.SparseAttentionConfig())
        assert owner.packages is None and owner.binary_provenance is None and not owner.seen


@pytest.mark.parametrize("topk,multiple,expected", [(63,64,64),(64,64,64),(65,128,128),(2048,128,2048)])
def test_actual_topk_padding_preserves_invalid_indices(topk, multiple, expected):
    value = torch.arange(topk, dtype=torch.int32)[None]
    value[0,0] = -1
    result = sparse._pad_topk_block(value, multiple)
    assert result.shape[-1] == expected
    assert torch.equal(result[:,:topk], value) and (result[:,topk:] == -1).all()
