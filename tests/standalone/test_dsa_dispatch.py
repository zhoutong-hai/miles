import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from miles.kernels.attention.dsa.backend_observer import BackendObserver

sparse = importlib.import_module("miles.kernels.attention.dsa.sparse_attention")


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


@pytest.mark.parametrize("topk,multiple,expected", [(63,64,64),(64,64,64),(65,128,128),(2048,128,2048)])
def test_actual_topk_padding_preserves_invalid_indices(topk, multiple, expected):
    value = torch.arange(topk, dtype=torch.int32)[None]
    value[0,0] = -1
    result = sparse._pad_topk_block(value, multiple)
    assert result.shape[-1] == expected
    assert torch.equal(result[:,:topk], value) and (result[:,topk:] == -1).all()
