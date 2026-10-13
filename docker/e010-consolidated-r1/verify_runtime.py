"""Fail candidate build on wrong closure/imports; no GPU readiness claim."""
import importlib.metadata
import inspect
import json
import sys

import torch
from flash_mla import flash_mla_sparse_fwd
import tilelang
import tvm_ffi

expected = {"tilelang": "0.1.14", "apache-tvm-ffi": "0.1.11", "torch-c-dlpack-ext": "0.1.5", "z3-solver": "4.15.4.0"}
assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == "2.11.0+cu130"
for name, version in expected.items():
    assert importlib.metadata.version(name) == version, name
assert importlib.metadata.version("flash-mla").startswith("1.0.0+b7643bd")
assert "d_v" in inspect.signature(flash_mla_sparse_fwd).parameters
assert hasattr(tilelang.PassConfigKey, "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE")
print(json.dumps({"imports": "PASS_NOT_GPU_QUALIFICATION", "torch": torch.__version__, "tilelang_file": tilelang.__file__, "tvm_ffi_file": tvm_ffi.__file__, "packages": expected}, sort_keys=True))
