"""Fail candidate build on wrong closure/imports; no GPU readiness claim."""
import importlib.metadata
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from flash_mla import flash_mla_sparse_fwd
import tilelang
import tvm_ffi
import flash_mla.cuda as flash_mla_binary

expected = {"tilelang": "0.1.14", "apache-tvm-ffi": "0.1.11", "torch-c-dlpack-ext": "0.1.5", "z3-solver": "4.15.4.0"}
assert sys.version_info[:2] == (3, 12)
assert torch.__version__ == "2.11.0+cu130"
for name, version in expected.items():
    assert importlib.metadata.version(name) == version, name
assert importlib.metadata.version("flash-mla").startswith("1.0.0+b7643bd")
assert "d_v" in inspect.signature(flash_mla_sparse_fwd).parameters
assert hasattr(tilelang.PassConfigKey, "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE")
binary = Path(flash_mla_binary.__file__)
with binary.open("rb") as stream:
    binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
build_receipt = {
    "status": "IMAGE_IMPORTS_ONLY_NOT_GPU_ACCEPTANCE",
    "torch": torch.__version__, "flashmla_version": importlib.metadata.version("flash-mla"),
    "flashmla_source": "b7643bd54521f563b839b98289b5cd048c062ba2",
    "cutlass_source": "147f5673d0c1c3dcf66f78d677fd647e4a020219",
    "binary_file": str(binary), "binary_sha256": binary_sha256,
    "wheel_hashes": Path("/opt/e010-kernel-wheels/SHA256SUMS").read_text().strip(),
}
Path("/opt/e010-kernel-wheels/build-receipt.json").write_text(json.dumps(build_receipt, indent=2) + "\n")
print(json.dumps({"imports": "PASS_NOT_GPU_QUALIFICATION", "torch": torch.__version__, "tilelang_file": tilelang.__file__, "tvm_ffi_file": tvm_ffi.__file__, "packages": expected}, sort_keys=True))
