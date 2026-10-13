# Candidate only. Build from Miles repo root; do not use without a recorded
# resulting immutable digest, import checks and GPU acceptance.
FROM radixark/miles:v0.1.0@sha256:45ae7833f97a01369dbd8f336784830a87d94587bc010c7b8065291cc15cd7ca
COPY docker/e010-consolidated-r1/requirements.txt /opt/e010-kernel-requirements.txt
RUN python -c 'import sys,torch; assert sys.version_info[:2] == (3,12); assert torch.__version__ == "2.11.0+cu130"' && \
    pip install --only-binary=:all: --no-deps --require-hashes -r /opt/e010-kernel-requirements.txt
# Do NOT use the published Torch2.13 wheel in the qualified Torch2.11 image.
# Build exact same FlashMLA source against this image's own Torch/CUDA ABI.
RUN git init /opt/e010-flash-mla && cd /opt/e010-flash-mla && \
    git remote add origin https://github.com/deepseek-ai/FlashMLA.git && \
    git fetch --depth 1 origin b7643bd54521f563b839b98289b5cd048c062ba2 && \
    git checkout --detach b7643bd54521f563b839b98289b5cd048c062ba2 && \
    git submodule update --init csrc/cutlass && \
    test "$(git -C csrc/cutlass rev-parse HEAD)" = 147f5673d0c1c3dcf66f78d677fd647e4a020219 && \
    FLASH_MLA_DISABLE_SM100=1 FLASH_MLA_DISABLE_SM90=0 MAX_JOBS=8 pip wheel --no-deps --no-build-isolation . -w /opt/e010-kernel-wheels && \
    pip install --no-deps /opt/e010-kernel-wheels/flash_mla-*.whl && \
    sha256sum /opt/e010-kernel-wheels/flash_mla-*.whl > /opt/e010-kernel-wheels/SHA256SUMS
COPY docker/e010-consolidated-r1/verify_runtime.py /opt/e010-verify-runtime.py
RUN python /opt/e010-verify-runtime.py
# The operator separately stages the exact committed Miles/Arrakis/Megatron/
# Bridge/SGLang sources. This layer does not upgrade their lifecycle or serving.
