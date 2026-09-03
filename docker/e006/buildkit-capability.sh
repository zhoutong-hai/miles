#!/usr/bin/env bash
set -euo pipefail

buildkit_version=v0.24.0
buildkit_sha256=af8064eca16077b4d6937745988ba2d2dfa439540874cdcd918318315f3ba1d3
build_root=/nvme/e006-buildkit-capability-v0.24.0
archive="${build_root}/buildkit-${buildkit_version}.linux-amd64.tar.gz"
bin_dir="${build_root}/dist/bin"
state_dir="${build_root}/state"
socket="${build_root}/buildkitd.sock"
log="${build_root}/buildkitd.log"
output="${build_root}/smoke.oci.tar"

mkdir -p "${build_root}/dist" "${state_dir}"
if [ ! -f "${archive}" ]; then
  curl --fail --location --retry 3 --output "${archive}" \
    "https://github.com/moby/buildkit/releases/download/${buildkit_version}/buildkit-${buildkit_version}.linux-amd64.tar.gz"
fi
test "$(sha256sum "${archive}" | awk '{print $1}')" = "${buildkit_sha256}"
if [ ! -x "${bin_dir}/buildkitd" ]; then
  tar -xzf "${archive}" -C "${build_root}/dist"
fi

rm -f "${socket}" "${output}"
"${bin_dir}/buildkitd" \
  --addr "unix://${socket}" \
  --root "${state_dir}" \
  --oci-worker=true \
  --oci-worker-snapshotter=native \
  --containerd-worker=false \
  >"${log}" 2>&1 &
buildkitd_pid=$!
cleanup() {
  kill "${buildkitd_pid}" 2>/dev/null || true
  wait "${buildkitd_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if [ -S "${socket}" ]; then
    break
  fi
  if ! kill -0 "${buildkitd_pid}" 2>/dev/null; then
    sed -n '1,200p' "${log}" >&2
    exit 1
  fi
  sleep 1
done
test -S "${socket}"

"${bin_dir}/buildctl" --addr "unix://${socket}" debug workers
"${bin_dir}/buildctl" --addr "unix://${socket}" build \
  --frontend dockerfile.v0 \
  --local context=docker/e006/buildkit-smoke \
  --local dockerfile=docker/e006/buildkit-smoke \
  --output "type=oci,dest=${output}"
test -s "${output}"
sha256sum "${output}"
echo E006_BUILDKIT_CAPABILITY_PASS
