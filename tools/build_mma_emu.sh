#!/usr/bin/env bash
set -euo pipefail

# Build the CUDA/C++ extensions in place.
#
# Use this instead of `pip install -e .`. The `dependencies` field in
# pyproject.toml is dynamic: setup.py reads requirements/cuda.txt, which pins
# torch==2.13.0 without a local version and pulls the [cu13] extras. Any build
# that resolves dependencies therefore replaces a CUDA 12 torch with the CUDA 13
# wheel from PyPI. Building the extensions directly skips resolution entirely.
#
# TORCH_CUDA_ARCH_LIST is pinned to the local GPU because the default sweeps
# every supported architecture, which takes far longer for no benefit here.
#
# Overridable via env: TORCH_CUDA_ARCH_LIST, MAX_JOBS, CCACHE_DIR,
# CMAKE_BUILD_TYPE, VLLM_TARGET_DEVICE, VLLM_PYTHON.
#
# Usage:
#   tools/build_mma_emu.sh            # incremental
#   tools/build_mma_emu.sh --clean    # discard CMake cache first

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) < 32 ? $(nproc) : 32 ))}"
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "No interpreter at $PYTHON." >&2
  echo "Create it with: uv venv --python 3.12 .venv" >&2
  exit 1
fi

# torch's cpp_extension raises on a CUDA major mismatch between nvcc and the
# torch build (a minor mismatch is only a warning). Fail here with a readable
# message rather than partway through the build.
"$PYTHON" - <<'EOF'
import re, subprocess, sys
import torch
from torch.utils.cpp_extension import CUDA_HOME

if CUDA_HOME is None:
    sys.exit("CUDA_HOME is not set and no CUDA toolkit was found.")

out = subprocess.check_output([f"{CUDA_HOME}/bin/nvcc", "--version"], text=True)
m = re.search(r"release (\d+)\.(\d+)", out)
if m is None:
    sys.exit(f"Could not parse a version from {CUDA_HOME}/bin/nvcc --version.")

nvcc = (int(m[1]), int(m[2]))
tc = tuple(int(x) for x in torch.version.cuda.split("."))
print(f"torch {torch.__version__}  |  nvcc {nvcc[0]}.{nvcc[1]}  |  arch {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'no GPU'}")

if nvcc[0] != tc[0]:
    sys.exit(
        f"CUDA major mismatch: nvcc {nvcc[0]}.{nvcc[1]} vs torch {torch.version.cuda}.\n"
        f"torch.utils.cpp_extension rejects this. Point CUDA_HOME at a CUDA "
        f"{tc[0]}.x toolkit, or install a torch built for CUDA {nvcc[0]}.x."
    )
EOF

if [[ "${1:-}" == "--clean" ]]; then
  echo "Removing build/ ..."
  rm -rf "$REPO_ROOT/build"
fi

echo "Building: arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS"
exec "$PYTHON" setup.py build_ext --inplace -j "$MAX_JOBS"
