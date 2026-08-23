#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The GPU a run executed on, recorded with the result.

A cross-architecture comparison is only worth reading if the two results came
off different architectures, so the result says which one it ran on rather than
leaving it to be inferred from the directory the file landed in.

    python common/device.py
"""

from __future__ import annotations

import json
import sys

# The name each SM version goes by. Two Blackwells share a tag because they
# share an accumulation: see tests/kernels/quantization/mma_emu_arch.py.
ARCH_TAGS = {89: "ada", 90: "hopper", 100: "blackwell", 120: "blackwell"}


def _absent(reason: str) -> dict:
    """No GPU, said in the shape a result expects."""
    return {"name": reason, "capability": None, "sm": None, "arch": None}


def info() -> dict:
    """`{name, capability, sm, arch}` for the visible device."""
    try:
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            return _absent("not a CUDA platform")
        capability = current_platform.get_device_capability()
        if capability is None:
            return _absent("no CUDA device")
        name = current_platform.get_device_name()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return _absent(f"could not be determined ({exc})")

    major, minor = capability[0], capability[1]
    sm = major * 10 + minor
    return {
        "name": name,
        "capability": f"{major}.{minor}",
        "sm": sm,
        "arch": ARCH_TAGS.get(sm, f"sm{sm}"),
    }


if __name__ == "__main__":
    print(json.dumps(info()))
    sys.exit(0)
