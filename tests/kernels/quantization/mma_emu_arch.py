# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Which accumulation each architecture implements, and what that means here.

Shared by the MMA-Emu tests and by the report header they print, so the run
says up front which comparison it is about to make rather than leaving it to be
inferred from skip messages.
"""

from collections.abc import Collection

import torch

from vllm.config.mma_emu import MmaEmuConfig

# design_space::Algorithm.
GDFS, COFDA = 1, 2
ALGORITHM_IDS = {"gdfs": GDFS, "cofda": COFDA}


def _device() -> tuple[int | None, str]:
    """(SM version, device name), or (None, reason) if there is no GPU."""
    try:
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            return None, "not a CUDA platform"
        capability = current_platform.get_device_capability()
        if capability is None:
            return None, "no CUDA device"
        return capability[0] * 10 + capability[1], current_platform.get_device_name()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return None, f"could not be determined ({exc})"


SM, DEVICE = _device()

# The accumulation each architecture implements, as the configuration you would
# pass to run it. FDA is CoFDA with the chunk spanning the K tile, so it is
# expressed as CoFDA with the corresponding chunk size.
ARCH_ACCUMULATION: dict[int, dict[str, MmaEmuConfig]] = {
    # Ada
    89: {"fp8": MmaEmuConfig(algorithm="cofda", f_bits=13, chunk_size=16)},
    # Hopper
    90: {"fp8": MmaEmuConfig(algorithm="cofda", f_bits=13, chunk_size=32)},
    # Blackwell, data center
    100: {
        "fp8": MmaEmuConfig(algorithm="cofda", f_bits=25, chunk_size=32),
        "nvfp4": MmaEmuConfig(algorithm="gdfs", f_bits=35, g_bits=6),
    },
    # Blackwell, workstation
    120: {
        "fp8": MmaEmuConfig(algorithm="cofda", f_bits=25, chunk_size=32),
        "nvfp4": MmaEmuConfig(algorithm="gdfs", f_bits=35, g_bits=6),
    },
}

# Architectures whose entry above was measured rather than taken from the
# paper. The header reports which of the two a run is working from.
VERIFIED = {89, 90, 100, 120}

# What each format is compared against, and what it needs to run.
NATIVE_REFERENCE = {
    "fp8": ("cutlass_scaled_mm", "mma_emu_scaled_fp8_mm", 89),
    "nvfp4": ("cutlass_scaled_fp4_mm", "mma_emu_scaled_nvfp4_mm", 100),
}


def describe(config: MmaEmuConfig) -> str:
    """An accumulation configuration, in the terms the paper uses."""
    if config.algorithm == "gdfs":
        return f"GDFS F={config.f_bits} G={config.g_bits} GS={config.group_size}"
    return f"CoFDA F={config.f_bits} CS={config.chunk_size}"


def _built(op: str) -> bool:
    return hasattr(torch.ops._C, op)


def report_lines(formats: Collection[str] | None = None) -> list[str]:
    """The header shown once at the start of a run.

    Args:
        formats: Report only these formats, so a run that collected no NVFP4
            tests does not describe a comparison it will not make. None reports
            every format.
    """
    if SM is None:
        return [f"MMA-Emu: no GPU to test against — {DEVICE}"]

    if not any(_built(op) for _, op, _ in NATIVE_REFERENCE.values()):
        return [
            (
                f"MMA-Emu: SM{SM} ({DEVICE}), but vLLM was not built with "
                f"the emulation kernels — these tests will skip"
            )
        ]

    lines = [f"MMA-Emu: SM{SM} ({DEVICE})"]
    accumulation = ARCH_ACCUMULATION.get(SM)

    for fmt, (native_op, emu_op, min_sm) in NATIVE_REFERENCE.items():
        if formats is not None and fmt not in formats:
            continue
        if not _built(emu_op):
            lines.append(f"  {fmt:<6} not built for SM{SM} — skipped")
        elif min_sm > SM:
            lines.append(f"  {fmt:<6} needs SM{min_sm} — skipped")
        elif accumulation is None or fmt not in accumulation:
            lines.append(f"  {fmt:<6} no accumulation recorded for SM{SM} — skipped")
        else:
            config = accumulation[fmt]
            lines.append(
                f"  {fmt:<6} emulating {describe(config):<24} "
                f"compared bit-exactly against {native_op}"
            )

    if accumulation is None:
        lines.append(
            f"  SM{SM} has no entry in ARCH_ACCUMULATION "
            f"(tests/kernels/quantization/mma_emu_arch.py); add one to test it"
        )
    elif SM not in VERIFIED:
        lines.append(
            f"  SM{SM}'s accumulation comes from the paper rather than a "
            f"measurement taken here"
        )
    return lines
