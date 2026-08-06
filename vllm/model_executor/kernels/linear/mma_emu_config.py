# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared configuration for the MMA accumulation emulation kernels.

The FP8 and NVFP4 kernels read the same VLLM_MMA_EMU_* environment variables
and answer the same question in ``can_implement``, so both go through here.
"""

import torch

import vllm.envs as envs

# Values of VLLM_MMA_EMU_ALGORITHM, matching design_space::Algorithm.
ALGORITHMS = {"gdfs": 1, "cofda": 2}


def resolve_algorithm() -> int | None:
    """The configured algorithm id, or None if unset or unrecognised."""
    name = envs.VLLM_MMA_EMU_ALGORITHM
    if name is None:
        return None
    return ALGORITHMS.get(name.strip().lower())


def config_error(is_fp4: bool) -> str | None:
    """Why the current configuration cannot be used, or None if it can.

    The bitwidth ranges and the wording that reports them come from the
    kernels, so they are stated once in design_space.cuh rather than restated
    here.
    """
    name = envs.VLLM_MMA_EMU_ALGORITHM
    if name is None:
        return (
            "MMA emulation is off; set VLLM_MMA_EMU_ALGORITHM to "
            f"{' or '.join(sorted(ALGORITHMS))}."
        )

    algorithm = resolve_algorithm()
    if algorithm is None:
        return (
            f"VLLM_MMA_EMU_ALGORITHM must be {' or '.join(sorted(ALGORITHMS))}, "
            f"got {name!r}."
        )

    return (
        torch.ops._C.mma_emu_config_error(
            algorithm,
            envs.VLLM_MMA_EMU_F,
            envs.VLLM_MMA_EMU_G,
            envs.VLLM_MMA_EMU_GS,
            envs.VLLM_MMA_EMU_CS,
            is_fp4,
        )
        or None
    )
