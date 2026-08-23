# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared configuration for the MMA accumulation emulation kernels.

The FP8 and NVFP4 kernels read the same accumulation settings and answer the
same question in ``can_implement``, so both go through here.
"""

import torch

from vllm.config.mma_emu import MmaEmuConfig

# Algorithm names, as they map onto design_space::Algorithm.
ALGORITHMS = {"gdfs": 1, "cofda": 2}


def emulation_config() -> MmaEmuConfig:
    """The accumulation configuration for the engine being constructed."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return config.kernel_config.mma_emu if config is not None else MmaEmuConfig()


def resolve_algorithm(config: MmaEmuConfig | None = None) -> int:
    """The configured algorithm id.

    Only called from a kernel's __init__, which runs after can_implement has
    accepted the layer, so an unset algorithm means that gate was bypassed
    rather than that the caller should cope with it.
    """
    config = config if config is not None else emulation_config()
    if config.algorithm is None:
        raise ValueError(
            "MMA emulation has no algorithm configured, but a kernel was "
            "constructed anyway; can_implement should have declined first."
        )
    return ALGORITHMS[config.algorithm]


def config_error(is_fp4: bool) -> str | None:
    """Why the current configuration cannot be used, or None if it can.

    The bitwidth ranges and the wording that reports them come from the
    kernels, so they are stated once in design_space.cuh rather than restated
    here.
    """
    config = emulation_config()

    if config.algorithm is None:
        return (
            "MMA emulation is off; set the algorithm, for example "
            '--kernel-config \'{"mma_emu": {"algorithm": "cofda"}}\'.'
        )

    return (
        torch.ops._C.mma_emu_config_error(
            ALGORITHMS[config.algorithm],
            config.f_bits,
            config.g_bits,
            config.group_size,
            config.chunk_size,
            is_fp4,
        )
        or None
    )
