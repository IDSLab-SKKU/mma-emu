# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from vllm.config.utils import config, get_hash_factors, hash_factors

MmaEmuAlgorithm = Literal["gdfs", "cofda"]


@config
class MmaEmuConfig:
    """Accumulation configuration for the MMA emulation kernels.

    Only consulted when `linear_backend` is `mma_emu`. Which values are
    accepted is decided by the kernels themselves, in `design_space.cuh`, and
    reported through `torch.ops._C.mma_emu_config_error`; the ranges are
    deliberately not restated here.
    """

    algorithm: MmaEmuAlgorithm | None = None
    """Accumulation algorithm to emulate. Unset means the emulation kernels
    decline every layer, so the native path runs."""

    f_bits: int = 13
    """Fractional bits F, shared by CoFDA and the GDFS inter-group stage."""

    g_bits: int = 6
    """Intra-group fractional bits G. GDFS only."""

    chunk_size: int = 32
    """Chunk size CS. FP8 CoFDA only; NVFP4 fixes it at 16."""

    group_size: int = 16
    """Group size GS. FP8 GDFS only; NVFP4 fixes it at 16."""

    def compute_hash(self) -> str:
        """
        Produces a hash unique to this accumulation configuration.

        Every field changes the arithmetic the kernels perform, so all of them
        belong in the compilation cache key. Two runs that compute different
        numbers must not share compiled artifacts.
        """
        return hash_factors(get_hash_factors(self, set()))
