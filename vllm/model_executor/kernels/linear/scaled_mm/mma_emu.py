# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FP8 linear kernel backed by the MMA accumulation emulation.

Computes the same product as the CUTLASS path, but on CUDA cores with the
accumulation arithmetic under software control, so the algorithm and its
bitwidths become parameters. Selected with `--linear-backend mma_emu` and
configured through the VLLM_MMA_EMU_* environment variables.
"""

from collections.abc import Sequence

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

from ..mma_emu_config import config_error, resolve_algorithm
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

class MmaEmuFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        # Read once: apply_scaled_mm runs per layer per forward pass.
        self.algorithm = resolve_algorithm()
        self.f_bits = envs.VLLM_MMA_EMU_F
        self.g_bits = envs.VLLM_MMA_EMU_G
        self.group_size = envs.VLLM_MMA_EMU_GS
        self.chunk_size = envs.VLLM_MMA_EMU_CS

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability >= 8.9."
        if not hasattr(torch.ops._C, "mma_emu_scaled_fp8_mm"):
            return False, "vLLM was not built with the MMA-Emu FP8 kernel."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # The emulation applies one scale in its epilogue, so a per-row or
        # per-column scale vector has nowhere to go.
        if c.activation_quant_key != kFp8StaticTensorSym:
            return False, "only static per-tensor activation scales are supported."
        if not c.weight_quant_key.scale.group_shape.is_per_tensor():
            return False, "only per-tensor weight scales are supported."

        err = config_error(is_fp4=False)
        return (False, err) if err else (True, None)

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The emulation kernel bounds-checks its tiles, so the weight needs no
        # alignment padding.
        return

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        output = ops.mma_emu_scaled_fp8_mm(
            A,
            B,
            As,
            Bs,
            out_dtype,
            bias,
            algorithm=self.algorithm,
            f_bits=self.f_bits,
            g_bits=self.g_bits,
            group_size=self.group_size,
            chunk_size=self.chunk_size,
        )
        return output.view(*output_shape[:-1], B.shape[1])
