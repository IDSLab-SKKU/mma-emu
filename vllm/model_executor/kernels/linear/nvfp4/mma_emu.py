# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVFP4 linear kernel backed by the MMA accumulation emulation.

Computes the same product as the CUTLASS path, but on CUDA cores with the
accumulation arithmetic under software control. Selected with
`--linear-backend mma_emu` and configured through `--kernel-config`. The
chunk and group sizes are fixed at 16 by the kernel, so only the algorithm,
F and G apply here.
"""

import torch

from vllm._custom_ops import mma_emu_scaled_nvfp4_mm, scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform

from ..mma_emu_config import config_error, emulation_config, resolve_algorithm
from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class MmaEmuNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the MMA accumulation emulation."""

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        # Read once: apply_weights runs per layer per forward pass.
        emu = emulation_config()
        self.algorithm = resolve_algorithm(emu)
        self.f_bits = emu.f_bits
        self.g_bits = emu.g_bits

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        if compute_capability is not None and compute_capability < 100:
            return False, "requires compute capability >= 10.0."
        if not hasattr(torch.ops._C, "mma_emu_scaled_nvfp4_mm"):
            return False, "vLLM was not built with the MMA-Emu NVFP4 kernel."
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        err = config_error(is_fp4=True)
        return (False, err) if err else (True, None)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The emulation reads the same swizzled block-scale layout and the same
        # padded weight as the CUTLASS path, so the preparation is shared.
        layer.weight_scale = torch.nn.Parameter(
            swizzle_blockscale(layer.weight_scale.data), requires_grad=False
        )
        padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(
            layer.weight.data
        )
        layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
        layer.weights_padding_cols = weights_padding_cols

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)

        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="cutlass",
            padded_n=x.shape[-1] + weights_padding_bytes * 2,
        )

        out = mma_emu_scaled_nvfp4_mm(
            x_fp4,
            layer.weight,
            x_blockscale,
            layer.weight_scale,
            layer.alpha,
            output_dtype,
            algorithm=self.algorithm,
            f_bits=self.f_bits,
            g_bits=self.g_bits,
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
