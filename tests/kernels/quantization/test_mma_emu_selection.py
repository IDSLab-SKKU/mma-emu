# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MMA-Emu linear kernel selection (CPU-only).

Run `pytest tests/kernels/quantization/test_mma_emu_selection.py`.

No GEMM runs here. These tests build a layer configuration, ask the kernel
classes whether they can implement it, and check what the selector picks. The
arithmetic is tested separately, in test_mma_emu.py.

What this file protects is the decision to put the emulation kernels at the head
of the candidate list. Selection takes the first kernel that says yes, and
CutlassFP8ScaledMMLinearKernel says yes to everything, so ordering the emulation
after it would make it unreachable. Ordering it first means the only thing
keeping it from taking every FP8 layer is that it declines when no accumulation
algorithm is configured — and a kernel that wrongly accepts still returns
correct numbers, just tens of times slower, so nothing would announce the
regression.
"""

import pytest
import torch
from pydantic import ValidationError

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig, LinearBackend
from vllm.config.mma_emu import MmaEmuConfig
from vllm.model_executor.kernels.linear import (
    _POSSIBLE_FP8_KERNELS,
    CutlassFP8ScaledMMLinearKernel,
    MmaEmuFP8ScaledMMLinearKernel,
    MmaEmuNvFp4LinearKernel,
    choose_scaled_mm_linear_kernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTensorSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)

pytestmark = pytest.mark.cpu_test

# The range checks live in the kernels and are reached through
# torch.ops._C.mma_emu_config_error, so those tests need the built extension —
# though not a GPU, since that operator registers under
# CompositeExplicitAutograd. The checks that matter most do not need it: they
# return before the operator is consulted.
HAS_OPS = hasattr(torch.ops._C, "mma_emu_config_error")
needs_ops = pytest.mark.skipif(
    not HAS_OPS, reason="vLLM was not built with the MMA-Emu kernels"
)

COFDA = MmaEmuConfig(algorithm="cofda")


def fp8_layer(
    activation_quant_key=kFp8StaticTensorSym,
    weight_quant_key=kFp8StaticTensorSym,
) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        weight_shape=(512, 256),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


def with_config(emu: MmaEmuConfig, linear_backend: LinearBackend = "auto"):
    """Make an engine configuration current, as engine startup would."""
    return set_current_vllm_config(
        VllmConfig(
            kernel_config=KernelConfig(mma_emu=emu, linear_backend=linear_backend)
        )
    )


# ============================================================================
# The opt-in gate
# ============================================================================


def test_fp8_declines_when_no_algorithm_is_configured():
    """Unconfigured means unused. This is what keeps the default path intact."""
    with with_config(MmaEmuConfig()):
        can_implement, reason = MmaEmuFP8ScaledMMLinearKernel.can_implement(fp8_layer())
    assert not can_implement
    assert "MMA emulation is off" in reason


def test_nvfp4_declines_when_no_algorithm_is_configured():
    with with_config(MmaEmuConfig()):
        can_implement, reason = MmaEmuNvFp4LinearKernel.can_implement(
            NvFp4LinearLayerConfig()
        )
    assert not can_implement
    assert "MMA emulation is off" in reason


@needs_ops
def test_fp8_accepts_once_an_algorithm_is_configured():
    """The gate has to open too, or the kernel would be dead code."""
    with with_config(COFDA):
        can_implement, reason = MmaEmuFP8ScaledMMLinearKernel.can_implement(fp8_layer())
    assert can_implement, reason


@needs_ops
@pytest.mark.parametrize(
    "emu", [MmaEmuConfig(algorithm="gdfs"), MmaEmuConfig(algorithm="cofda")]
)
def test_fp8_accepts_either_algorithm(emu: MmaEmuConfig):
    with with_config(emu):
        can_implement, reason = MmaEmuFP8ScaledMMLinearKernel.can_implement(fp8_layer())
    assert can_implement, reason


# ============================================================================
# The capability gate
# ============================================================================
#
# The emulation applies one scale in its epilogue, so a per-token or
# per-channel scale vector has nowhere to go. Accepting one would drop it
# silently rather than fail.


@pytest.mark.parametrize(
    "activation_quant_key", [kFp8StaticTokenSym, kFp8DynamicTensorSym]
)
def test_fp8_declines_activation_scales_it_cannot_apply(activation_quant_key):
    with with_config(COFDA):
        can_implement, reason = MmaEmuFP8ScaledMMLinearKernel.can_implement(
            fp8_layer(activation_quant_key=activation_quant_key)
        )
    assert not can_implement
    assert "activation scales" in reason


def test_fp8_declines_per_channel_weight_scales():
    with with_config(COFDA):
        can_implement, reason = MmaEmuFP8ScaledMMLinearKernel.can_implement(
            fp8_layer(weight_quant_key=kFp8StaticChannelSym)
        )
    assert not can_implement
    assert "weight scales" in reason


# ============================================================================
# The configuration gate
# ============================================================================
#
# The accepted ranges live in design_space.cuh. These check that a rejection
# reaches Python with the kernels' own wording, rather than Python restating
# bounds that could drift.


@needs_ops
@pytest.mark.parametrize(
    "emu,expected",
    [
        (MmaEmuConfig(algorithm="cofda", f_bits=6), "f_bits must be in"),
        (MmaEmuConfig(algorithm="cofda", f_bits=36), "f_bits must be in"),
        (MmaEmuConfig(algorithm="cofda", chunk_size=8), "chunk_size must be"),
        (MmaEmuConfig(algorithm="gdfs", g_bits=33), "g_bits must be in"),
        (MmaEmuConfig(algorithm="gdfs", group_size=4), "group_size must be"),
    ],
)
def test_fp8_declines_configurations_the_kernels_reject(emu, expected):
    with with_config(emu):
        can_implement, reason = MmaEmuFP8ScaledMMLinearKernel.can_implement(fp8_layer())
    assert not can_implement
    assert expected in reason, reason


@needs_ops
def test_nvfp4_g_bound_is_tighter_than_fp8():
    """G = 32 is lossless for FP8 products and out of range for E2M1 ones."""
    emu = MmaEmuConfig(algorithm="gdfs", g_bits=32)
    with with_config(emu):
        fp8_ok, _ = MmaEmuFP8ScaledMMLinearKernel.can_implement(fp8_layer())
        fp4_ok, fp4_reason = MmaEmuNvFp4LinearKernel.can_implement(
            NvFp4LinearLayerConfig()
        )
    assert fp8_ok
    assert not fp4_ok
    assert "g_bits must be in" in fp4_reason


def test_config_rejects_an_unrecognised_algorithm():
    """An unrecognised name never reaches can_implement: the field is a
    Literal, so the configuration layer refuses it first. Recorded here so the
    guard is not moved back into the kernel by mistake."""
    with pytest.raises(ValidationError):
        MmaEmuConfig(algorithm="fda")  # type: ignore[arg-type]


# ============================================================================
# What the selector actually picks
# ============================================================================
#
# The tests above check what the kernel answers. These check the consequence,
# so that reordering the candidate list is caught even if can_implement is
# untouched.


@needs_ops
def test_cutlass_is_selected_when_emulation_is_off():
    """The emulation is first in the list, and must still lose by default."""
    with with_config(MmaEmuConfig()):
        chosen = choose_scaled_mm_linear_kernel(
            fp8_layer(), _POSSIBLE_FP8_KERNELS, compute_capability=120
        )
    assert chosen is CutlassFP8ScaledMMLinearKernel


@needs_ops
def test_emulation_is_selected_once_configured():
    with with_config(COFDA):
        chosen = choose_scaled_mm_linear_kernel(
            fp8_layer(), _POSSIBLE_FP8_KERNELS, compute_capability=120
        )
    assert chosen is MmaEmuFP8ScaledMMLinearKernel


@needs_ops
def test_linear_backend_selects_the_emulation():
    with with_config(COFDA, linear_backend="mma_emu"):
        chosen = choose_scaled_mm_linear_kernel(
            fp8_layer(), _POSSIBLE_FP8_KERNELS, compute_capability=120
        )
    assert chosen is MmaEmuFP8ScaledMMLinearKernel


@needs_ops
def test_asking_for_the_emulation_without_configuring_it_is_an_error():
    """Explicit intent must not fall back silently to a different kernel."""
    with (
        with_config(MmaEmuConfig(), linear_backend="mma_emu"),
        pytest.raises(ValueError, match="Failed to find a kernel"),
    ):
            choose_scaled_mm_linear_kernel(
                fp8_layer(), _POSSIBLE_FP8_KERNELS, compute_capability=120
            )


def test_emulation_precedes_cutlass_in_the_candidate_list():
    """Ordered after it, the emulation would be unreachable: Cutlass accepts
    every configuration, and selection stops at the first acceptance."""
    from vllm.platforms import PlatformEnum

    candidates = _POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA]
    assert candidates.index(MmaEmuFP8ScaledMMLinearKernel) < candidates.index(
        CutlassFP8ScaledMMLinearKernel
    )
