# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MMA accumulation emulation kernels.

Run `pytest tests/kernels/quantization/test_mma_emu.py`.

The emulation reproduces on CUDA cores the accumulation arithmetic a tensor
core performs in fixed function. Every commercially deployed accumulation
configuration falls inside the design space the kernels accept, so on real
silicon the emulation configured to match the GPU it runs on should reproduce
the native CUTLASS result bit for bit. That is what these tests check, for the
architecture they happen to run on.

They go through the linear-kernel classes rather than calling the operators
directly, comparing MmaEmu*Kernel against Cutlass*Kernel on the same inputs.
That is the claim worth testing — that the emulation is a drop-in for the
native path — and it covers the weight preparation and activation quantisation
around the GEMM, which calling the operators bypasses. The checks that are
inherently about an operator, its schema and the inputs it refuses, stay at
that level and are grouped at the end.

A failure does not necessarily mean the emulation is wrong. It can also mean
the accumulation configuration this architecture implements differs from the
one recorded in ARCH_ACCUMULATION below. Only SM120 has been checked directly;
the other entries come from the paper.
"""

import contextlib
from typing import NoReturn

import pytest
import torch

from tests.kernels.quantization.mma_emu_arch import (
    ALGORITHM_IDS,
    ARCH_ACCUMULATION,
    COFDA,
    GDFS,
    SM,
)
from tests.kernels.utils import opcheck
from vllm._custom_ops import scaled_fp4_quant
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import KernelConfig
from vllm.config.mma_emu import MmaEmuConfig
from vllm.model_executor.kernels.linear import (
    CutlassFP8ScaledMMLinearKernel,
    CutlassNvFp4LinearKernel,
    MmaEmuFP8ScaledMMLinearKernel,
    MmaEmuNvFp4LinearKernel,
)
from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("MMA-Emu kernels require CUDA", allow_module_level=True)

if not hasattr(torch.ops._C, "mma_emu_scaled_fp8_mm"):
    pytest.skip("vLLM was not built with the MMA-Emu kernels", allow_module_level=True)

FLOAT4_MAX, FLOAT8_MAX = 6.0, 448.0


def _skip(reason: str) -> NoReturn:
    """pytest.skip raises, but it is not annotated as such, so a caller that
    relies on it to narrow a type has to say so."""
    pytest.skip(reason)
    raise AssertionError("pytest.skip returned")


def arch_config(fmt: str) -> MmaEmuConfig:
    """The configuration this GPU implements, or skip if it is not recorded."""
    accumulation = ARCH_ACCUMULATION.get(SM) if SM is not None else None
    if accumulation is None:
        _skip(
            f"No accumulation configuration recorded for SM{SM}. Add one to "
            f"ARCH_ACCUMULATION in mma_emu_arch.py to test this architecture."
        )
    if fmt not in accumulation:
        _skip(f"SM{SM} has no {fmt} entry in ARCH_ACCUMULATION.")
    return accumulation[fmt]


# Taken from test_cutlass_scaled_mm.py, so the emulation is exercised over the
# same shapes as the kernel it has to reproduce. M covers a single token, counts
# that are not multiples of the tile (33, 100), and batches up to 512.
#
# Every entry is 16-aligned in N and K. Below that, cutlass_scaled_mm hands off
# to triton_scaled_mm, which would change what the emulation is compared
# against without saying so.
FP8_MNK = [
    (1, 256, 128),
    (1, 16384, 1024),
    (1, 24576, 496),
    (16, 256, 496),
    (16, 16384, 128),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (33, 8192, 128),
    (64, 2048, 496),
    (64, 16384, 1024),
    (100, 8192, 496),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]

# NVFP4 packs two elements per byte and scales blocks of 16, so the operator
# requires N and K to be multiples of 32.
NVFP4_MNK = [mnk for mnk in FP8_MNK if mnk[1] % 32 == 0 and mnk[2] % 32 == 0]

# Shapes the operator would refuse, which the kernel classes reach it with by
# padding the weights and slicing the output. NVFP4 has no fallback to another
# kernel, so unlike FP8 the reference stays CUTLASS here, and this is the only
# way the padding path runs at all.
NVFP4_PADDED_MNK = [
    (128, 250, 496),  # rows and columns both padded
    (64, 100, 240),
    (128, 256, 496),  # columns only
]

OUT_DTYPES = [torch.bfloat16, torch.float16]


# ============================================================================
# Building the kernels
# ============================================================================


@contextlib.contextmanager
def fp8_kernels(n: int, k: int, emu: MmaEmuConfig, out_dtype: torch.dtype):
    """The emulation and CUTLASS FP8 kernels, prepared as a layer would."""
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8StaticTensorSym,
        weight_shape=(k, n),
        input_dtype=out_dtype,
        out_dtype=out_dtype,
    )
    names = ("weight", "weight_scale", "input_scale", "input_scale_ub")
    with set_current_vllm_config(VllmConfig(kernel_config=KernelConfig(mma_emu=emu))):
        built = []
        for cls in (MmaEmuFP8ScaledMMLinearKernel, CutlassFP8ScaledMMLinearKernel):
            kernel = cls(config, names)
            # Sets logical_output_size, which the CUTLASS kernel reads. A
            # weight of the right shape is all it needs.
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(
                torch.empty(k, n, dtype=torch.float8_e4m3fn, device="cuda"),
                requires_grad=False,
            )
            kernel.process_weights_after_loading(layer)
            built.append(kernel)
        yield built[0], built[1]


def fp8_inputs(m: int, n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, generator=g, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    # B is [K, N] and column-major.
    b = (
        (torch.randn(n, k, generator=g, device="cuda") * 0.3)
        .to(torch.float8_e4m3fn)
        .t()
    )
    scale_a = torch.full((1,), 0.05, device="cuda", dtype=torch.float32)
    scale_b = torch.full((1,), 0.07, device="cuda", dtype=torch.float32)
    return a, b, scale_a, scale_b


def nvfp4_layer(n: int, k: int, seed: int, dtype: torch.dtype) -> torch.nn.Module:
    """A layer as the quantization method leaves it, before the kernel's
    process_weights_after_loading: packed weights, unswizzled block scales."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, generator=g, device="cuda", dtype=dtype) * 0.3
    w_global_scale = (FLOAT4_MAX * FLOAT8_MAX) / w.abs().max().float()
    w_quant, w_block_scale = scaled_fp4_quant(
        w, w_global_scale, is_sf_swizzled_layout=False
    )

    x_global_scale = torch.tensor(
        FLOAT4_MAX * FLOAT8_MAX / 3.0, device="cuda", dtype=torch.float32
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_quant, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_block_scale, requires_grad=False)
    layer.alpha = (1.0 / (x_global_scale * w_global_scale)).to(torch.float32)
    layer.input_global_scale_inv = x_global_scale
    layer.output_size_per_partition = n
    return layer


def assert_bit_exact(native: torch.Tensor, emulated: torch.Tensor, config):
    if torch.equal(native, emulated):
        return
    diff = (native.float() - emulated.float()).abs()
    mismatched = int((native != emulated).sum())
    pytest.fail(
        f"emulation at {config} does not reproduce the native result on "
        f"SM{SM}: {mismatched}/{native.numel()} elements differ, "
        f"max|d|={diff.max():.3e}, mean|d|={diff.mean():.3e}"
    )


# ============================================================================
# FP8
# ============================================================================


@pytest.mark.parametrize("m,n,k", FP8_MNK)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_matches_native(m: int, n: int, k: int, out_dtype, use_bias: bool):
    """FP8 emulation at this architecture's accumulation must match CUTLASS."""
    emu = arch_config("fp8")
    a, b, scale_a, scale_b = fp8_inputs(m, n, k, m * 31 + k)
    bias = torch.randn(n, device="cuda", dtype=out_dtype) if use_bias else None
    args = dict(
        A=a,
        B=b,
        out_dtype=out_dtype,
        As=scale_a,
        Bs=scale_b,
        bias=bias,
        output_shape=[m, n],
    )

    with fp8_kernels(n, k, emu, out_dtype) as (emulation, cutlass):
        emulated = emulation.apply_scaled_mm(**args)
        native = cutlass.apply_scaled_mm(**args)

    assert emulated.shape == native.shape
    assert emulated.dtype == native.dtype
    assert_bit_exact(native, emulated, emu)


@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_m_sweep(use_bias: bool):
    """Every M from 1 to 127, to catch a tile boundary the shape list steps
    over."""
    emu = arch_config("fp8")
    for nk in range(32, 128, 32):
        bias = (
            torch.randn(nk, device="cuda", dtype=torch.bfloat16) if use_bias else None
        )
        with fp8_kernels(nk, nk, emu, torch.bfloat16) as (emulation, cutlass):
            for m in range(1, 128):
                a, b, scale_a, scale_b = fp8_inputs(m, nk, nk, m)
                args = dict(
                    A=a,
                    B=b,
                    out_dtype=torch.bfloat16,
                    As=scale_a,
                    Bs=scale_b,
                    bias=bias,
                    output_shape=[m, nk],
                )
                emulated = emulation.apply_scaled_mm(**args)
                native = cutlass.apply_scaled_mm(**args)
                if not torch.equal(native, emulated):
                    diff = (native.float() - emulated.float()).abs()
                    pytest.fail(
                        f"M={m} N=K={nk} bias={use_bias}: max|d|={diff.max():.3e}"
                    )


# ============================================================================
# NVFP4
# ============================================================================


needs_nvfp4 = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires compute capability >= 10.0",
)


def run_nvfp4_pair(m: int, n: int, k: int, emu: MmaEmuConfig, out_dtype):
    """Both kernels, each over its own copy of the same layer."""
    g = torch.Generator(device="cuda").manual_seed(m * 31 + k)
    x = torch.randn(m, k, generator=g, device="cuda", dtype=out_dtype) * 0.3

    outputs, padding = [], []
    with set_current_vllm_config(VllmConfig(kernel_config=KernelConfig(mma_emu=emu))):
        for cls in (MmaEmuNvFp4LinearKernel, CutlassNvFp4LinearKernel):
            layer = nvfp4_layer(n, k, seed=n * 17 + k, dtype=out_dtype)
            kernel = cls(NvFp4LinearLayerConfig())
            kernel.process_weights_after_loading(layer)
            outputs.append(kernel.apply_weights(layer, x, bias=None))
            padding.append(layer.weights_padding_cols)
    return outputs[0], outputs[1], padding[0]


@needs_nvfp4
@pytest.mark.parametrize("m,n,k", NVFP4_MNK)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_nvfp4_matches_native(m: int, n: int, k: int, out_dtype):
    """NVFP4 emulation at this architecture's accumulation must match CUTLASS."""
    emu = arch_config("nvfp4")
    emulated, native, _ = run_nvfp4_pair(m, n, k, emu, out_dtype)
    assert emulated.shape == native.shape
    assert emulated.dtype == native.dtype
    assert_bit_exact(native, emulated, emu)


@needs_nvfp4
@pytest.mark.parametrize("m,n,k", NVFP4_PADDED_MNK)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_nvfp4_matches_native_when_padded(m: int, n: int, k: int, out_dtype):
    """The operator requires N and K to be multiples of 32; the kernel classes
    get there by padding the weights and slicing the output back."""
    emu = arch_config("nvfp4")
    emulated, native, padding_cols = run_nvfp4_pair(m, n, k, emu, out_dtype)
    assert padding_cols > 0, (
        f"N={n} K={k} was expected to need padding but did not, so this case "
        f"exercises nothing the aligned shapes do not"
    )
    assert emulated.shape == native.shape
    assert_bit_exact(native, emulated, emu)


# ============================================================================
# The operators themselves
# ============================================================================
#
# Schemas, and the inputs the operators refuse. None of this is reachable
# through the kernel classes.


def test_fp8_opcheck():
    """The schema was written by hand; opcheck confirms it matches behaviour."""
    emu = arch_config("fp8")
    a, b, scale_a, scale_b = fp8_inputs(64, 256, 512, 0)
    out = torch.empty((64, 256), dtype=torch.bfloat16, device="cuda")
    opcheck(
        torch.ops._C.mma_emu_scaled_fp8_mm,
        (
            out,
            a,
            b,
            scale_a,
            scale_b,
            None,
            ALGORITHM_IDS[emu.algorithm],
            emu.f_bits,
            emu.g_bits,
            emu.group_size,
            emu.chunk_size,
        ),
    )


# mma_emu_config_error is not opchecked: it takes no tensor and returns a
# string, which opcheck compares with assert_close.


@needs_nvfp4
def test_nvfp4_opcheck():
    emu = arch_config("nvfp4")

    def quantized(rows):
        g = torch.Generator(device="cuda").manual_seed(rows)
        x = torch.randn(rows, 512, generator=g, device="cuda", dtype=torch.bfloat16)
        return scaled_fp4_quant(x, torch.tensor(1.0, device="cuda"))

    a, a_sf = quantized(64)
    b, b_sf = quantized(256)
    alpha = torch.ones(1, device="cuda", dtype=torch.float32)
    out = torch.empty((64, 256), dtype=torch.bfloat16, device="cuda")
    opcheck(
        torch.ops._C.mma_emu_scaled_nvfp4_mm,
        (
            out,
            a,
            b,
            a_sf,
            b_sf,
            alpha,
            ALGORITHM_IDS[emu.algorithm],
            emu.f_bits,
            emu.g_bits,
        ),
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        ((COFDA, 13, 6, 16, 32), None),
        ((GDFS, 35, 32, 8, 32), None),
        ((3, 13, 6, 16, 32), "algorithm must be"),
        ((COFDA, 6, 6, 16, 32), "f_bits must be in"),
        ((COFDA, 36, 6, 16, 32), "f_bits must be in"),
        ((GDFS, 13, 33, 16, 32), "g_bits must be in"),
        ((COFDA, 13, 6, 16, 8), "chunk_size must be"),
        ((GDFS, 13, 6, 4, 32), "group_size must be"),
    ],
)
def test_fp8_config_validation(config, expected):
    """The kernels decide which configurations are accepted, and say why not."""
    err = torch.ops._C.mma_emu_config_error(*config, False)
    if expected is None:
        assert err == "", f"expected {config} to be accepted, got {err!r}"
    else:
        assert expected in err, f"expected {expected!r} in {err!r}"


def test_nvfp4_g_bound_is_tighter_than_fp8():
    """E2M1 products reach their lossless width sooner than FP8 E4M3 ones."""
    assert torch.ops._C.mma_emu_config_error(GDFS, 13, 32, 16, 32, False) == ""
    assert "g_bits" in torch.ops._C.mma_emu_config_error(GDFS, 13, 32, 16, 32, True)
    assert torch.ops._C.mma_emu_config_error(GDFS, 13, 6, 16, 16, True) == ""


def test_fp8_operator_rejects_a_bad_configuration():
    """Reachable directly, so it validates rather than trusting the caller."""
    a, b, scale_a, scale_b = fp8_inputs(64, 256, 512, 0)
    out = torch.empty((64, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="f_bits must be in"):
        torch.ops._C.mma_emu_scaled_fp8_mm(
            out, a, b, scale_a, scale_b, None, COFDA, 100, 6, 16, 32
        )


# The kernels index with a packed stride, so a sliced view would be read as
# though it were dense. cutlass_scaled_mm has the same limitation but does not
# say so, returning a wrong product instead.


def test_fp8_operator_rejects_non_contiguous_a():
    whole = (torch.randn(256, 1024, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    _, b, scale_a, scale_b = fp8_inputs(128, 256, 512, 0)
    a = whole[0:128, 0:512]
    assert a.stride(0) != a.size(1)
    out = torch.empty((128, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops._C.mma_emu_scaled_fp8_mm(
            out, a, b, scale_a, scale_b, None, COFDA, 13, 6, 16, 32
        )


def test_fp8_operator_rejects_non_contiguous_out():
    a, b, scale_a, scale_b = fp8_inputs(128, 256, 512, 0)
    out = torch.empty((128, 512), dtype=torch.bfloat16, device="cuda")[:, 0:256]
    assert out.stride(0) != out.size(1)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops._C.mma_emu_scaled_fp8_mm(
            out, a, b, scale_a, scale_b, None, COFDA, 13, 6, 16, 32
        )
