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

A failure here does not necessarily mean the emulation is wrong. It can also
mean the accumulation configuration this architecture actually implements
differs from the one recorded in ARCH_ACCUMULATION below. Only SM120 has been
checked directly; the other entries come from the paper.
"""

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import quant_nvfp4_tensor
from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("MMA-Emu kernels require CUDA", allow_module_level=True)

if not hasattr(torch.ops._C, "mma_emu_scaled_fp8_mm"):
    pytest.skip("vLLM was not built with the MMA-Emu kernels", allow_module_level=True)

GDFS, COFDA = 1, 2

CAPABILITY = current_platform.get_device_capability()
SM = CAPABILITY[0] * 10 + CAPABILITY[1]


class Config(dict):
    """An accumulation configuration, as keyword arguments for the ops."""

    def __repr__(self) -> str:
        algo = "gdfs" if self["algorithm"] == GDFS else "cofda"
        rest = " ".join(
            f"{k[0].upper()}={v}" for k, v in self.items() if k != "algorithm"
        )
        return f"{algo} {rest}"


# The accumulation each architecture implements. FDA is CoFDA with the chunk
# spanning the K tile, so it is expressed as CoFDA with the corresponding CS.
ARCH_ACCUMULATION: dict[int, dict[str, Config]] = {
    # Ada
    89: {
        "fp8": Config(
            algorithm=COFDA, f_bits=13, g_bits=6, group_size=16, chunk_size=16
        ),
    },
    # Hopper
    90: {
        "fp8": Config(
            algorithm=COFDA, f_bits=13, g_bits=6, group_size=16, chunk_size=32
        ),
    },
    # Blackwell, data center
    100: {
        "fp8": Config(
            algorithm=COFDA, f_bits=25, g_bits=6, group_size=16, chunk_size=32
        ),
        "nvfp4": Config(algorithm=GDFS, f_bits=35, g_bits=6),
    },
    # Blackwell, workstation
    120: {
        "fp8": Config(
            algorithm=COFDA, f_bits=25, g_bits=6, group_size=16, chunk_size=32
        ),
        "nvfp4": Config(algorithm=GDFS, f_bits=35, g_bits=6),
    },
}


def arch_config(fmt: str) -> Config:
    """The configuration this GPU implements, or skip if it is not recorded."""
    if SM not in ARCH_ACCUMULATION:
        pytest.skip(
            f"No accumulation configuration recorded for SM{SM}. Add one to "
            f"ARCH_ACCUMULATION to test this architecture."
        )
    if fmt not in ARCH_ACCUMULATION[SM]:
        pytest.skip(f"SM{SM} has no {fmt} entry in ARCH_ACCUMULATION.")
    return ARCH_ACCUMULATION[SM][fmt]


# Taken from test_cutlass_scaled_mm.py, so the emulation is exercised over the
# same shapes as the kernel it has to reproduce. M covers a single token, counts
# that are not multiples of the tile (33, 100), and batches up to 512; K includes
# 496, which is 16- but not 32-aligned.
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

# NVFP4 packs two elements per byte and scales blocks of 16, so the kernel
# requires N and K to be multiples of 32. That drops the 496 entries above.
NVFP4_MNK = [mnk for mnk in FP8_MNK if mnk[1] % 32 == 0 and mnk[2] % 32 == 0]

OUT_DTYPES = [torch.bfloat16, torch.float16]


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


def assert_bit_exact(native: torch.Tensor, emulated: torch.Tensor, config: Config):
    if torch.equal(native, emulated):
        return
    diff = (native.float() - emulated.float()).abs()
    mismatched = int((native != emulated).sum())
    pytest.fail(
        f"emulation at {config!r} does not reproduce the native result on "
        f"SM{SM}: {mismatched}/{native.numel()} elements differ, "
        f"max|d|={diff.max():.3e}, mean|d|={diff.mean():.3e}"
    )


@pytest.mark.parametrize("m,n,k", FP8_MNK)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_matches_native(m: int, n: int, k: int, out_dtype, use_bias: bool):
    """FP8 emulation at this architecture's accumulation must match CUTLASS."""
    config = arch_config("fp8")
    a, b, scale_a, scale_b = fp8_inputs(m, n, k, seed=m * 31 + k)
    bias = torch.randn(n, device="cuda", dtype=out_dtype) if use_bias else None

    native = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    emulated = ops.mma_emu_scaled_fp8_mm(
        a, b, scale_a, scale_b, out_dtype, bias, **config
    )

    assert emulated.shape == native.shape
    assert emulated.dtype == native.dtype
    assert_bit_exact(native, emulated, config)


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires compute capability >= 10.0",
)
@pytest.mark.parametrize("m,n,k", NVFP4_MNK)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_nvfp4_matches_native(m: int, n: int, k: int, out_dtype):
    """NVFP4 emulation at this architecture's accumulation must match CUTLASS."""
    config = arch_config("nvfp4")
    g = torch.Generator(device="cuda").manual_seed(m * 31 + k)
    a_ref = torch.randn(m, k, generator=g, device="cuda", dtype=out_dtype) * 0.3
    b_ref = torch.randn(n, k, generator=g, device="cuda", dtype=out_dtype) * 0.3

    a, a_sf, a_gs = quant_nvfp4_tensor(a_ref)
    b, b_sf, b_gs = quant_nvfp4_tensor(b_ref)
    alpha = (1.0 / (a_gs * b_gs)).to(torch.float32).reshape(1)

    native = ops.cutlass_scaled_fp4_mm(a, b, a_sf, b_sf, alpha, out_dtype)
    emulated = ops.mma_emu_scaled_nvfp4_mm(a, b, a_sf, b_sf, alpha, out_dtype, **config)

    assert emulated.shape == native.shape
    assert emulated.dtype == native.dtype
    assert_bit_exact(native, emulated, config)


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            Config(algorithm=COFDA, f_bits=13, g_bits=6, group_size=16, chunk_size=32),
            None,
        ),
        (
            Config(algorithm=GDFS, f_bits=35, g_bits=32, group_size=8, chunk_size=32),
            None,
        ),
        (
            Config(algorithm=3, f_bits=13, g_bits=6, group_size=16, chunk_size=32),
            "algorithm must be",
        ),
        (
            Config(algorithm=COFDA, f_bits=6, g_bits=6, group_size=16, chunk_size=32),
            "f_bits must be in",
        ),
        (
            Config(algorithm=COFDA, f_bits=36, g_bits=6, group_size=16, chunk_size=32),
            "f_bits must be in",
        ),
        (
            Config(algorithm=GDFS, f_bits=13, g_bits=33, group_size=16, chunk_size=32),
            "g_bits must be in",
        ),
        (
            Config(algorithm=COFDA, f_bits=13, g_bits=6, group_size=16, chunk_size=8),
            "chunk_size must be",
        ),
        (
            Config(algorithm=GDFS, f_bits=13, g_bits=6, group_size=4, chunk_size=32),
            "group_size must be",
        ),
    ],
)
def test_fp8_config_validation(config: Config, expected: str | None):
    """The kernels decide which configurations are accepted, and say why not."""
    err = torch.ops._C.mma_emu_config_error(
        config["algorithm"],
        config["f_bits"],
        config["g_bits"],
        config["group_size"],
        config["chunk_size"],
        False,
    )
    if expected is None:
        assert err == "", f"expected {config!r} to be accepted, got {err!r}"
    else:
        assert expected in err, f"expected {expected!r} in {err!r}"


def test_nvfp4_g_bound_is_tighter_than_fp8():
    """E2M1 products reach their lossless width sooner than FP8 E4M3 ones."""
    assert torch.ops._C.mma_emu_config_error(GDFS, 13, 32, 16, 32, False) == ""
    assert "g_bits" in torch.ops._C.mma_emu_config_error(GDFS, 13, 32, 16, 32, True)
    assert torch.ops._C.mma_emu_config_error(GDFS, 13, 6, 16, 16, True) == ""


@pytest.mark.parametrize("m,n,k", [(64, 256, 512)])
def test_fp8_rejects_bad_config_at_the_operator(m: int, n: int, k: int):
    """The operator rejects a configuration even when Python is bypassed."""
    a, b, scale_a, scale_b = fp8_inputs(m, n, k, seed=0)
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="f_bits must be in"):
        torch.ops._C.mma_emu_scaled_fp8_mm(
            out, a, b, scale_a, scale_b, None, COFDA, 100, 6, 16, 32
        )


# ============================================================================
# Operator schema
# ============================================================================


def test_fp8_opcheck():
    """The schema was written by hand; opcheck confirms it matches behaviour."""
    config = arch_config("fp8")
    a, b, scale_a, scale_b = fp8_inputs(64, 256, 512, seed=0)
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
            config["algorithm"],
            config["f_bits"],
            config["g_bits"],
            config["group_size"],
            config["chunk_size"],
        ),
    )


# mma_emu_config_error is not opchecked: it takes no tensor and returns a
# string, which opcheck compares with assert_close.


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="NVFP4 requires compute capability >= 10.0",
)
def test_nvfp4_opcheck():
    config = arch_config("nvfp4")
    g = torch.Generator(device="cuda").manual_seed(0)

    def rand(rows):
        return (
            torch.randn(rows, 512, generator=g, device="cuda", dtype=torch.bfloat16)
            * 0.3
        )

    a_ref, b_ref = rand(64), rand(256)
    a, a_sf, a_gs = quant_nvfp4_tensor(a_ref)
    b, b_sf, b_gs = quant_nvfp4_tensor(b_ref)
    alpha = (1.0 / (a_gs * b_gs)).to(torch.float32).reshape(1)
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
            config["algorithm"],
            config["f_bits"],
            config["g_bits"],
        ),
    )


# ============================================================================
# Tile boundaries
# ============================================================================


@pytest.mark.parametrize("use_bias", [False, True])
def test_fp8_m_sweep(use_bias: bool):
    """Every M from 1 to 128, to catch a tile-boundary case the shape list
    happens to step over."""
    config = arch_config("fp8")
    for nk in range(32, 128, 32):
        for m in range(1, 128):
            a, b, scale_a, scale_b = fp8_inputs(m, nk, nk, seed=m)
            bias = (
                torch.randn(nk, device="cuda", dtype=torch.bfloat16)
                if use_bias
                else None
            )
            native = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16, bias)
            emulated = ops.mma_emu_scaled_fp8_mm(
                a, b, scale_a, scale_b, torch.bfloat16, bias, **config
            )
            if not torch.equal(native, emulated):
                diff = (native.float() - emulated.float()).abs()
                pytest.fail(f"M={m} N=K={nk} bias={use_bias}: max|d|={diff.max():.3e}")


# ============================================================================
# Layout
# ============================================================================
#
# The kernels index with a packed stride, so a sliced view would be read as
# though it were dense. cutlass_scaled_mm has the same limitation but does not
# say so, returning a wrong product instead; these check that the emulation
# refuses rather than following it.


def test_fp8_rejects_non_contiguous_a():
    whole = (torch.randn(256, 1024, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    _, b, scale_a, scale_b = fp8_inputs(128, 256, 512, seed=0)
    a = whole[0:128, 0:512]
    assert a.stride(0) != a.size(1)
    out = torch.empty((128, 256), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops._C.mma_emu_scaled_fp8_mm(
            out, a, b, scale_a, scale_b, None, COFDA, 13, 6, 16, 32
        )


def test_fp8_rejects_non_contiguous_out():
    a, b, scale_a, scale_b = fp8_inputs(128, 256, 512, seed=0)
    out = torch.empty((128, 512), dtype=torch.bfloat16, device="cuda")[:, 0:256]
    assert out.stride(0) != out.size(1)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops._C.mma_emu_scaled_fp8_mm(
            out, a, b, scale_a, scale_b, None, COFDA, 13, 6, 16, 32
        )
