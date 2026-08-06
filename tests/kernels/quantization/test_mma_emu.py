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


# M covers a single token, an odd count that exercises the tile bounds checks,
# and larger batches. N and K stay 16-aligned for FP8 and 32-aligned for NVFP4.
FP8_MNK = [
    (1, 256, 128),
    (7, 512, 256),
    (16, 256, 512),
    (64, 1024, 256),
    (128, 256, 1024),
    (256, 512, 512),
]

NVFP4_MNK = [
    (16, 256, 512),
    (64, 512, 256),
    (128, 256, 1024),
    (256, 512, 512),
]

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
# Grouped MoE
# ============================================================================
#
# The MoE kernel is built for SM90 only: cutlass_moe_mm has SM90 and SM100
# implementations and no SM120 one, so Hopper is where the emulation can be
# compared against a native result at all. These tests have never been run —
# they were written on a machine that cannot build the kernel — so treat a
# failure as a finding rather than assuming the test is right.

MOE_CASES = [
    # (num_experts, m_per_expert, n, k)
    (2, 64, 256, 256),
    (4, 32, 128, 512),
    (8, 16, 256, 128),
]


def make_moe_inputs(num_experts: int, m_per_expert: int, n: int, k: int, seed: int):
    """Build cutlass_moe_mm's arguments.

    a is [P, K] row-major with the experts concatenated; b holds N * K elements
    per expert with K contiguous, which is built as [E, N, K] and handed over as
    its [E, K, N] transpose, the way vLLM does it.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    total_rows = num_experts * m_per_expert

    expert_offsets = torch.arange(
        0, total_rows + 1, m_per_expert, device="cuda", dtype=torch.int64
    )
    problem_sizes = torch.zeros((num_experts, 3), device="cuda", dtype=torch.int32)
    problem_sizes[:, 0] = m_per_expert
    problem_sizes[:, 1] = n
    problem_sizes[:, 2] = k

    a = (torch.randn(total_rows, k, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn(num_experts, n, k, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    b = b.transpose(1, 2)

    # Per-tensor activation scale, one weight scale per expert.
    a_scales = torch.full((1, 1), 0.05, device="cuda", dtype=torch.float32)
    b_scales = torch.full((num_experts, 1), 0.07, device="cuda", dtype=torch.float32)

    ab_strides = torch.full(
        (num_experts,), a.stride(0), device="cuda", dtype=torch.int64
    )
    c_strides = torch.full((num_experts,), n, device="cuda", dtype=torch.int64)

    return dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        expert_offsets=expert_offsets[:-1],
        problem_sizes=problem_sizes,
        ab_strides=ab_strides,
        c_strides=c_strides,
        total_rows=total_rows,
        n=n,
    )


@pytest.mark.skipif(
    SM != 90,
    reason=f"the MoE emulation kernel is built for SM90 only, this is SM{SM}",
)
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "mma_emu_moe_mm"),
    reason="vLLM was not built with the MMA-Emu MoE kernel",
)
@pytest.mark.parametrize("num_experts,m_per_expert,n,k", MOE_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_moe_matches_native(
    num_experts: int, m_per_expert: int, n: int, k: int, out_dtype
):
    """Grouped MoE emulation at SM90's accumulation must match cutlass_moe_mm.

    CoFDA only; the MoE kernel does not implement GDFS.
    """
    config = arch_config("fp8")
    assert config["algorithm"] == COFDA, (
        "the MoE kernel is CoFDA only, but ARCH_ACCUMULATION records GDFS for "
        f"SM{SM}'s FP8 path"
    )

    inp = make_moe_inputs(num_experts, m_per_expert, n, k, seed=num_experts * 31 + k)
    shape = (inp["total_rows"], inp["n"])

    native = torch.zeros(shape, device="cuda", dtype=out_dtype)
    torch.ops._C.cutlass_moe_mm(
        native,
        inp["a"],
        inp["b"],
        inp["a_scales"],
        inp["b_scales"],
        inp["expert_offsets"],
        inp["problem_sizes"],
        inp["ab_strides"],
        inp["ab_strides"],
        inp["c_strides"],
        False,
        False,
    )

    emulated = torch.zeros(shape, device="cuda", dtype=out_dtype)
    torch.ops._C.mma_emu_moe_mm(
        emulated,
        inp["a"],
        inp["b"],
        inp["a_scales"],
        inp["b_scales"],
        inp["expert_offsets"],
        inp["problem_sizes"],
        inp["ab_strides"],
        inp["ab_strides"],
        inp["c_strides"],
        False,
        False,
        config["algorithm"],
        config["f_bits"],
        config["chunk_size"],
    )

    assert_bit_exact(native, emulated, config)


@pytest.mark.skipif(
    SM != 90,
    reason=f"the MoE emulation kernel is built for SM90 only, this is SM{SM}",
)
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "mma_emu_moe_mm"),
    reason="vLLM was not built with the MMA-Emu MoE kernel",
)
def test_moe_narrower_accumulator_diverges():
    """A narrow accumulator must change the result.

    Without this, test_moe_matches_native would also pass if the operator
    silently ignored the configuration and called the native path.
    """
    config = arch_config("fp8")
    inp = make_moe_inputs(4, 32, 128, 512, seed=0)
    shape = (inp["total_rows"], inp["n"])

    def run(f_bits: int, chunk_size: int) -> torch.Tensor:
        out = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        torch.ops._C.mma_emu_moe_mm(
            out,
            inp["a"],
            inp["b"],
            inp["a_scales"],
            inp["b_scales"],
            inp["expert_offsets"],
            inp["problem_sizes"],
            inp["ab_strides"],
            inp["ab_strides"],
            inp["c_strides"],
            False,
            False,
            COFDA,
            f_bits,
            chunk_size,
        )
        return out

    narrow_f, narrow_cs = 7, 16
    assert not torch.equal(
        run(config["f_bits"], config["chunk_size"]), run(narrow_f, narrow_cs)
    ), (
        f"F={narrow_f} CS={narrow_cs} produced the same result as this "
        f"architecture's own accumulation, so the configuration is not "
        f"reaching the kernel"
    )


@pytest.mark.skipif(
    SM != 90,
    reason=f"the MoE emulation kernel is built for SM90 only, this is SM{SM}",
)
@pytest.mark.skipif(
    not hasattr(torch.ops._C, "mma_emu_moe_mm"),
    reason="vLLM was not built with the MMA-Emu MoE kernel",
)
def test_moe_rejects_gdfs():
    """GDFS is not implemented for MoE and must be refused, not approximated."""
    inp = make_moe_inputs(2, 32, 128, 256, seed=0)
    out = torch.zeros(
        (inp["total_rows"], inp["n"]), device="cuda", dtype=torch.bfloat16
    )
    with pytest.raises(RuntimeError, match="CoFDA"):
        torch.ops._C.mma_emu_moe_mm(
            out,
            inp["a"],
            inp["b"],
            inp["a_scales"],
            inp["b_scales"],
            inp["expert_offsets"],
            inp["problem_sizes"],
            inp["ab_strides"],
            inp["ab_strides"],
            inp["c_strides"],
            False,
            False,
            GDFS,
            13,
            32,
        )
