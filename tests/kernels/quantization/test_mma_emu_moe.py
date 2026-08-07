# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the grouped MoE MMA accumulation emulation kernel.

Run `pytest tests/kernels/quantization/test_mma_emu_moe.py`.

Separate from test_mma_emu.py because the MoE kernel is built for SM90 alone.
cutlass_moe_mm has SM90 and SM100 implementations and no SM120 one, so Hopper
is the only place the emulation can be compared against a native result, and a
kernel that cannot be checked should not ship enabled elsewhere.

These tests have never run. They were written on a machine that cannot build
the kernel, so treat a failure as a finding about the test as readily as about
the kernel.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("MMA-Emu kernels require CUDA", allow_module_level=True)

from tests.kernels.quantization.mma_emu_arch import COFDA, GDFS, SM  # noqa: E402

OUT_DTYPES = [torch.bfloat16, torch.float16]

# The MoE kernel is CoFDA only, and Hopper is the only architecture it is built
# for, so its accumulation is stated here rather than looked up per
# architecture the way test_mma_emu.py does.
SM90_COFDA = {"algorithm": COFDA, "f_bits": 13, "chunk_size": 32}

only_sm90 = pytest.mark.skipif(
    SM != 90,
    reason=f"the MoE emulation kernel is built for SM90 only, this is SM{SM}",
)
needs_kernel = pytest.mark.skipif(
    not hasattr(torch.ops._C, "mma_emu_moe_mm"),
    reason="vLLM was not built with the MMA-Emu MoE kernel",
)


def assert_bit_exact(native: torch.Tensor, emulated: torch.Tensor, config: dict):
    if torch.equal(native, emulated):
        return
    diff = (native.float() - emulated.float()).abs()
    mismatched = int((native != emulated).sum())
    pytest.fail(
        f"emulation at {config} does not reproduce the native result on "
        f"SM{SM}: {mismatched}/{native.numel()} elements differ, "
        f"max|d|={diff.max():.3e}, mean|d|={diff.mean():.3e}"
    )


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


@only_sm90
@needs_kernel
@pytest.mark.parametrize("num_experts,m_per_expert,n,k", MOE_CASES)
@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
def test_moe_matches_native(
    num_experts: int, m_per_expert: int, n: int, k: int, out_dtype
):
    """Grouped MoE emulation at SM90's accumulation must match cutlass_moe_mm.

    CoFDA only; the MoE kernel does not implement GDFS.
    """
    config = SM90_COFDA
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


@only_sm90
@needs_kernel
def test_moe_narrower_accumulator_diverges():
    """A narrow accumulator must change the result.

    Without this, test_moe_matches_native would also pass if the operator
    silently ignored the configuration and called the native path.
    """
    config = SM90_COFDA
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


@only_sm90
@needs_kernel
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
