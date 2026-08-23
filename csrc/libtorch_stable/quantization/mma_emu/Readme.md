# MMA-Emu — Configurable MMA Accumulation Emulation Kernels

Bit-accurate emulation of the internal accumulation arithmetic of commercial
matrix multiply-accumulate (MMA) units, running on CUDA cores so that every
alignment and truncation step is under software control.

Tensor cores are fixed-function: their accumulation precision and their order of
operations cannot be changed from software. These kernels reproduce that
arithmetic on CUDA cores instead, which makes the accumulation algorithm and its
bitwidths configurable parameters. That is what allows an LLM to be run
end-to-end under an arbitrary MMA accumulation configuration.

Based on the paper *"Not All Dot Products Are Equal: The Hidden MMA Arithmetic
Design Space Drives Cross-Architecture LLM Inference Gaps"*.

## Accumulation algorithms

| Algorithm | Description | Parameters |
| --- | --- | --- |
| **FDA** (Fused-Dot-Add) | Aligns all `K` partial products to the max exponent, truncates to `F` fractional bits (round-to-zero), sums in fixed point | `F` |
| **CoFDA** (Chain-of-FDA) | Chains FDA over chunks of `CS` products. `CS = K` reduces to FDA | `F`, `CS` |
| **GDFS** (Group-Dot-Fused-Sum) | Two-level: intra-group accumulation at `G` bits over groups of `GS`, then inter-group accumulation at `F` bits | `F`, `G`, `GS` |

The running accumulator `C` is C-fused: it participates in the reduced-precision
datapath, aligned to the same max exponent and truncated to `F` fractional bits
alongside the `K` products.

## Supported configurations

| Layer | Format | Algorithms | Notes |
| --- | --- | --- | --- |
| Dense linear | FP8 E4M3 | GDFS, CoFDA | Per-tensor scales |
| Dense linear | NVFP4 E2M1 | GDFS, CoFDA | UE4M3 block scales, block size 16, alpha epilogue. `CS` and `GS` are fixed at 16 |

Each has a native tensor-core counterpart, which is also the correctness
reference: `cutlass_scaled_mm` and `cutlass_scaled_fp4_mm`. The emulation
configured to match the GPU it runs on should reproduce that result bit for
bit.

## Files

Layered, with dependencies flowing strictly in one direction. The directory is
flat, so the layer a file belongs to is recorded here and in its header comment
rather than in the path.

| Layer | Files | Depends on |
| --- | --- | --- |
| **Core** — format-agnostic accumulation arithmetic | `types.cuh` · `fp32_utils.cuh` · `accumulator.cuh` · `gdfs_group.cuh` · `design_space.cuh` · `tiling.cuh` | — |
| **Formats** — per-format element and scale arithmetic | `fp8_e4m3.cuh` · `fp4_e2m1.cuh` · `nvfp4_ue4m3.cuh` · `scale_swizzle.cuh` | Core |
| **GEMM** — kernels | `mma_emu_scaled_fp8_mm_kernels.cuh` · `mma_emu_scaled_nvfp4_mm_kernels.cuh` | Core, Formats |
| **Entry** — torch operators | `mma_emu_scaled_fp8_mm_entry.cu` · `mma_emu_scaled_nvfp4_mm_entry.cu` | GEMM |

`mma_emu_config_check.h` sits beside these rather than in a layer. It is
host-only, and both the operator entry points and the Python kernel selector go
through it, so which configurations are accepted — and the wording that reports
a rejection — is stated once.

Nothing in Core or Formats depends on torch; the torch boundary is confined to
the Entry files. The GEMM headers take raw pointers, dimensions and a stream.

`design_space.cuh` is the single source of truth for which `F`, `G`, `CS` and
`GS` values the kernels accept. Every dispatch table and every validation check
derives from it; nothing restates those values.

## Runtime versus compile-time parameters

`F` and `G` only feed shift amounts and truncation masks, so they are kernel
arguments and their whole accepted range is available without rebuilding. `CS`
and `GS` size a register array, so they must be known at compile time and are
enumerated instead.

That split is why the dense kernels need twelve instantiations rather than
hundreds. When adding a parameter, decide first which kind it is.

## Torch operators

| Operator | Arch | Built when |
| --- | --- | --- |
| `mma_emu_scaled_fp8_mm` | SM89+ | `VLLM_ENABLE_MMA_EMU_FP8` |
| `mma_emu_scaled_nvfp4_mm` | SM100+, CUDA 12.8+ | `VLLM_ENABLE_MMA_EMU_NVFP4` |
| `mma_emu_config_error` | any | `VLLM_ENABLE_MMA_EMU_FP8` |

The architecture gates are in `CMakeLists.txt`, next to the sources they select.
These are CUDA-core kernels, so they need no CUTLASS and no
architecture-specific MMA instruction; only the element type constrains them.

`mma_emu_config_error` takes no tensor, so it registers under
`CompositeExplicitAutograd` rather than `CUDA`. It returns an empty string for
an accepted configuration and an explanation otherwise.
