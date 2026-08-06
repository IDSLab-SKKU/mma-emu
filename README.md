<!-- markdownlint-disable MD001 MD041 -->

# mma-emu

A fork of [vLLM](https://github.com/vllm-project/vllm) that can run a model
end to end under an **arbitrary MMA accumulation configuration**.

Tensor cores are fixed function. Their accumulation precision and their order
of operations cannot be changed from software, so the arithmetic that actually
produces a GEMM result differs between architectures in ways you cannot
observe, let alone vary. This fork reproduces that arithmetic on CUDA cores
instead, which turns the accumulation algorithm and its bitwidths into
parameters you set at run time.

Based on the paper *"Not All Dot Products Are Equal: The Hidden MMA Arithmetic
Design Space Drives Cross-Architecture LLM Inference Gaps"*.

## What it does

```bash
vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --linear-backend mma_emu \
  --kernel-config '{"mma_emu": {"algorithm": "cofda", "f_bits": 13, "chunk_size": 16}}'
```

That runs an FP8 model whose linear layers accumulate the way an Ada tensor
core does, on whatever GPU you happen to own. Change `f_bits` and the
arithmetic changes with it.

The emulation is bit-accurate. Configured to match the GPU it runs on, it
reproduces the native CUTLASS result exactly:

| | sampled token ids |
| --- | --- |
| `cutlass_scaled_mm` on SM120 | `[12366, 11, 902, 374, 1101, ...]` |
| emulated, CoFDA `F=25 CS=32` | `[12366, 11, 902, 374, 1101, ...]` |
| emulated, CoFDA `F=7 CS=16` | `[12366, 13, 1102, 374, 279, ...]` |

The third row is the point: at a narrower accumulator the model's output
diverges, and you can watch it happen.

## Accumulation algorithms

| Algorithm | Description | Parameters |
| --- | --- | --- |
| **FDA** (Fused-Dot-Add) | Aligns all `K` partial products to the max exponent, truncates to `F` fractional bits (round-to-zero), sums in fixed point | `F` |
| **CoFDA** (Chain-of-FDA) | Chains FDA over chunks of `CS` products. `CS = K` reduces to FDA | `F`, `CS` |
| **GDFS** (Group-Dot-Fused-Sum) | Two-level: intra-group accumulation at `G` bits over groups of `GS`, then inter-group accumulation at `F` bits | `F`, `G`, `GS` |

The running accumulator is C-fused: it enters the reduced-precision datapath
alongside the products.

## Supported

| Format | Algorithms | Scales | Notes |
| --- | --- | --- | --- |
| FP8 E4M3 | GDFS, CoFDA | per-tensor | `CS` ∈ {16, 32}, `GS` ∈ {8, 16} |
| NVFP4 E2M1 | GDFS, CoFDA | UE4M3 block, size 16 | `CS` and `GS` fixed at 16 |

Dense linear layers only; MoE is not covered. Everything else in vLLM behaves
as upstream does.

### Parameter ranges

| | Range | |
| --- | --- | --- |
| `F` | 7 – 35 | |
| `G` (FP8) | 3 – 32 | 32 is the lossless width for FP8 E4M3 products |
| `G` (NVFP4) | 3 – 6 | 6 is the lossless width for E2M1 products |

`F` and `G` are kernel arguments rather than template parameters, so the whole
range is available without rebuilding. `CS` and `GS` size a register array and
so stay compile-time, which is why they are enumerated rather than ranged.

The kernels are the authority on what is accepted:
`torch.ops._C.mma_emu_config_error(...)` returns an empty string for a valid
configuration and an explanation otherwise.

## Configuration

The accumulation lives under `mma_emu` in `--kernel-config`:

| Field | Values | |
| --- | --- | --- |
| `algorithm` | `gdfs`, `cofda` | Unset means the emulation kernels decline every layer, so the native path runs |
| `f_bits` | 7 – 35 | |
| `g_bits` | 3 – 32 / 3 – 6 | GDFS only |
| `chunk_size` | 16, 32 | FP8 CoFDA only |
| `group_size` | 8, 16 | FP8 GDFS only |

Selection is separate from configuration on purpose. `--linear-backend mma_emu`
is what you ask for once, while the bitwidths describe the architecture being
emulated and change from run to run over a fixed model.

Because it is configuration rather than environment, a sweep is a plain loop in
one process, and each setting reaches the compilation cache key — two runs that
compute different numbers do not share compiled artifacts:

```python
for f in (7, 13, 21, 25, 35):
    llm = LLM(model=MODEL, kernel_config={
        "linear_backend": "mma_emu",
        "mma_emu": {"algorithm": "cofda", "f_bits": f, "chunk_size": 32},
    })
    ...
    del llm
```

## Building

The emulation kernels run on CUDA cores, so they need no CUTLASS and no
architecture-specific MMA instruction — only an architecture that supports the
element type. FP8 needs SM89; NVFP4 needs SM100 and CUDA 12.8.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
uv pip install --python .venv/bin/python -r requirements/common.txt
uv pip install --python .venv/bin/python -e . --no-deps --no-build-isolation

tools/build_mma_emu.sh
```

**Use `tools/build_mma_emu.sh` rather than `pip install -e .` on its own.**
The `dependencies` field in `pyproject.toml` is dynamic, so any build that
resolves dependencies reads `requirements/cuda.txt` and replaces your torch
with a different CUDA build.

### Environment caveats

These are properties of the machine this fork was developed on — an SM120 GPU
on a driver capped at CUDA 12.8 — rather than of the fork itself, but they are
the difference between a working install and a confusing one.

**On CUDA 13 these no longer apply, and one of them is actively harmful.** See
[CUDA13_MIGRATION.md](CUDA13_MIGRATION.md) before installing on a newer
toolchain.

**CUDA 12.x with a pre-580 driver.** torch 2.13.0 has no cu128 build, and its
cu130 and cu132 builds need driver ≥ 580. On an older driver, cu129 is the only
option. `torch.utils.cpp_extension` refuses a CUDA major-version mismatch
between nvcc and torch, so the toolkit has to be CUDA 12.x as well.
`tools/build_mma_emu.sh` checks this before building.

**Triton needs a matching ptxas.** For SM100 and above, Triton uses a separate
`ptxas-blackwell` binary that ships as CUDA 13.1 and emits PTX ISA 9.1, which a
CUDA 12.x driver cannot load — every Triton kernel then fails with
`device kernel image is invalid`. Point it at the CUDA 12 ptxas:

```bash
export TRITON_PTXAS_BLACKWELL_PATH=$PWD/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
rm -rf ~/.triton/cache   # otherwise the previous cubins are reused and it still fails
```

**FlashInfer.** If you skipped installing it, set `VLLM_USE_FLASHINFER_SAMPLER=0`.

## Testing

```bash
pytest tests/kernels/quantization/test_mma_emu.py
```

The tests detect the GPU, look up the accumulation its tensor cores implement,
and require the emulation at that configuration to reproduce
`cutlass_scaled_mm` and `cutlass_scaled_fp4_mm` bit for bit over a sweep of
matrix shapes.

They need a GPU and so are not run in CI, which lints only.

**Only SM120 has been checked directly.** The SM89, SM90 and SM100 entries in
`ARCH_ACCUMULATION` come from the paper. A failure on those architectures may
mean the recorded configuration is wrong rather than that the emulation is.

## Relationship to upstream vLLM

This is a research fork, not a competing distribution. It tracks upstream at
commit `43c4bdcae` and changes as little of it as possible: the emulation lives
in new files under `csrc/libtorch_stable/quantization/mma_emu/`, and six
upstream files are touched, to register the kernels and their configuration.

Issues and pull requests about vLLM itself belong at
[vllm-project/vllm](https://github.com/vllm-project/vllm), not here.

## License

Apache-2.0, as vLLM is. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
