<!-- markdownlint-disable MD001 MD041 -->

# NADPE

Not All Dot Products Are Equal — a fork of
[vLLM](https://github.com/vllm-project/vllm) that can run a model end to end
under an **arbitrary MMA accumulation configuration**.

NADPE is the project; `mma_emu` is the backend it adds to vLLM, named for what
it does — emulate MMA accumulation.

Tensor cores are fixed function. Their accumulation precision and their order
of operations cannot be changed from software, so the arithmetic that actually
produces a GEMM result differs between architectures in ways you cannot
observe, let alone vary. This fork reproduces that arithmetic on CUDA cores
instead, which turns the accumulation algorithm and its bitwidths into
parameters you set at run time.

Correctness is defined against the native path: configured to match the GPU it
runs on, the emulation reproduces `cutlass_scaled_mm` (FP8) and
`cutlass_scaled_fp4_mm` (NVFP4) bit for bit.

The emulation runs on CUDA cores rather than tensor cores, and is slower than
the native path by a wide margin. This is an instrument for studying the
arithmetic, not a way to serve a model.

Based on the paper *"Not All Dot Products Are Equal: The Hidden MMA Arithmetic
Design Space Drives Cross-Architecture LLM Inference Gaps"*, accepted at
**MICRO 2026**.

- **Start here** — [Quick start](#quick-start) · [Building](#building) · [Examples](#examples)
- **The design space** — [Accumulation algorithms](#accumulation-algorithms) · [Architecture accumulation](#architecture-accumulation) · [Supported](#supported) · [Configuration](#configuration)
- **The fork** — [Testing](#testing) · [Relationship to upstream vLLM](#relationship-to-upstream-vllm) · [Citation](#citation) · [License](#license)

## Quick start

There is no wheel. The emulation is CUDA and C++, so it has to be compiled:
[Building](#building) has the sequence and says what each flag is for. Expect
~30 minutes.

Once built, serve a model whose linear layers accumulate the way an Ada tensor
core does — on whatever GPU you happen to own:

```bash
vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --linear-backend mma_emu \
  --kernel-config '{"mma_emu": {"algorithm": "cofda", "f_bits": 13, "chunk_size": 16}}'
```

Change `f_bits` and the arithmetic changes with it. The server is the ordinary
OpenAI-compatible one, so `vllm chat` will talk to it.

## Accumulation algorithms

| Algorithm | Description | Parameters |
| --- | --- | --- |
| **FDA** (Fused-Dot-Add) | Aligns all `K` partial products to the max exponent, truncates to `F` fractional bits (round-to-zero), sums in fixed point | `F` |
| **CoFDA** (Chain-of-FDA) | Chains FDA over chunks of `CS` products. `CS = K` reduces to FDA | `F`, `CS` |
| **GDFS** (Group-Dot-Fused-Sum) | Two-level: intra-group accumulation at `G` bits over groups of `GS`, then inter-group accumulation at `F` bits | `F`, `G`, `GS` |

The running accumulator is C-fused: it enters the reduced-precision datapath
alongside the products.

## Architecture accumulation

What each architecture's tensor cores do, as the configuration that reproduces
it:

| Architecture | FP8 E4M3 | NVFP4 E2M1 |
| --- | --- | --- |
| SM89 Ada | CoFDA `F=13 CS=16` | — |
| SM90 Hopper | CoFDA `F=13 CS=32` | — |
| SM100 Blackwell, data center | CoFDA `F=25 CS=32` | GDFS `F=35 G=6 GS=16` |
| SM120 Blackwell, workstation | CoFDA `F=25 CS=32` | GDFS `F=35 G=6 GS=16` |

All four were measured here rather than taken from the paper. The table lives in
[`mma_emu_arch.py`](tests/kernels/quantization/mma_emu_arch.py), which the tests
and their report header both read, so a run says which architecture it is
reproducing and on what evidence.

## Supported

| Layer | Format | Algorithms | Scales | Notes |
| --- | --- | --- | --- | --- |
| Dense linear | FP8 E4M3 | GDFS, CoFDA | per-tensor | `CS` ∈ {16, 32}, `GS` ∈ {8, 16} |
| Dense linear | NVFP4 E2M1 | GDFS, CoFDA | UE4M3 block, size 16 | `CS` and `GS` fixed at 16 |

Everything else in vLLM behaves as upstream does.

`F` is a kernel argument, so its whole range works without rebuilding. `CS`,
`GS` and FP8's `G` select template instantiations, so they are enumerated;
NVFP4's `G` stays a range. The authority is
[`design_space.cuh`](csrc/libtorch_stable/quantization/mma_emu/design_space.cuh),
which `torch.ops._C.mma_emu_config_error(...)` applies at run time, returning an
empty string when the configuration is valid.

Operands must be contiguous — the operators index with a packed stride, so a
sliced view is read as dense and silently returns a wrong product, as
`cutlass_scaled_mm` also does. Nothing on vLLM's own path passes one.

## Configuration

The accumulation lives under `mma_emu` in `--kernel-config`:

| Field | Values | |
| --- | --- | --- |
| `algorithm` | `gdfs`, `cofda` | Unset means the emulation kernels decline every layer, so the native path runs |
| `f_bits` | 7 – 35 | |
| `g_bits` | FP8: 3, 4, 5, 6, 32 — NVFP4: 3 – 6 | GDFS only. FP8's is an enumerated set, not a range |
| `chunk_size` | 16, 32 | FP8 CoFDA only |
| `group_size` | 8, 16 | FP8 GDFS only |

Selection is separate from configuration on purpose: `--linear-backend mma_emu`
is asked for once, while the bitwidths describe the architecture being emulated
and change from run to run over a fixed model.

> **Name the baseline explicitly.** `--linear-backend auto` does not mean
> CUTLASS. The FP8 candidate list runs MMA-Emu, Marlin, FlashInfer, CUTLASS,
> torch; the emulation declines only because no algorithm is configured, and
> FlashInfer sits ahead of CUTLASS and does not decline, so it takes the layer
> wherever it is installed — and `requirements/cuda.txt` installs it. Comparing
> against `cutlass_scaled_mm` means saying `--linear-backend cutlass`.

## Examples

[`NADPE_examples/`](NADPE_examples/) runs a model end to end under an emulated
accumulation. Each has its own README.

| | What it does | Needs |
| --- | --- | --- |
| [`chat/`](NADPE_examples/chat/) | Serves one accumulation and prompts it by hand — the shortest way to watch a narrower accumulator change the tokens | one GPU |
| [`eval/`](NADPE_examples/eval/) | Scores a list of accumulations through lm-eval, so a divergence becomes a number | one GPU, hours |
| [`cross_arch/`](NADPE_examples/cross_arch/) | Emulates Hopper's accumulation on a Blackwell GPU and checks whether it reproduces the Hopper result bit for bit | an H100 (SM90) and a Blackwell GPU (SM100 or SM120) |

## Building

### Requirements

The fork branches from upstream `main` at `43c4bdcae`, 2026-08-04 — between the
v0.27 and v0.28 releases. vLLM cuts release branches, so no tag names that
commit. What upstream pins there:

| | Required | Stated in |
| --- | --- | --- |
| Python | 3.10 – 3.14 | `pyproject.toml`, `requires-python = ">=3.10,<3.15"` |
| PyTorch | **2.13.0**, exactly | `requirements/cuda.txt`, pinned with `==` |
| torchvision | 0.28.0 | `requirements/cuda.txt` |
| torchaudio | 2.11.0 | `requirements/cuda.txt` |
| CUDA | **13.0** | `VLLM_MAIN_CUDA_VERSION` in `vllm/envs.py`; `docker/Dockerfile` builds on 13.0.3 |
| Driver | ≥ 580 | what CUDA 13.0 requires |
| Architectures | SM89, SM90, SM100, SM120 | the four whose accumulation is reproduced; see [Architecture accumulation](#architecture-accumulation) |

### First build

Roughly 30 minutes at `MAX_JOBS=32`, nearly all of it compiling CUDA.

```bash
uv venv --python 3.12 .venv

# cu130, not cu132: that index carries no torchaudio 2.11.0
uv pip install --python .venv/bin/python \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

# the build backend, which --no-build-isolation below needs present already
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt

# the runtime set
uv pip install --python .venv/bin/python -r requirements/cuda.txt

# the test set. Both flags are load-bearing: arctic-inference pins torch==2.7.0
# in build-system.requires, which cu130 does not carry, and without the strategy
# uv stops at the first index holding *a* torch
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  -r requirements/test/cuda.txt

# --no-deps keeps the +cu130 torch: pyproject's dependencies are dynamic, so a
# resolving install silently replaces it with PyPI's default build. MAX_JOBS
# caps concurrency, or the heaviest CUTLASS units meet the OOM killer.
# Release, or setup.py falls back to RelWithDebInfo.
CMAKE_BUILD_TYPE=Release MAX_JOBS=32 uv pip install --python .venv/bin/python \
  -e . --no-deps --no-build-isolation
```

### Rebuilding

After editing a kernel:

```bash
CMAKE_BUILD_TYPE=Release MAX_JOBS=32 .venv/bin/python setup.py build_ext --inplace
```

## Testing

```bash
pytest tests/kernels/quantization/test_mma_emu.py
```

Sweeps matrix shapes and requires the emulation to reproduce the native CUTLASS
kernel bit for bit — `cutlass_scaled_mm` for FP8, `cutlass_scaled_fp4_mm` for
NVFP4 — at the accumulation the GPU under test implements. It goes through the
linear-kernel classes rather than the operators, so the weight preparation and
activation quantisation around the GEMM are covered with the arithmetic. Needs
a GPU, so CI does not run it.

## Relationship to upstream vLLM

This is a research fork, not a competing distribution. It tracks upstream `main`
at commit `43c4bdcae`, between the v0.27 and v0.28 releases, and changes as
little of it as possible: the emulation lives in new files under
`csrc/libtorch_stable/quantization/mma_emu/`, and nine upstream source files are
touched — six to register the kernels and their configuration, and three FP8
dispatches pinned so the native baseline holds still. Documentation and
repository configuration account for the rest; [NOTICE](NOTICE) lists every file
in each category.

Issues and pull requests about vLLM itself belong at
[vllm-project/vllm](https://github.com/vllm-project/vllm), not here.

## Citation

To be added.

## License

Apache-2.0, as vLLM is. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
