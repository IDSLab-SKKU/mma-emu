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

The server is the ordinary OpenAI-compatible one, so anything that speaks that
API will do. The shortest way to put a prompt through the emulation is the
client shipped alongside it, which defaults to `http://localhost:8000/v1`:

```bash
vllm chat            # --url if the server is not on the default address
```

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

| Layer | Format | Algorithms | Scales | Notes |
| --- | --- | --- | --- | --- |
| Dense linear | FP8 E4M3 | GDFS, CoFDA | per-tensor | `CS` ∈ {16, 32}, `GS` ∈ {8, 16} |
| Dense linear | NVFP4 E2M1 | GDFS, CoFDA | UE4M3 block, size 16 | `CS` and `GS` fixed at 16 |
| Grouped MoE | FP8 E4M3 | CoFDA | per-tensor, per-expert | **Hopper only, and not yet verified** |

Everything else in vLLM behaves as upstream does.

The operators require contiguous operands. They index with a packed stride, so
a sliced view would be read as though it were dense; `cutlass_scaled_mm` has the
same limitation and returns a wrong product instead of saying so. Nothing on
vLLM's own path passes one.

**The MoE kernel is provisional.** It builds for SM90 alone, because the check
that makes the dense kernels trustworthy is not available elsewhere:
`cutlass_moe_mm` has SM90 and SM100 implementations and no SM120 one, so on a
consumer Blackwell card there is nothing to compare against. The kernel
compiles and its arithmetic is the dense CoFDA path with an expert dimension
added, but **no bit-exactness run has been done**. Treat it as unverified until
one has. There is no Python integration for it yet either — the operator exists,
nothing selects it.

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

`--linear-backend` takes any of vLLM's backends, and naming one is how you get
the baseline the emulation is supposed to reproduce:

```bash
vllm serve nvidia/Llama-3.1-8B-Instruct-FP8 --linear-backend cutlass
```

Ask for it explicitly rather than dropping the flag. Leaving it at `auto` does
not mean CUTLASS: the FP8 candidate list runs MMA-Emu, Marlin, FlashInfer,
CUTLASS, torch, and the emulation only declines because no algorithm is
configured. FlashInfer sits ahead of CUTLASS and does not decline, so it takes
the layer wherever it is installed — and `requirements/cuda.txt` installs it.
Comparing against `cutlass_scaled_mm` means saying `cutlass`.

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

### Requirements

The fork tracks upstream at `43c4bdcae`, which pins:

| | Required | Stated in |
| --- | --- | --- |
| Python | 3.10 – 3.14 | `pyproject.toml`, `requires-python = ">=3.10,<3.15"` |
| PyTorch | **2.13.0**, exactly | `requirements/cuda.txt`, pinned with `==` |
| torchvision | 0.28.0 | `requirements/cuda.txt` |
| torchaudio | 2.11.0 | `requirements/cuda.txt` |
| CUDA | **13.0** | `VLLM_MAIN_CUDA_VERSION` in `vllm/envs.py`; `docker/Dockerfile` builds on 13.0.3 |
| Driver | ≥ 580 | what CUDA 13.0 requires |

The emulation kernels themselves run on CUDA cores, so they need no CUTLASS and
no architecture-specific MMA instruction — only an architecture that supports
the element type. FP8 needs SM89; NVFP4 needs SM100 and CUDA 12.8.

### Build

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
uv pip install --python .venv/bin/python -r requirements/cuda.txt
uv pip install --python .venv/bin/python pytest tblib
CMAKE_BUILD_TYPE=Release MAX_JOBS=32 uv pip install --python .venv/bin/python \
  -e . --no-deps --no-build-isolation
```

Use `cu130` and not `cu132`, which carries torch 2.13.0 and torchvision 0.28.0
but no torchaudio 2.11.0.

`requirements/build/cuda.txt` is the build backend — cmake, ninja, setuptools —
which `--no-build-isolation` needs present already; `requirements/cuda.txt` is
the runtime set. `pytest` and `tblib` are all the kernel tests need on top of
it: `tests/conftest.py` wants transformers and Pillow too, but the runtime set
has already brought those.

Not `requirements/test/cuda.txt`. That is upstream's whole CI matrix, and one
of its entries — `arctic-inference`, for a suffix-decoding test unrelated to
anything here — is an sdist whose build requires `torch==2.7.0`, a version the
cu130 index does not carry. Nothing in it is needed to exercise the emulation.

Nothing on the last line is decoration:

| | If you drop it |
| --- | --- |
| `--no-deps` | `dependencies` in `pyproject.toml` is dynamic, so the install resolves `requirements/cuda.txt`, sees `torch==2.13.0` with no local version, and replaces the `+cu130` wheel with PyPI's default build |
| `--no-build-isolation` | the build runs in a throwaway environment against a different torch |
| `MAX_JOBS=32` | `setup.py` uses every core; the heaviest CUTLASS units take several GB each, and the build dies to the OOM killer rather than to an error you can read |
| `CMAKE_BUILD_TYPE=Release` | `setup.py` falls back to `RelWithDebInfo`, carrying debug info through every CUDA unit — slower to compile, and a far larger `.so` |

The target architecture comes from the GPU in the machine. Building for another
one — the grouped MoE kernel is Hopper-only — means saying so:

```bash
TORCH_CUDA_ARCH_LIST=9.0 CMAKE_BUILD_TYPE=Release MAX_JOBS=32 \
  uv pip install --python .venv/bin/python -e . --no-deps --no-build-isolation
```

After editing a kernel, compile in place rather than reinstalling:

```bash
CMAKE_BUILD_TYPE=Release MAX_JOBS=32 .venv/bin/python setup.py build_ext --inplace
```

Keep that environment identical between runs. CMake reconfigures when any of it
moves, which turns an incremental rebuild into a full one — the same reason not
to `rm -rf build` out of habit. To narrow it further, the `mma_emu` sources
compile into a single extension:

```bash
BUILD=$(echo build/temp.*)
cmake --build "$BUILD" -j 32 --target _C_stable_libtorch
cmake --install "$BUILD" --component _C_stable_libtorch --prefix "$PWD"
```

### Build check

The configure step names the kernels it is building:

```text
-- Building mma_emu_scaled_fp8_mm for archs: 12.0
-- Building mma_emu_scaled_nvfp4_mm for archs: 12.0
-- Not building mma_emu_moe_mm: Hopper (9.0) only, and it is not in CUDA target architectures
```

Then, in order of what each rules out:

```bash
# the wheel that was installed is the one that survived
.venv/bin/python -c "import torch; print(torch.__version__)"        # 2.13.0+cu130

# the operators registered
.venv/bin/python -c "import torch, vllm._C_stable_libtorch; print([
    o for o in ('mma_emu_scaled_fp8_mm', 'mma_emu_scaled_nvfp4_mm', 'mma_emu_moe_mm')
    if hasattr(torch.ops._C, o)])"

# twelve kernel instantiations, no more. They are device symbols, so they live
# in the fatbinary rather than in the ELF symbol table nm reads
cuobjdump --dump-elf-symbols vllm/_C_stable_libtorch.abi3.so \
  | grep -oE "mma_emu_scaled_[a-z0-9_]+kernel[a-zA-Z0-9_]*" | sort -u | wc -l
```

What settles it is the bit-exactness run below.

## Testing

`pytest` and `tblib`, installed as part of the build above, are all these need.

```bash
pytest tests/kernels/quantization/test_mma_emu.py \
       tests/kernels/quantization/test_mma_emu_selection.py \
       tests/kernels/quantization/test_mma_emu_moe.py
```

| File | What it checks | Needs |
| --- | --- | --- |
| `test_mma_emu.py` | The emulation reproduces the native kernel bit for bit, over a sweep of matrix shapes | a GPU |
| `test_mma_emu_selection.py` | The emulation is not selected unless asked for, and refuses what it cannot do | nothing |
| `test_mma_emu_moe.py` | The grouped MoE kernel against `cutlass_moe_mm` | an SM90 GPU |

The first goes through the linear-kernel classes rather than the operators, so
it covers the weight preparation and activation quantisation around the GEMM as
well as the arithmetic.

The second matters more than its size suggests. The emulation kernels sit at the
head of the candidate list, so the only thing keeping them off every layer is
that they decline when no algorithm is configured — and a kernel that wrongly
accepts still returns correct numbers, just far slower, so nothing would
announce it. Those tests need neither a GPU nor a built extension.

The run says which comparison it is about to make:

```text
MMA-Emu: SM120 (NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition)
  fp8    emulating CoFDA F=25 CS=32     compared bit-exactly against cutlass_scaled_mm
  nvfp4  emulating GDFS F=35 G=6 GS=16  compared bit-exactly against cutlass_scaled_fp4_mm
  moe    not built for SM120 — skipped
```

They need a GPU and so are not run in CI, which lints only.

**Only SM120 has been checked directly.** The SM89, SM90 and SM100 entries in
`ARCH_ACCUMULATION`, in `tests/kernels/quantization/mma_emu_arch.py`, come from
the paper. A failure on those architectures may mean the recorded configuration
is wrong rather than that the emulation is, and the header says so when it is
running on one of them.

## Relationship to upstream vLLM

This is a research fork, not a competing distribution. It tracks upstream at
commit `43c4bdcae` and changes as little of it as possible: the emulation lives
in new files under `csrc/libtorch_stable/quantization/mma_emu/`, and six
upstream files are touched, to register the kernels and their configuration.

Issues and pull requests about vLLM itself belong at
[vllm-project/vllm](https://github.com/vllm-project/vllm), not here.

## License

Apache-2.0, as vLLM is. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
