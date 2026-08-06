# Moving this fork to CUDA 13

This fork was developed on a machine capped at **CUDA 12.8** by its driver
(570.195.03; CUDA 13 needs ≥ 580). Upstream vLLM at the commit this fork tracks
(`43c4bdcae`) assumes CUDA 13, so several changes here exist **only** to work
around that gap, and several things were skipped or downgraded for the same
reason.

On a machine with CUDA ≥ 13.0 and a driver ≥ 580, those workarounds should be
undone. This document lists every one of them, why it exists, and how to tell
whether the revert worked.

**Read this before changing anything else.** Some of these workarounds are
harmless on CUDA 13 and some actively make things worse — the Triton one in
particular downgrades a working toolchain.

---

## Summary

| # | What | Action on CUDA 13 | Severity if left |
| --- | --- | --- | --- |
| 1 | torch built for cu129 | Reinstall for cu130/cu132 | Blocks 2, 3, 5 |
| 2 | `CMakeLists.txt` gates cooperative topk on CUDA ≥ 13 | **Revert commit `8517c1675`** | Loses an upstream feature |
| 3 | `[cu13]` requirement pins skipped | Install them | Missing FA4, humming kernels |
| 4 | `TRITON_PTXAS_BLACKWELL_PATH` set to CUDA 12 ptxas | **Unset it** | Downgrades Triton codegen |
| 5 | FlashInfer not installed, `VLLM_USE_FLASHINFER_SAMPLER=0` | Install it, drop the flag | Slower sampling |
| 6 | `torchvision`/`torchaudio` from the cu129 index | Reinstall from the matching index | Import failures |
| 7 | mma_emu arch gates use plain arch strings | Verify against CUDA 13 family targets | Possibly no kernels built |

Items 2 and 4 are the ones that matter most. Everything else is ordinary
dependency work.

---

## 1. torch was pinned to cu129

**Why.** `requirements/cuda.txt` asks for `torch==2.13.0`, which has **no cu128
build** — cu128 stops at torch 2.9.1. The cu130 and cu132 builds need driver
≥ 580. On driver 570 the only option was cu129.

`torch.utils.cpp_extension` (`_check_cuda_version`) raises a `RuntimeError` when
nvcc's CUDA **major** version differs from torch's, so the toolkit had to stay
on CUDA 12 as well. A minor mismatch — nvcc 12.8 against torch cu12.9 — is only
a warning, which is why that combination worked.

**On CUDA 13.** Install torch the way upstream intends:

```bash
uv pip install --python .venv/bin/python torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130   # or cu132
```

`tools/build_mma_emu.sh` preflights the nvcc/torch major versions and will
refuse to build on a mismatch, so a half-migrated environment fails early and
says why.

**Also update** the install instructions in `README.md`, which currently name
cu129, and the environment-caveats section, which explains a constraint that no
longer applies.

---

## 2. `CMakeLists.txt` gates cooperative topk on CUDA ≥ 13 — revert this

**This is the one change in this fork that exists purely because of CUDA 12.**

**Why.** `csrc/libtorch_stable/cooperative_topk.cuh` calls

- `cuda::ptx::mbarrier_try_wait_parity`
- `cuda::ptx::mbarrier_arrive_expect_tx`
- `cuda::ptx::cp_async_bulk`

through their **`sem_relaxed`** overloads. The CUDA 12.x libcu++ provides only
the `sem_acquire` / `sem_release` forms, so the file fails to compile on CUDA
12 for **every** architecture:

```text
error: no instance of overloaded function "cuda::ptx::mbarrier_try_wait_parity"
       matches the argument list
```

Upstream's guard admits CUDA ≥ 12.0 and its `else()` branch enables the feature
for `9.0a;10.0a;10.1a;10.3a;12.0a;12.1a`, so an **unmodified** upstream tree
cannot build on CUDA 12.x. That is an upstream bug, not something this fork
introduced. The patch raises the outer guard to 13.0 and deletes the dead
`else()` branch, in the same idiom upstream uses for its CUDA ≥ 13
`--compress-mode=size` block.

**On CUDA 13.** Revert it. The feature compiles and the fork should not carry a
gratuitous difference:

```bash
git revert 8517c1675          # [Build] Gate cooperative topk on CUDA >= 13.0
```

Then confirm the file actually builds rather than being skipped:

```bash
tools/build_mma_emu.sh 2>&1 | grep cooperative_topk
```

**If the revert is kept instead** (for example, to keep supporting CUDA 12
users), leave the comment in place explaining why, and consider reporting the
bug upstream — it affects every CUDA 12.x user, not only this fork. As of
2026-08-06 no upstream issue had been checked for, because `gh` was not
available on the development machine.

---

## 3. `[cu13]` requirement pins were skipped

**Why.** `requirements/cuda.txt` pins

```text
nvidia-cutlass-dsl[cu13]==4.6.0
humming-kernels[cu13]==0.1.10
```

Both want CUDA 13 runtime libraries. Neither is on the MMA emulation path —
`nvidia-cutlass-dsl` is for FA4 attention and `humming-kernels` for
quantization GEMM — so they were left out and the environment was built from
`requirements/common.txt` plus torch directly.

**On CUDA 13.** Install the requirements file as upstream intends:

```bash
uv pip install --python .venv/bin/python -r requirements/cuda.txt
```

Watch that this does **not** replace torch: the file pins `torch==2.13.0`
without a local version, so a resolver may pull the default PyPI build over the
one you installed deliberately. Check afterwards:

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```

---

## 4. `TRITON_PTXAS_BLACKWELL_PATH` — unset it

**This workaround makes a CUDA 13 system worse. Remove it first.**

**Why.** Triton 3.7.1 (shipped with torch 2.13.0) uses a **separate ptxas
binary for SM100 and above**:

```python
# triton/backends/nvidia/compiler.py
def get_ptxas(arch: int) -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas_blackwell if arch >= 100 else knobs.nvidia.ptxas
```

| binary | version |
| --- | --- |
| `triton/backends/nvidia/bin/ptxas` | 12.8 |
| `triton/backends/nvidia/bin/ptxas-blackwell` | **13.1** |

On SM120 that produced `.version 9.1` PTX (CUDA 13.1 ISA), which a CUDA 12.x
driver cannot load. **Every** Triton kernel then failed, including a trivial
element-wise add:

```text
RuntimeError: Triton Error [CUDA]: device kernel image is invalid
```

This is easy to misdiagnose. The ordinary `ptxas` next to it *is* 12.8, so
checking only that one hides the cause. The failure also surfaces deep inside
vLLM — the first Triton kernel to run is the sampler's
`_gumbel_sample_kernel` — which makes it look like a vLLM bug.

The workaround forces Triton to use the CUDA 12 ptxas, producing `.version 8.7`:

```bash
export TRITON_PTXAS_BLACKWELL_PATH=$PWD/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
rm -rf ~/.triton/cache   # required: cached cubins are reused otherwise
```

**On CUDA 13.** Do not set the variable. Letting Triton use its own
`ptxas-blackwell` is correct there, and forcing the 12.8 binary would give up
whatever the newer ISA provides.

Clear the cache once after migrating, since it may hold cubins built with the
downgraded ptxas:

```bash
rm -rf ~/.triton/cache
```

**Also remove** the corresponding section from `README.md` (Environment
caveats) and `AGENTS.md` (Environment), and drop it from the test and build
commands documented there.

---

## 5. FlashInfer was not installed

**Why.** It was excluded along with the rest of `requirements/cuda.txt`
(item 3). Without it, engine startup fails:

```text
ModuleNotFoundError: No module named 'flashinfer'
  ... vllm/v1/sample/ops/topk_topp_sampler.py, flashinfer_sampler_supported()
```

so every run set `VLLM_USE_FLASHINFER_SAMPLER=0`.

**On CUDA 13.** Installing `requirements/cuda.txt` brings it in, and the flag
can go. Remove it from `README.md` and from any scripts that set it.

---

## 6. `torchvision` and `torchaudio` came from the cu129 index

**Why.** The model registry imports `torchvision`
(`vllm/transformers_utils/processors/minimax_m3.py`), so it is needed even for
a text-only model. They were installed to match the cu129 torch:

```bash
uv pip install torchvision==0.28.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

**On CUDA 13.** Reinstall from the index matching the new torch, or let
`requirements/cuda.txt` handle it.

---

## 7. Verify the mma_emu architecture gates

**Why this needs checking rather than reverting.** The gates added for the
emulation kernels use plain architecture strings:

```cmake
cuda_archs_loose_intersection(MMA_EMU_FP8_ARCHS "8.9;9.0;10.0;12.0" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MMA_EMU_NVFP4_ARCHS "10.0;12.0" "${CUDA_ARCHS}")
```

Upstream switches to **family-specific targets** on CUDA ≥ 13 — `10.0f`,
`12.0f` and so on — and its `CUDA_SUPPORTED_ARCHS` differs by CUDA version.
Whether `cuda_archs_loose_intersection` still matches plain `12.0` against a
CUDA 13 arch list was never tested here.

**On CUDA 13.** Confirm the kernels are actually built. The configure step
prints one line per format:

```text
-- Building mma_emu_scaled_fp8_mm for archs: ...
-- Building mma_emu_scaled_nvfp4_mm for archs: ...
```

If either says *"as no compatible archs found"*, the gate needs the
family-suffixed form. Follow what the neighbouring `SCALED_MM_ARCHS` block
does, which already branches on CUDA 13.

The emulation kernels themselves need no architecture-specific MMA instruction
— they run on CUDA cores — so only the element types constrain them: SM89 for
FP8 E4M3, SM100 and CUDA 12.8 for the NVFP4 packed types.

---

## After migrating: what to check

```bash
# 1. Toolchain agrees with itself
tools/build_mma_emu.sh          # preflight prints torch, nvcc and arch

# 2. Kernels are bit-exact against native CUTLASS on this GPU
pytest tests/kernels/quantization/test_mma_emu.py

# 3. Twelve instantiations, no more
nm -DC --defined-only vllm/_C_stable_libtorch.abi3.so \
  | grep -coE "vllm::mma_emu::mma_emu_scaled_[a-z0-9_]+kernel<[^>]*>"

# 4. End to end, with and without emulation
#    At the architecture's own accumulation the token ids must match native;
#    at a narrow accumulator they must not.
```

Step 2 is the one that matters. It detects the GPU, looks its accumulation up
in `ARCH_ACCUMULATION`, and requires the emulation at that configuration to
reproduce `cutlass_scaled_mm` and `cutlass_scaled_fp4_mm` exactly.

### An opportunity while you are there

Only **SM120** has been verified directly. The SM89, SM90 and SM100 entries in
`tests/kernels/quantization/test_mma_emu.py` come from the paper and have never
been run. A rented instance is a chance to close that gap:

| GPU | SM | Entry to verify |
| --- | --- | --- |
| L40S, RTX 4090 | 89 | CoFDA `F=13 CS=16` |
| H100, H200 | 90 | CoFDA `F=13 CS=32` (FDA expressed as a chunk spanning the K tile) |
| B200 | 100 | CoFDA `F=25 CS=32`, GDFS `F=35 G=6` |

If a test fails on one of these, that is **information, not a defect** — it
means the accumulation that architecture implements differs from what the table
records. Update the table and say so in the commit message.

The SM90 entry deserves particular suspicion. FDA sums the whole `K` dimension
as one chunk, and it is expressed here as CoFDA with `CS=32` because that is
the K tile the kernel uses. Those are equivalent only if the hardware's chunk
really is the tile width.

---

## Reverting in one pass

```bash
# 2. the only CUDA-12-specific code change
git revert 8517c1675

# 1, 3, 5, 6. dependencies
uv pip install --python .venv/bin/python torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements/cuda.txt
.venv/bin/python -c "import torch; print(torch.__version__)"   # must still be cu130

# 4. the Triton workaround
unset TRITON_PTXAS_BLACKWELL_PATH
rm -rf ~/.triton/cache

# rebuild and check
tools/build_mma_emu.sh --clean
pytest tests/kernels/quantization/test_mma_emu.py
```

Then update the documentation that describes the CUDA 12 environment:
`README.md` (install commands and Environment caveats), `AGENTS.md`
(Environment), and delete this file once nothing in it applies.
