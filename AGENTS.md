# Agent Instructions for mma-emu

> These instructions apply to **all** AI-assisted contributions to
> `IDSLab-SKKU/mma-emu`.

This is a research fork of [vLLM](https://github.com/vllm-project/vllm) that
emulates MMA accumulation arithmetic on CUDA cores. Read [README.md](README.md)
first — it explains what the fork adds and, importantly, the environment
constraints that make the difference between a working install and a confusing
one.

**This is not `vllm-project/vllm`.** Changes to vLLM itself belong upstream.
Nothing here should be proposed to upstream except the fixes explicitly marked
as upstream bugs (see [Divergence](#divergence)).

---

## 1. Environment

**Never use system `python3` or bare `pip`/`pip install`.** All Python commands
go through `uv` and `.venv/bin/python`.

Follow the build instructions in [README.md](README.md). Two points bite hard
enough to repeat:

- **Build with `tools/build_mma_emu.sh`**, not `pip install -e .`. The
  `dependencies` field in `pyproject.toml` is dynamic, so a dependency-resolving
  build replaces torch with a different CUDA build and silently breaks the
  environment.
- **Set `TRITON_PTXAS_BLACKWELL_PATH`** on SM100+ with a CUDA 12.x driver, and
  clear `~/.triton/cache` when you do. Without it every Triton kernel fails with
  `device kernel image is invalid`, which looks like a vLLM bug and is not. On
  CUDA 13 do **not** set it — see [CUDA13_MIGRATION.md](CUDA13_MIGRATION.md),
  which lists every workaround that exists only because of CUDA 12.

## 2. Divergence

The value of this fork is that it stays close to upstream, so it can follow it.
Guard that.

**Prefer new files over edits to existing ones.** The emulation lives in
`csrc/libtorch_stable/quantization/mma_emu/` and in `mma_emu.py` modules under
`vllm/model_executor/kernels/linear/`. A new file carries no rebase risk. Only
registration touchpoints are real divergence, and there are currently six:

| File | Why |
| --- | --- |
| `CMakeLists.txt` | Adds the sources; also gates cooperative topk on CUDA ≥ 13 |
| `csrc/libtorch_stable/torch_bindings.cpp` | Operator schemas and implementations |
| `csrc/libtorch_stable/ops.h` | Operator declarations |
| `vllm/config/kernel.py` | `mma_emu` sub-config, and `mma_emu` in the `LinearBackend` literal |
| `vllm/_custom_ops.py` | Python wrappers |
| `vllm/model_executor/kernels/linear/__init__.py` | Kernel registration |

The cooperative topk change in `CMakeLists.txt` is an **upstream bug**, not a
fork-local need: the guard admits CUDA ≥ 12.0 while the source requires the
CUDA 13 CCCL, so an unmodified tree cannot build on CUDA 12.x. It is written in
upstream's own idiom so it can be submitted as-is.

**Keep upstream-file edits in their own commits**, separate from additions, so a
future rebase is a replay of a short labelled series rather than archaeology.

## 3. Single source of truth

`csrc/libtorch_stable/quantization/mma_emu/design_space.cuh` states which `F`,
`G`, `CS` and `GS` values the kernels accept. Nothing else restates them —
`mma_emu_config_check.h` phrases the rejection, the operator entry points use it
as a backstop, and Python reaches the same function through
`torch.ops._C.mma_emu_config_error`.

If you add a parameter, decide first whether it is a **value** (a shift amount
or a mask — make it a kernel argument, bounded by a range) or a **shape** (an
array extent or a type — it must stay a template parameter, enumerated). That
distinction is why there are twelve kernel instantiations rather than hundreds.

## 4. Tests

```bash
pytest tests/kernels/quantization/test_mma_emu.py
```

The bit-exactness tests need a GPU, so CI lints only. Run them locally before
proposing a change to the kernels.

When adding tests:

- **Test against the native path.** The emulation configured to match the GPU
  should reproduce `cutlass_scaled_mm` or `cutlass_scaled_fp4_mm` exactly. A
  test that only checks the emulation against itself proves little.
- **Make failures diagnostic.** Report how many elements differ and by how much,
  not just that they do.
- Extend `FP8_MNK` / `NVFP4_MNK` rather than writing a new sweep.

## 5. Style

- Match the surrounding code, which is vLLM's.
- Python line length 88. Google-style docstrings.
- Minimize comments; prefer legible code. When a comment is needed, say *why*,
  since the *what* is in the code.
- Run `pre-commit run --all-files` before proposing a change.

## 6. Commit messages

Explain why the change is what it is, not what the diff already shows. Add
attribution trailers:

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

## 7. Accountability

- Pure code-agent contributions are **not allowed**. A human must understand and
  defend the change end to end.
- The submitter must review every changed line and run the kernel tests.
- Descriptions of AI-assisted work must state that AI assistance was used, and
  must include the test results — for anything touching the kernels, the output
  of the bit-exactness tests on the hardware you ran them on.
