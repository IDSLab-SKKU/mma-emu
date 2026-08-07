# Contributing to mma-emu

This is a research fork of [vLLM](https://github.com/vllm-project/vllm). Start
with [README.md](README.md) for what it does and how to build it, and
[AGENTS.md](AGENTS.md) for the conventions — those apply to human contributors
too, not only to AI-assisted ones.

**Changes to vLLM itself belong upstream**, at
[vllm-project/vllm](https://github.com/vllm-project/vllm). This fork keeps its
distance from upstream deliberately so it can follow it; a change here that
could have been made there makes that harder. See the divergence table in
[AGENTS.md](AGENTS.md) for what is already modified and why.

**Install the git hooks before your first commit.** They run the same lint
and type checks CI does, on the files you touched:

```bash
uv pip install --python .venv/bin/python -r requirements/lint.txt
.venv/bin/pre-commit install
```

Two things worth knowing before you start:

- **Build with `tools/build_mma_emu.sh`.** A plain `pip install -e .` will
  replace your torch with a different CUDA build, because `dependencies` in
  `pyproject.toml` is dynamic.
- **Run the kernel tests before proposing a change to the kernels.** They need
  a GPU and so are not run in CI:

  ```bash
  pytest tests/kernels/quantization/test_mma_emu.py
  ```

  Include the results and the architecture you ran them on.
