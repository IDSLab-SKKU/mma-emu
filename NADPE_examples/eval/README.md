# Evaluating a list of accumulations

Evaluate the baseline on a task, confirm the emulation reproduces it, then
evaluate the accumulations no part implements. Its sibling
[`../chat/`](../chat/) serves one accumulation to prod by hand.

Run everything below from this directory, with the virtualenv active:

```bash
cd NADPE_examples/eval
source ../../.venv/bin/activate
```

## 1. Evaluate the baseline

```bash
python run.py configs/llama8b-fp8-gsm8k-native.yaml
```

`native` is `cutlass_scaled_mm`, the kernel the emulation is supposed to
reproduce; an NVFP4 checkpoint resolves the same word to
`cutlass_scaled_fp4_mm`.

## 2. Emulate the hardware

```bash
python run.py configs/llama8b-fp8-gsm8k-emulation.yaml
```

This runs one accumulation — the one your GPU already does in hardware.
**It must score exactly what step 1 scored**, since the emulation reproduces
the baseline rather than approximating it; a difference is a bug, not an
experiment.

| GPU | FP8 | NVFP4 | its MMA accumulation |
| --- | --- | --- | --- |
| Ada, SM89 | yes | — | CoFDA F=13 CS=16 |
| Hopper, SM90 | yes | — | CoFDA F=13 CS=32 |
| Blackwell, SM100 and SM120 | yes | yes | CoFDA F=25 CS=32, GDFS F=35 G=6 |

The NVFP4 kernels need SM100 or newer.

## 3. Change the accumulation

```bash
python run.py configs/llama8b-fp8-gsm8k-sweep.yaml --dry-run
python run.py configs/llama8b-fp8-gsm8k-sweep.yaml
```

Start with `--dry-run`. Every point is a fresh engine and a fresh model load,
so a config that expands to more than you meant is an hour you do not get back.
Values the kernels will not take are refused here, before anything loads. Each
run appends to `results/<config>-<timestamp>.jsonl` as it finishes, under a
first line recording the model, task, engine and GPU.

## The other coordinates

```bash
python run.py --list
```

Files are named `<model>-<format>-<task>-<kind>`, so a glob is an axis:
`configs/llama8b-nvfp4-*` is one checkpoint, `configs/*-sweep.yaml` every
sweep. `gsm8k_cot` measures the accumulation under decode, `wikitext` under
prefill, where nothing is generated. The kind is decided by what the number
means, not by how many points: one accumulation nobody's silicon implements is
a sweep, since its result is a finding rather than a check.

## Tips

- **`--limit N`** cuts every point to the first `N` questions — documents, for
  `wikitext`. The whole task otherwise, since `limit` is `null` in every config.
  Pass the flag rather than editing a file, which keeps the file saying what the
  experiment is.
- **A sweep** multiplies the fields it lists over `mma_emu`; two fields are
  taken as a product. Points that resolve to the same arithmetic are measured
  once, so a sweep may overlap a named part without paying twice.
- **The engine** is [`../common/engine.py`](../common/engine.py). A config may
  override any of it, plus `max_model_len`, `gpu_memory_utilization`,
  `max_num_seqs` and `enable_chunked_prefill` — but change one only when the
  comparison is about that setting.
- **The runner** is [`../common/runner.py`](../common/runner.py), shared with
  [`../cross_arch/`](../cross_arch/); `run.py` here only hands it this
  directory. A config may also name several `tasks`, which one model load then
  answers together.
- **`results/`** is gitignored: the numbers depend on the GPU and the sampled
  subset.
