# Reproducing one architecture on another

Two GPUs disagree about a quantized model's output. This asks whether the MMA
accumulation explains *all* of it: take the accumulation one of them
implements, emulate it on the other, and check whether the second machine now
produces what the first did — bit for bit, not approximately.

Siblings measure one machine: [`../eval/`](../eval/) scores a list of
accumulations, [`../chat/`](../chat/) serves one by hand.

## Run it

| | GPU | config | ~time |
| --- | --- | --- | --- |
| baseline | H100 (SM90) | `configs/llama8b-fp8-mc-native.yaml` | 5 min |
| emulation | RTX PRO 6000 (SM120) | `configs/llama8b-fp8-mc-hopper.yaml` | 3 h |

**Nothing checks which GPU you are on.** Both configs on one machine produce two
files that agree perfectly and prove nothing; the table is the whole guard.
`compare.py` does say so when both sides name the same architecture.

```bash
cd NADPE_examples/cross_arch && source ../../.venv/bin/activate

python run.py configs/llama8b-fp8-mc-native.yaml   # on the H100
python run.py configs/llama8b-fp8-mc-hopper.yaml   # on the Blackwell

python compare.py                                  # the pair in results/
```

Every score should be identical and every logprob bit-identical. `compare.py`
exits non-zero if not, and `--out cmp.md` writes the table as markdown. It
compares the two results it finds; name them if there are more than two.

`results/` is gitignored, as `../eval/results/` is, so carrying one machine's
results to the other is yours to do — copy its `results/*.jsonl` across before
running `compare.py`. Filenames carry no timestamp, so a re-run overwrites
rather than accumulating.

Use `--limit 20` on both sides to check the plumbing before spending the three
hours. Both sides must use the same limit; it is recorded in the result.

## Why the configs say what they say

They differ in one field — `native: true` against `emulation: {...}`. The rest
is identical on purpose, because the experiment means nothing if the
accumulation is not the sole difference, and a second one is easy to introduce
by accident:

- **`flash_attn_version: 2`** — unset, vLLM picks per architecture: FA3 on
  SM90, FA2 on SM120. The two sides would run different attention, and no
  agreement in the linear layers could make their logprobs match.
- **`max_num_seqs: 1`** — the running batch decides the shapes the GEMMs see.
- **`kv_cache_dtype`, `attention_backend`, `batch_invariant`** — from
  [`../common/engine.py`](../common/engine.py), which explains each. Overridable,
  but here that is a way to get a wrong answer.

The tasks are multiple choice, so every answer choice is a scored logprob and
the comparison reaches the arithmetic directly rather than through sampled text.

Three more keys make this a cross-architecture check rather than a sweep:
`log_samples: true` keeps every choice, `timestamped: false` names the results
file a place rather than an event, and `seed: 42` pins lm-eval's seeds. The
runner reading them is [`../common/runner.py`](../common/runner.py), shared with
[`../eval/`](../eval/).

## What comes out

`results/<config>.jsonl` — a provenance line (GPU, commit, model, accumulation,
engine) and one line per task. `results/<config>-samples.jsonl` — lm-eval's full
sample records, one per line, which is what makes the logprob check possible;
roughly 10 MB per side for a full run, which is why it is not committed.

`compare.py` compares scores for equality rather than closeness, since the
emulation reproduces the hardware rather than approximating it. It checks the
samples too — per answer choice, and by prompt hash, so a divergence in what was
*asked* cannot pass as agreement about the answer.

Which accumulation each architecture implements:
[`mma_emu_arch.py`](../../tests/kernels/quantization/mma_emu_arch.py). Which
values the kernels take:
[`design_space.cuh`](../../csrc/libtorch_stable/quantization/mma_emu/design_space.cuh)
— a config asking for anything else is refused before a model loads.
