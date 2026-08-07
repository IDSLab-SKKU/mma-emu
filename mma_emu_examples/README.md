# Running a model under an emulated accumulation

Two things live here. `serve_one.sh` puts one accumulation configuration behind
an API server so you can send it prompts. `run_eval.py` walks a list of
configurations, scores each on a task, and reports them side by side — which is
what turns "the numbers diverge" into "the divergence costs this much".

Both assume the build in the [README](../README.md) is done, including
`requirements/test/cuda.txt`, which is where `lm-eval` comes from.

## Serving one configuration

```bash
mma_emu_examples/serve_one.sh                # SM89: CoFDA F=13 CS=16
mma_emu_examples/serve_one.sh cofda 7 16     # algorithm, F, CS
mma_emu_examples/serve_one.sh native         # cutlass_scaled_mm, the baseline
```

Then talk to it with `vllm chat`, or anything that speaks the OpenAI API.
`MODEL` and `PORT` are environment overrides.

`native` asks for CUTLASS by name rather than dropping the flag, because the
FP8 candidate list runs MMA-Emu, Marlin, FlashInfer, CUTLASS — and FlashInfer,
unlike the emulation, does not decline when no algorithm is configured.

## Comparing configurations

```bash
python mma_emu_examples/run_eval.py mma_emu_examples/configs/architectures.yaml --dry-run
python mma_emu_examples/run_eval.py mma_emu_examples/configs/architectures.yaml
```

Start with `--dry-run`. It prints the runs the config expands to and stops,
which is worth the second it takes: each run is a fresh server and a fresh
model load, so a config that expands to more points than you meant is an hour
you do not get back.

## What comes out

```text
results/architectures-20260807-061530.jsonl
results/architectures-20260807-061530.md
```

The JSONL gains a line as each run finishes, so an interrupted sweep keeps what
it measured. Its **first** line is not a result but the conditions that produced
them — model, task, `limit`, `num_fewshot`, the commit, and which GPU — because
a score means nothing without those and nobody remembers them a month later.

The `.md` beside it is the comparison as a table, also printed when the run
ends. It is aligned, so it reads in a terminal and pastes into a document
unchanged:

```markdown
| configuration              | exact_match,strict-match |
| -------------------------- | ------------------------ |
| native (cutlass_scaled_mm) | 0.8123 ± 0.0276          |
| SM89 (CoFDA F=13 CS=16)    | 0.7950 ± 0.0286          |
```

Each invocation writes its own timestamped pair rather than appending to a
shared file, so runs at different `limit`s or on different commits cannot be
mistaken for each other. `--results` overrides the path when you do want to
continue into an existing file.

Because the measurements are on disk before the table is built, a run that dies
after them has still done its work. `--render` turns a results file back into a
table without serving anything:

```bash
python mma_emu_examples/run_eval.py --render results/f_bits_sweep-20260807-071451.jsonl
```

`results/` is gitignored. The numbers depend on the GPU and on the sampled
subset, so a copy in the tree would be somebody's measurement rather than
everyone's; keep the ones worth having in a paper or a README.

## Writing a config

Configurations are not enumerated one file per point — the design space is `F`
7–35, `G` 3–32, two algorithms and two chunk sizes, which is hundreds of
combinations. A config names an experiment, and the runs in it are expressed
three ways.

**By architecture.** The accumulation each one implements is already recorded
in `tests/kernels/quantization/mma_emu_arch.py` for the kernel tests, so naming
an architecture is enough:

```yaml
runs:
  - arch: [89, 90, 100, 120]
```

**Outright**, for a point that is nobody's architecture:

```yaml
runs:
  - mma_emu: {algorithm: cofda, f_bits: 13, chunk_size: 16}
```

**By sweep**, which multiplies the listed fields over the configuration above
them. `name` may reference any field:

```yaml
runs:
  - name: "CoFDA F={f_bits} CS=32"
    mma_emu: {algorithm: cofda, chunk_size: 32}
    sweep: {f_bits: [7, 13, 21, 25, 35]}
```

Two swept fields are taken as a product, so lists stay short on purpose.

The rest of the file is the experiment: `model`, `task`, `limit`,
`num_fewshot`, and `server_args` shared by every run.

`--enforce-eager` is added to every server whether the config asks for it or
not, here and in `serve_one.sh`. Each point pays engine startup once and then
answers a few hundred questions, so CUDA graph capture and torch.compile —
around twenty seconds a run — are overhead the measurement never earns back,
and holding the execution path fixed keeps the comparison about the
accumulation rather than about what got compiled.

## What this costs

Every point is a separate server, because the emulation kernels read their
bitwidths when a layer is built rather than per call — changing `f_bits` means
building the layers again, which means loading the model again.

Worse, the emulation is slow in a way that dominates everything else. It runs
the GEMM on CUDA cores rather than tensor cores, and an 8B model that would
generate thousands of tokens a second natively manages a couple. Measured on
SM120: `Avg generation throughput: 1.5 tokens/s` across 24 concurrent
requests. GSM8K's 200 questions are then not minutes but most of a day.

Three knobs follow from that:

| | default | why |
| --- | --- | --- |
| `timeout` | 86400 | lm-eval's is 300s, which fires before a single emulated request finishes and turns the run into retries |
| `num_concurrent` | 1 | lm-eval's own default, and the one that shows you a single request's latency plainly. Whether batching helps is untested — the emulation is compute-bound, so a bigger batch spends no fewer FLOPs, but a batch of one decodes as a matrix-vector product and leaves the GPU idle |
| `max_gen_toks` | lm-eval's 256 | the generated length is what you are paying for |

Start much smaller than you would natively — `limit: 10` is a reasonable first
run, and tells you whether the plumbing works before you spend a night on
statistics. Raise it once you know which comparison you care about, and expect
the native baseline to be the only fast point in the sweep.
