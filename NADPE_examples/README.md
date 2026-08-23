# Running a model under an emulated accumulation

This fork replaces the accumulation inside a quantized linear layer with an
emulated one. Before anything below:

| | |
| --- | --- |
| **build** | the [top-level README](../README.md). `eval/` and `cross_arch/` also need `lm-eval`, which the `requirements/test/cuda.txt` phase installs; `chat/` does not |
| **virtualenv** | `source ../.venv/bin/activate`, from here |
| **checkpoint** | `nvidia/Llama-3.1-8B-Instruct-FP8`, the default in all three, or `nvidia/Llama-3.1-8B-Instruct-NVFP4` |

Other checkpoints may work, but only a quantized layer has a kernel to replace,
and FP8 is taken only with static per-tensor scales.

## [`chat/`](chat/)

Puts one accumulation behind an OpenAI-compatible server and prompts it by
hand. Restart with a different one and ask the same question to see whether the
accumulation changed the tokens at all: the one the GPU implements reproduces
the baseline exactly; a narrower one diverges within a sentence.

```bash
cd chat
./serve.sh cofda 25 32       # F=25, CS=32 — what Blackwell does in FP8
vllm chat -q "48 clips in April, half as many in May. How many altogether?"
```

## [`eval/`](eval/)

Runs a list of accumulations through lm-eval and reports them side by side, so
a divergence becomes a number: `exact_match` on `gsm8k_cot` for decode,
perplexity on `wikitext` for prefill. One engine and one model load per point,
so a sweep is hours.

```bash
cd eval
python run.py --list                                          # the configs
python run.py configs/llama8b-fp8-gsm8k-sweep.yaml --dry-run  # the points
python run.py configs/llama8b-fp8-gsm8k-sweep.yaml
```

## [`cross_arch/`](cross_arch/)

Takes the accumulation one GPU implements, emulates it on a different one, and
asks whether the second machine now reproduces the first bit for bit — scores
and per-sample logprobs both. Needs two machines, and both sides' results in
one `results/` directory — which you bring together yourself, since results are
not committed.

```bash
cd cross_arch
python run.py configs/llama8b-fp8-mc-native.yaml   # on the H100
python run.py configs/llama8b-fp8-mc-hopper.yaml   # on the Blackwell
python compare.py                                  # once both results are here
```

## [`common/`](common/)

`runner.py` is the script `eval/` and `cross_arch/` both run — each `run.py` is
a shim that hands it its own directory. The two ask different questions but
build the same engine and write the same shape of result, so what separates
them is their configs. The keys one may state, beyond `engine.py`'s below:

| key | what it does |
| --- | --- |
| `task` / `tasks` | one task or several, all through one model load |
| `native`, `emulation` | the accumulations to measure; `emulation` may carry a `sweep` |
| `log_samples` | keep every answer choice — one accumulation per file |
| `timestamped` | `false` names the results file a place rather than an event |
| `seed` | pin lm-eval's seeds instead of taking its defaults |

`engine.py` pins everything outside the linear layers, so a difference between
two runs is the accumulation. Its `DEFAULTS`, which `eval/` and `cross_arch/`
override from a config and `chat/` from the environment:

| setting | default | what it is |
| --- | --- | --- |
| `kv_cache_dtype` | `bfloat16` | the dtype the KV cache is stored in |
| `attention_backend` | `FLASH_ATTN` | which attention implementation runs |
| `use_deep_gemm` | `false` | whether DeepGEMM takes the FP8 block-scaled GEMMs |
| `batch_invariant` | `true` | whether a token's result is independent of the batch it lands in |
| `flash_attn_version` | unset | which FlashAttention version — 2, 3 or 4 |

`mma_emu.py` is what they all ask before a model load — the checkpoint's
format, and whether the kernels accept the configuration. `device.py` names the
GPU a run executed on, which is recorded with every result and is what
`cross_arch/` holds its two sides apart by.
