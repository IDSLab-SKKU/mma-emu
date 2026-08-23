# Chatting with the emulation

Serve one accumulation, ask the model a question, then change the accumulation
and ask the same question again. That is the whole loop.

Run everything below from this directory, with the virtualenv active so `vllm`
is the one this repo built:

```bash
cd NADPE_examples/chat
source ../../.venv/bin/activate
```

## 1. Serve the baseline

```bash
./serve.sh native
```

`native` is `cutlass_scaled_mm`, the kernel the emulation is supposed to
reproduce. Wait for `Application startup complete`, then leave it running and
open a second terminal.

## 2. Ask it something with a checkable answer

```bash
vllm chat -q "Natalia sold clips to 48 friends in April, then half as many in May. How many altogether?"
```

The answer should be **72**. No flags: `vllm chat` already defaults to
port 8000 and takes the model name from the server. Drop `-q` for an
interactive session. Keep the question — every step below asks the same one.

## 3. Emulate the hardware

Stop the server with Ctrl-C and serve what your own GPU does in FP8:

| your GPU | serve it with |
| --- | --- |
| Ada — RTX 40, L4, L40S | `./serve.sh cofda 13 16` |
| Hopper — H100, H200 | `./serve.sh cofda 13 32` |
| Blackwell — B200, RTX 50, RTX PRO 6000 | `./serve.sh cofda 25 32` |

The numbers name the accumulation itself: **`cofda [F [CS]]`** accumulates
products in chunks of `CS`, each folded into a running sum kept to `F`
fractional bits.

On your own row **the answer should be identical**, token for token — the
emulation reproduces the baseline rather than approximating it.

## 4. Change the accumulation

Both parameters move. `F`, the fractional bits the running sum keeps, is a
range; `CS`, the chunk each fold spans, is an enumerated set, because it picks
a template instantiation rather than a value. `gdfs` takes FP8 too: groups of
`GS` summed at `G` fractional bits, the group results accumulated at `F`.

```bash
./serve.sh cofda 7 32       # F=7,  CS=32
./serve.sh cofda 25 16      # F=25, CS=16
./serve.sh gdfs 25 32 16    # F=25, G=32, GS=16
```

`serve.sh` checks both against the kernels before loading the model:

```console
$ ./serve.sh cofda 99
f_bits must be in [7, 35], got 99
The accepted values live in design_space.cuh.
```

The accepted values live in
[`design_space.cuh`](../../csrc/libtorch_stable/quantization/mma_emu/design_space.cuh)
and differ between FP8 and NVFP4, which is why the checkpoint's format is read
first.

Ask step 2's question after each restart, moving one parameter at a time.

## Other models

| model | format | try |
| --- | --- | --- |
| `nvidia/Llama-3.1-8B-Instruct-FP8` | FP8, per-tensor | the default; no `--model` needed |
| `nvidia/Llama-3.1-8B-Instruct-NVFP4` | NVFP4 | `--model nvidia/Llama-3.1-8B-Instruct-NVFP4` |

The format is read from the checkpoint's `hf_quant_config.json`, not taken as a
flag; anything else is refused before the weights load. Other checkpoints may
work, but the FP8 kernels take only **static per-tensor** scales — which rules
out most community FP8 releases, whose layers all decline the emulation.

NVFP4 takes both algorithms, but the kernel fixes `GS` and `CS` at 16, so `F` —
and `G` for `gdfs` — is the whole choice. Its baseline is
`cutlass_scaled_fp4_mm`, under the same `native`:

```bash
./serve.sh --model nvidia/Llama-3.1-8B-Instruct-NVFP4 native
./serve.sh --model nvidia/Llama-3.1-8B-Instruct-NVFP4 gdfs 35 6 16   # F=35, G=6, GS=16
./serve.sh --model nvidia/Llama-3.1-8B-Instruct-NVFP4 cofda 25 16    # F=25, CS=16
```
