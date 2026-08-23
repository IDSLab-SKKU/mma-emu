#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: serve.sh [--model NAME] cofda [F_BITS [CHUNK_SIZE]]
       serve.sh [--model NAME] gdfs  [F_BITS [G_BITS [GROUP_SIZE]]]
       serve.sh [--model NAME] native
       serve.sh --help

Serve one accumulation behind an OpenAI-compatible server, to prod by hand
rather than to measure. See README.md for the models this accepts, the values
each algorithm takes, and what to ask it.

  --model NAME  checkpoint to serve (default: nvidia/Llama-3.1-8B-Instruct-FP8)
  native        serve cutlass_scaled_mm instead, the baseline

environment:
  PORT                 (default: 8000)
  VLLM_PYTHON          (default: <repo>/.venv/bin/python)
  KV_CACHE_DTYPE, ATTENTION_BACKEND, VLLM_USE_DEEP_GEMM,
  VLLM_BATCH_INVARIANT, FLASH_ATTN_VERSION   (see ../README.md)

examples:
  serve.sh cofda 25 32                       # what SM100 and SM120 do in FP8
  serve.sh --model nvidia/Llama-3.1-8B-Instruct-NVFP4 gdfs 35 6 16
  serve.sh native                            # the baseline
EOF
}

MODEL="${MODEL:-nvidia/Llama-3.1-8B-Instruct-FP8}"
if [[ "${1:-}" == "--model" ]]; then
  MODEL="$2"
  shift 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON="$REPO_ROOT/NADPE_examples/common"
PORT="${PORT:-8000}"
PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
VLLM="$(dirname "$PYTHON")/vllm"

# Eager, to match what eval/run.py measures; the rest of the engine comes from
# common/engine.py, which eval/run.py reads too. Greedy because the checkpoint's
# generation_config asks for temperature 0.6, which would move the answer on its
# own and leave the accumulation unreadable.
FIXED=(--enforce-eager --override-generation-config '{"temperature": 0}')
eval "$("$PYTHON" "$COMMON/engine.py" --env)"
while IFS= read -r flag; do FIXED+=("$flag"); done < <("$PYTHON" "$COMMON/engine.py" --serve-flags)

# CUTLASS by name. Left at auto, FlashInfer sits ahead of it in the FP8
# candidate list and would take the layer instead.
if [[ "${1:-}" == "native" ]]; then
  echo "Serving $MODEL on :$PORT natively" >&2
  exec "$VLLM" serve "$MODEL" --linear-backend cutlass \
    "${FIXED[@]}" --port "$PORT"
fi

ALGORITHM="${1:-cofda}"
case "$ALGORITHM" in
  cofda) ARGS=(--f-bits "${2:-}" --chunk-size "${3:-}") ;;
  gdfs) ARGS=(--f-bits "${2:-}" --g-bits "${3:-}" --group-size "${4:-}") ;;
  *)
    echo "Unknown algorithm '$ALGORITHM'. Expected cofda, gdfs or native." >&2
    echo "Run with --help for the arguments each one takes." >&2
    exit 2
    ;;
esac

# Drop the flags whose value was not given, so the defaults apply.
PAIRS=()
for ((i = 0; i < ${#ARGS[@]}; i += 2)); do
  [[ -n "${ARGS[i + 1]}" ]] && PAIRS+=("${ARGS[i]}" "${ARGS[i + 1]}")
done

# Reads the format out of the checkpoint and asks the kernels whether they
# would take these values, so a wrong one costs a second rather than a model
# load. Refusals are the kernels' own wording.
KERNEL_CONFIG=$("$PYTHON" "$COMMON/mma_emu.py" \
  --model "$MODEL" --algorithm "$ALGORITHM" "${PAIRS[@]}")

echo "Serving $MODEL on :$PORT emulating $KERNEL_CONFIG" >&2
exec "$VLLM" serve "$MODEL" \
  --linear-backend mma_emu \
  --kernel-config "$KERNEL_CONFIG" \
  "${FIXED[@]}" \
  --port "$PORT"
