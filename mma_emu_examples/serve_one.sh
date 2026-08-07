#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: serve_one.sh cofda [F_BITS [CHUNK_SIZE]]
       serve_one.sh gdfs  [F_BITS [G_BITS [GROUP_SIZE]]]
       serve_one.sh native
       serve_one.sh --help

Serve one accumulation configuration, to prod by hand rather than to measure.
run_eval.py starts and stops a server per point and reports scores; this one
leaves a server up so you can send it prompts and watch what a narrower
accumulator does to them. Talk to it with `vllm chat`, or anything that speaks
the OpenAI API.

arguments:
  cofda         F_BITS      fractional bits of the accumulator  (default: 13)
                CHUNK_SIZE  products per chunk, 16 or 32        (default: 16)

  gdfs          F_BITS      inter-group accumulation width      (default: 35)
                G_BITS      intra-group accumulation width      (default: 6)
                GROUP_SIZE  products per group, 8 or 16         (default: 16)

  native        serve cutlass_scaled_mm instead, the baseline the emulation is
                supposed to reproduce. Named rather than left to auto, because
                FlashInfer sits ahead of CUTLASS in the FP8 candidate list and
                would take the layer.

environment:
  MODEL         (default: nvidia/Llama-3.1-8B-Instruct-FP8)
  PORT          (default: 8000)
  VLLM_PYTHON   (default: <repo>/.venv/bin/python)

examples:
  serve_one.sh                    # SM89: CoFDA F=13 CS=16, the Ada accumulation
  serve_one.sh cofda 7 16         # a narrower accumulator, to watch it diverge
  serve_one.sh cofda 25 32        # what SM100 and SM120 do
  serve_one.sh gdfs 35 6 16       # the NVFP4 accumulation those two use
  serve_one.sh native             # the baseline
  PORT=8010 serve_one.sh          # somewhere else

Every server is given --enforce-eager, to match what run_eval.py measures.
The accepted ranges live in the kernels; an out-of-range value is refused at
startup with an explanation rather than here.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-nvidia/Llama-3.1-8B-Instruct-FP8}"
PORT="${PORT:-8000}"
PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
VLLM="$(dirname "$PYTHON")/vllm"

# Eager, to match what run_eval.py measures. CUDA graph capture and
# torch.compile cost about twenty seconds of startup and change nothing about
# the arithmetic, so a server raised to look at one accumulation is better off
# without them.
ENFORCE_EAGER=(--enforce-eager)

# Ask for CUTLASS by name. Left at auto, FlashInfer sits ahead of it in the FP8
# candidate list and would take the layer instead.
if [[ "${1:-}" == "native" ]]; then
  exec "$VLLM" serve "$MODEL" --linear-backend cutlass \
    "${ENFORCE_EAGER[@]}" --port "$PORT"
fi

ALGORITHM="${1:-cofda}"

# The two algorithms are shaped differently: CoFDA chunks the products, GDFS
# accumulates groups at a second width. Sending the fields of one to the other
# would suggest they had been chosen.
case "$ALGORITHM" in
  cofda)
    KERNEL_CONFIG=$(printf \
      '{"mma_emu": {"algorithm": "cofda", "f_bits": %s, "chunk_size": %s}}' \
      "${2:-13}" "${3:-16}")
    ;;
  gdfs)
    KERNEL_CONFIG=$(printf \
      '{"mma_emu": {"algorithm": "gdfs", "f_bits": %s, "g_bits": %s, "group_size": %s}}' \
      "${2:-35}" "${3:-6}" "${4:-16}")
    ;;
  *)
    echo "Unknown algorithm '$ALGORITHM'. Expected cofda, gdfs or native." >&2
    echo "Run with --help for the arguments each one takes." >&2
    exit 2
    ;;
esac

echo "Serving $MODEL on :$PORT emulating $KERNEL_CONFIG" >&2
exec "$VLLM" serve "$MODEL" \
  --linear-backend mma_emu \
  --kernel-config "$KERNEL_CONFIG" \
  "${ENFORCE_EAGER[@]}" \
  --port "$PORT"
