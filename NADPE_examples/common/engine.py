#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The engine settings both examples hold fixed.

Everything around the linear layers is pinned at the widest thing available,
stated once so that a server raised by hand is the engine `eval/run.py`
measures. `eval` overrides them from its config, `chat` from the environment.

    python common/engine.py --env           # export VLLM_... lines, for a shell
    python common/engine.py --serve-flags   # vllm serve flags, one per line
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

# Each default is the value that keeps a comparison about the accumulation.
DEFAULTS: dict[str, Any] = {
    # An FP8 cache quantizes the residual stream a second time.
    "kv_cache_dtype": "bfloat16",
    # Named: auto prefers FlashInfer on SM100, which quantizes Q to FP8.
    "attention_backend": "FLASH_ATTN",
    # DeepGEMM rounds FP8 block scales to a power of two (UE8M0).
    "use_deep_gemm": False,
    # Without it the batch decides which kernel runs, and the score moves.
    "batch_invariant": True,
    # Unset, vLLM picks per architecture: FA3 on SM90, FA4 on SM100, else FA2.
    "flash_attn_version": None,
}

# How the environment names each setting, for the shell-facing side.
ENV_ALIASES = {
    "kv_cache_dtype": "KV_CACHE_DTYPE",
    "attention_backend": "ATTENTION_BACKEND",
    "use_deep_gemm": "VLLM_USE_DEEP_GEMM",
    "batch_invariant": "VLLM_BATCH_INVARIANT",
    "flash_attn_version": "FLASH_ATTN_VERSION",
}


def resolve(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The defaults with `overrides` on top. A key absent or None keeps its
    default — except `flash_attn_version`, whose default *is* None.
    """
    settings = dict(DEFAULTS)
    for key, value in (overrides or {}).items():
        if key in DEFAULTS and (value is not None or key == "flash_attn_version"):
            settings[key] = value
    return settings


def env_vars(settings: dict[str, Any]) -> dict[str, str]:
    """The variables vLLM reads before an engine exists.

    Neither is an engine argument: DeepGEMM caches its answer on first ask and
    the batch-invariant flag is read from the environment by the CUDA sources.
    """
    return {
        "VLLM_USE_DEEP_GEMM": "1" if settings["use_deep_gemm"] else "0",
        "VLLM_BATCH_INVARIANT": "1" if settings["batch_invariant"] else "0",
    }


def engine_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """The settings that are engine arguments, as `LLM(**kwargs)` takes them."""
    kwargs: dict[str, Any] = {
        "kv_cache_dtype": settings["kv_cache_dtype"],
        "attention_backend": settings["attention_backend"],
    }
    # There is no --flash-attn-version; the version rides inside
    # attention_config, and omitting the key leaves vLLM's own choice alone.
    if settings["flash_attn_version"] is not None:
        kwargs["attention_config"] = {
            "flash_attn_version": settings["flash_attn_version"]
        }
    return kwargs


def serve_flags(settings: dict[str, Any]) -> list[str]:
    """The same settings as `vllm serve` flags."""
    flags = [
        "--kv-cache-dtype",
        str(settings["kv_cache_dtype"]),
        "--attention-backend",
        str(settings["attention_backend"]),
    ]
    if settings["flash_attn_version"] is not None:
        flags += [
            "--attention-config",
            f'{{"flash_attn_version": {settings["flash_attn_version"]}}}',
        ]
    return flags


def from_environ() -> dict[str, Any]:
    """What the shell asked for, in the terms `resolve` uses."""
    overrides: dict[str, Any] = {}
    for key, name in ENV_ALIASES.items():
        raw = os.environ.get(name)
        if not raw:
            continue
        if key in ("use_deep_gemm", "batch_invariant"):
            overrides[key] = raw != "0"
        elif key == "flash_attn_version":
            overrides[key] = int(raw)
        else:
            overrides[key] = raw
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--env",
        action="store_true",
        help="`export NAME=value` lines, to eval in a shell",
    )
    group.add_argument(
        "--serve-flags",
        action="store_true",
        help="vllm serve flags, one per line, to read into an array",
    )
    args = parser.parse_args()

    settings = resolve(from_environ())
    if args.env:
        for name, value in env_vars(settings).items():
            print(f"export {name}={value}")
    else:
        print("\n".join(serve_flags(settings)))


if __name__ == "__main__":
    sys.exit(main())
