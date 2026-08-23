#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The accumulation to emulate, checked against the model and the kernels.

Two questions, both cheap next to a model load: which format is this
checkpoint, since FP8 and NVFP4 are different kernels with different accepted
values, and would the kernels take these values. The second goes through
`mma_emu_config_error`, the check the operators use as a backstop.

    python common/mma_emu.py --model M --algorithm cofda --f-bits 25
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# design_space.cuh names these; the integers are what the operators take.
ALGORITHMS = {"gdfs": 1, "cofda": 2}

# Per architecture, as recorded in design_space.cuh's own header comment.
DEFAULTS = {
    "cofda": {"f_bits": 13, "chunk_size": 16},
    "gdfs": {"f_bits": 35, "g_bits": 6, "group_size": 16},
}


def detect_format(model: str) -> str:
    """`fp8` or `nvfp4`, read from the checkpoint rather than asked for.

    Raises SystemExit naming what was found, because a model the emulation
    cannot take is better refused now than after it is loaded.
    """
    algo = _quant_algo(model)
    if algo is None:
        raise SystemExit(
            f"{model} does not look quantized, so there are no layers for the "
            f"emulation to replace. See chat/README.md for what is supported."
        )
    if "NVFP4" in algo or "FP4" in algo:
        return "nvfp4"
    if "FP8" in algo:
        return "fp8"
    raise SystemExit(
        f"{model} is quantized as {algo}, which the emulation kernels do not "
        f"implement. See chat/README.md for what is supported."
    )


def _quant_algo(model: str) -> str | None:
    """The checkpoint's own name for how it was quantized, or None."""
    # vLLM's helper rather than huggingface_hub: it looks in a local directory
    # and the cache before the network, and is the tagged path the repo
    # requires. Imported here so `--help` does not pay for vLLM.
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    for name in ("hf_quant_config.json", "config.json"):
        raw = get_hf_file_to_dict(name, model)
        if raw is None:
            continue
        # ModelOpt writes {"quantization": {"quant_algo": ...}}; the in-config
        # form is {"quantization_config": {...}} with either key.
        for section in (raw.get("quantization"), raw.get("quantization_config")):
            if isinstance(section, dict):
                algo = section.get("quant_algo") or section.get("quant_method")
                if algo:
                    return str(algo).upper()
    return None


def config_error(fields: dict[str, Any], fmt: str) -> str | None:
    """Why the kernels would refuse this, or None. Their wording, not ours."""
    import torch

    import vllm._custom_ops  # noqa: F401  - registers torch.ops._C

    return (
        torch.ops._C.mma_emu_config_error(
            ALGORITHMS[fields["algorithm"]],
            fields["f_bits"],
            fields["g_bits"],
            fields["group_size"],
            fields["chunk_size"],
            fmt == "nvfp4",
        )
        or None
    )


def resolve(algorithm: str, **overrides: Any) -> dict[str, Any]:
    """Every field the check needs, this algorithm's defaults filled in."""
    fields = {"algorithm": algorithm, "f_bits": 13, "g_bits": 6}
    fields |= {"chunk_size": 32, "group_size": 16}
    fields |= DEFAULTS[algorithm]
    fields |= {k: v for k, v in overrides.items() if v is not None}
    return fields


def chosen(fields: dict[str, Any]) -> dict[str, Any]:
    """Only the fields this algorithm reads.

    `g_bits` and `group_size` mean nothing to CoFDA and `chunk_size` means
    nothing to GDFS; sending them anyway would suggest they were chosen.
    """
    keys = ["algorithm", "f_bits"]
    keys += (
        ["chunk_size"] if fields["algorithm"] == "cofda" else ["g_bits", "group_size"]
    )
    return {k: fields[k] for k in keys}


def describe(fields: dict[str, Any]) -> str:
    """The configuration in the terms the paper uses."""
    if fields["algorithm"] == "cofda":
        return f"CoFDA F={fields['f_bits']} CS={fields['chunk_size']}"
    return f"GDFS F={fields['f_bits']} G={fields['g_bits']} GS={fields['group_size']}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    parser.add_argument("--f-bits", type=int)
    parser.add_argument("--g-bits", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()

    fmt = detect_format(args.model)
    fields = resolve(
        args.algorithm,
        f_bits=args.f_bits,
        g_bits=args.g_bits,
        chunk_size=args.chunk_size,
        group_size=args.group_size,
    )
    error = config_error(fields, fmt)
    if error:
        raise SystemExit(f"{error}\nThe accepted values live in design_space.cuh.")

    print(json.dumps({"mma_emu": chosen(fields)}))


if __name__ == "__main__":
    sys.exit(main())
