#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare task accuracy across MMA accumulation configurations.

Evaluate the baseline on a task, confirm the emulation reproduces it, then
evaluate the accumulations no part implements. The runner is shared with
`../cross_arch/`; what makes this a sweep rather than a comparison across two
machines is the configs, in `configs/`.

    python NADPE_examples/eval/run.py configs/llama8b-fp8-gsm8k-native.yaml
    python NADPE_examples/eval/run.py configs/llama8b-fp8-*.yaml --dry-run
    python NADPE_examples/eval/run.py --list
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import runner  # noqa: E402

if __name__ == "__main__":
    runner.main(Path(__file__).resolve().parent)
