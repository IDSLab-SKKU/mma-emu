#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one side of the cross-architecture check.

The claim is that the MMA accumulation is the only thing separating two
architectures' output. Testing it takes two machines: the baseline on the GPU
whose accumulation is being reproduced, and that same accumulation emulated on
the other one. This runs one side; `compare.py` reads both.

The runner is shared with `../eval/`. What makes this a cross-architecture
check is in `configs/`: one accumulation, every sample logged, and a results
file named as a place rather than an event, since git is how the two machines
exchange them.

    python cross_arch/run.py configs/llama8b-fp8-mc-native.yaml
    python cross_arch/run.py configs/llama8b-fp8-mc-hopper.yaml --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import runner  # noqa: E402

if __name__ == "__main__":
    runner.main(Path(__file__).resolve().parent)
