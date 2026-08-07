# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def pytest_report_header() -> list[str]:
    """Say which comparison the MMA-Emu tests are about to make.

    Which architecture the run is on decides what the emulation is configured
    to reproduce and which native kernel it is checked against, and a wrong
    answer to either turns into a skip or a mismatch further down. Stating it
    once at the top is cheaper than reading it out of skip messages.
    """
    from tests.kernels.quantization.mma_emu_arch import report_lines

    return report_lines()
