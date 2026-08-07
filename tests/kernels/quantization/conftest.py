# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

_MMA_EMU_MODULES = ("test_mma_emu",)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Say which comparison the MMA-Emu tests are about to make.

    Which architecture the run is on decides what the emulation is configured
    to reproduce and which native kernel it is checked against, and a wrong
    answer to either turns into a skip or a mismatch further down. Stating it
    once is cheaper than reading it out of skip messages.

    Reported here rather than from pytest_report_header, which runs before
    collection and so cannot tell whether any of these tests were selected.
    """
    if session.config.option.verbose < 0:  # -q
        return

    selected = any(
        item.nodeid.rsplit("/", 1)[-1].startswith(_MMA_EMU_MODULES)
        for item in session.items
    )
    if not selected:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # pragma: no cover - no terminal, e.g. under xdist
        return

    from tests.kernels.quantization.mma_emu_arch import report_lines

    reporter.write_line("")
    for line in report_lines():
        reporter.write_line(line)
