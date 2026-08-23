# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

# The module that compares the emulation against a native kernel. The
# selection tests are not it: they decide which kernel a layer picks and never
# run a GEMM, so no comparison is theirs to describe.
_COMPARING_MODULE = "test_mma_emu.py"

# Which format a test is about, read off its name. Ordered, so the NVFP4 test
# whose name also says fp8 counts once, as NVFP4.
_FORMATS = ("nvfp4", "fp8")


def _collected_formats(session: pytest.Session) -> set[str]:
    """The formats this run actually selected tests for."""
    formats = set()
    for item in session.items:
        module = item.nodeid.rsplit("/", 1)[-1].split("::", 1)[0]
        if module != _COMPARING_MODULE:
            continue
        for fmt in _FORMATS:
            if fmt in item.name:
                formats.add(fmt)
                break
    return formats


def pytest_collection_finish(session: pytest.Session) -> None:
    """Say which comparison the MMA-Emu tests are about to make.

    Which architecture the run is on decides what the emulation is configured
    to reproduce and which native kernel it is checked against, and a wrong
    answer to either turns into a skip or a mismatch further down. Stating it
    once is cheaper than reading it out of skip messages.

    Reported here rather than from pytest_report_header, which runs before
    collection and so cannot tell which of these tests were selected.
    """
    if session.config.option.verbose < 0:  # -q
        return

    formats = _collected_formats(session)
    if not formats:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:  # pragma: no cover - no terminal, e.g. under xdist
        return

    from tests.kernels.quantization.mma_emu_arch import report_lines

    reporter.write_line("")
    for line in report_lines(formats):
        reporter.write_line(line)
