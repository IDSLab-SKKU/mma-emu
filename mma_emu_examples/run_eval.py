#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare task accuracy across MMA accumulation configurations.

The emulation kernels read their bitwidths when a layer is built, so every
configuration needs its own engine. A comparison is therefore a sequence of
servers, one per point, each evaluated in turn and torn down. Results are
appended as they arrive, so an interrupted sweep keeps what it measured.

    python mma_emu_examples/run_eval.py configs/architectures.yaml --dry-run
    python mma_emu_examples/run_eval.py configs/architectures.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(REPO_ROOT))

# The accumulation each architecture implements is already recorded for the
# kernel tests. Importing it keeps the paper's table in one place.
from tests.kernels.quantization.mma_emu_arch import (  # noqa: E402
    ARCH_ACCUMULATION,
    DEVICE,
    SM,
)
from vllm.config.mma_emu import MmaEmuConfig  # noqa: E402


@dataclass
class Run:
    """One point of the comparison: a label and the server that produces it."""

    name: str
    linear_backend: str
    mma_emu: dict | None = None
    extra_args: list[str] = field(default_factory=list)

    def server_args(self, model: str) -> list[str]:
        args = [model, "--linear-backend", self.linear_backend]
        if self.mma_emu is not None:
            args += ["--kernel-config", json.dumps({"mma_emu": self.mma_emu})]
        return args + self.extra_args

    def describe(self, model: str) -> str:
        """The server this run stands for, verbatim.

        A dry run is for catching a config that says something other than what
        was meant, so it shows what would actually be passed rather than a
        summary of it.
        """
        return "vllm serve " + shlex.join(self.server_args(model))


def kernel_config(config: MmaEmuConfig) -> dict:
    """The `mma_emu` fields that apply to this algorithm, and no others.

    `g_bits` and `group_size` mean nothing to CoFDA and `chunk_size` means
    nothing to GDFS; sending them anyway would suggest they were chosen.
    """
    fields: dict = {"algorithm": config.algorithm, "f_bits": config.f_bits}
    if config.algorithm == "cofda":
        fields["chunk_size"] = config.chunk_size
    else:
        fields["g_bits"] = config.g_bits
        fields["group_size"] = config.group_size
    return fields


def label(fields: dict) -> str:
    """The configuration in the terms the paper uses."""
    if fields["algorithm"] == "cofda":
        return f"CoFDA F={fields['f_bits']} CS={fields['chunk_size']}"
    return f"GDFS F={fields['f_bits']} G={fields['g_bits']} GS={fields['group_size']}"


def expand(spec: dict, fmt: str, extra_args: list[str]) -> list[Run]:
    """Turn one entry of `runs` into the points it stands for.

    Three forms are accepted, in increasing specificity: `arch` names
    architectures whose accumulation is already recorded, `mma_emu` gives one
    configuration outright, and `sweep` multiplies whichever fields it lists
    over that configuration.
    """
    if "arch" in spec:
        runs = []
        for sm in spec["arch"]:
            if sm not in ARCH_ACCUMULATION:
                raise SystemExit(
                    f"No recorded accumulation for SM{sm}. "
                    f"Known: {sorted(ARCH_ACCUMULATION)}"
                )
            if fmt not in ARCH_ACCUMULATION[sm]:
                raise SystemExit(f"SM{sm} has no {fmt} accumulation recorded.")
            fields = kernel_config(ARCH_ACCUMULATION[sm][fmt])
            runs.append(
                Run(
                    name=f"SM{sm} ({label(fields)})",
                    linear_backend="mma_emu",
                    mma_emu=fields,
                    extra_args=extra_args,
                )
            )
        return runs

    backend = spec.get("linear_backend", "mma_emu")
    if "mma_emu" not in spec:
        return [
            Run(
                name=spec.get("name", backend),
                linear_backend=backend,
                extra_args=extra_args,
            )
        ]

    base = dict(spec["mma_emu"])
    sweep = spec.get("sweep", {})
    if not sweep:
        return [
            Run(
                name=spec.get("name", label(base)),
                linear_backend=backend,
                mma_emu=base,
                extra_args=extra_args,
            )
        ]

    axes = list(sweep)
    runs = []
    for point in itertools.product(*(sweep[axis] for axis in axes)):
        fields = base | dict(zip(axes, point))
        runs.append(
            Run(
                name=spec.get("name", "").format(**fields) or label(fields),
                linear_backend=backend,
                mma_emu=fields,
                extra_args=extra_args,
            )
        )
    return runs


def wait_for_health(port: int, timeout: float, server: subprocess.Popen) -> None:
    """Block until the server answers, it dies, or the wait runs out.

    Watching the process matters as much as watching the port: a server that
    rejects its configuration or cannot bind is gone in seconds, and polling
    alone would report that as a timeout minutes later, naming the wrong
    problem.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {server.returncode} before it "
                f"answered. Its output is above."
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    raise TimeoutError(
        f"Server did not answer on port {port} within {timeout:.0f}s, and is "
        f"still running. Something else may hold the port, or the model is "
        f"slower to load than --startup-timeout allows."
    )


def evaluate(model: str, port: int, cfg: dict) -> dict:
    """Run the task against the serving endpoint and return its metrics.

    Emulated accumulation runs on CUDA cores rather than tensor cores and
    generates at a small fraction of the native rate, so lm-eval's 300-second
    request timeout fires long before a request finishes and the run dies in
    retries. `timeout` is raised to a day — during a comparison the thing
    worth waiting for is the answer, and a stalled server shows up as a silent
    GPU rather than as a fast error.

    That is the only default overridden here. Everything else is lm-eval's —
    `num_concurrent` 1, `max_gen_toks` 256, `tokenized_requests` — and the
    config can set any of them by name. Measured at 51 seconds a question on
    SM120, batching is not obviously the lever it looks like: the emulation is
    compute-bound, so a larger batch spends no fewer FLOPs.
    """
    import lm_eval

    settings = {
        "model": model,
        "base_url": f"http://127.0.0.1:{port}/v1/completions",
        "timeout": 86400,
    }
    for key in ("num_concurrent", "max_gen_toks", "tokenized_requests", "timeout"):
        if cfg.get(key) is not None:
            settings[key] = cfg[key]
    model_args = ",".join(f"{k}={v}" for k, v in settings.items())
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=[cfg["task"]],
        limit=cfg.get("limit"),
        num_fewshot=cfg.get("num_fewshot"),
        apply_chat_template=cfg.get("apply_chat_template", False),
    )
    return results["results"][cfg["task"]]


def append(path: Path, record: dict) -> None:
    """Add one line, flushed, so an interrupted sweep keeps what it measured."""
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def provenance(cfg: dict, stamp: str) -> dict:
    """The first line of a results file: what was measured, and on what.

    Scores are only comparable against the conditions that produced them, and
    those conditions are what nobody remembers a month later.
    """
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None

    return {
        "kind": "provenance",
        "timestamp": stamp,
        "commit": commit,
        "device": {"sm": SM, "name": DEVICE},
        "model": cfg["model"],
        "task": cfg["task"],
        "limit": cfg.get("limit"),
        "num_fewshot": cfg.get("num_fewshot"),
        "server_args": cfg.get("server_args", ""),
    }


def _stderr_key(metric: str) -> str:
    """lm-eval names the error for `exact_match,foo` as `exact_match_stderr,foo`."""
    name, comma, suffix = metric.partition(",")
    return f"{name}_stderr{comma}{suffix}"


def render(records: list[dict]) -> str:
    """The comparison as a Markdown table, aligned so a terminal reads it too."""
    if not records:
        return "No runs completed."

    # Selected by the type of the value rather than by name. lm-eval mixes
    # labels into the same dict as the scores — `name`, `alias`, `sample_len` —
    # and the set differs by task, so any list of names to skip would be a
    # guess that holds until the next task.
    metrics = []
    for record in records:
        for key, value in record["metrics"].items():
            if "_stderr" in key or key in metrics:
                continue
            if isinstance(value, float):
                metrics.append(key)

    header = ["configuration", *metrics]
    rows = []
    for record in records:
        row = [record["name"]]
        for metric in metrics:
            value = record["metrics"].get(metric)
            error = record["metrics"].get(_stderr_key(metric))
            if value is None:
                row.append("—")
            elif isinstance(error, float):
                row.append(f"{value:.4f} ± {error:.4f}")
            else:
                row.append(f"{value:.4f}")
        rows.append(row)

    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    return "\n".join(
        [line(header), "| " + " | ".join("-" * w for w in widths) + " |"]
        + [line(row) for row in rows]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare task accuracy across MMA accumulation configurations.\n"
            "\n"
            "The emulation kernels read their bitwidths when a layer is\n"
            "built, so every configuration needs its own engine: a\n"
            "comparison is a sequence of servers, one per point, each\n"
            "evaluated in turn and torn down. Results land in\n"
            "results/<config>-<timestamp>.jsonl as they arrive, so an\n"
            "interrupted sweep keeps what it measured, and a Markdown table\n"
            "of the comparison is written beside them."
        ),
        epilog=(
            "examples:\n"
            "  run_eval.py configs/architectures.yaml --dry-run\n"
            "  run_eval.py configs/architectures.yaml\n"
            "  run_eval.py configs/f_bits_sweep.yaml --port 8010\n"
            "\n"
            "Start with --dry-run. Each point is a fresh server and a fresh\n"
            "model load, so a config that expands to more points than you\n"
            "meant is an hour you do not get back.\n"
            "\n"
            "Every server is given --enforce-eager, whether the config asks\n"
            "or not: compiling for a few hundred questions never pays back,\n"
            "and a fixed execution path keeps the comparison about the\n"
            "accumulation.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, nargs="?", help="experiment YAML")
    parser.add_argument(
        "--render",
        type=Path,
        metavar="JSONL",
        help="rebuild the table from a results file and stop, without serving "
        "anything. The measurements are written as they arrive, so this "
        "recovers a run that died after them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the runs the config expands to, and stop",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port each server listens on in turn (default: 8000)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for each server (default: 900)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="JSONL to append to (default: alongside the config)",
    )
    args = parser.parse_args()

    if args.render is not None:
        records = [json.loads(line) for line in args.render.read_text().splitlines()]
        table = render([r for r in records if r.get("kind") != "provenance"])
        print(table)
        table_path = args.render.with_suffix(".md")
        table_path.write_text(table)
        print(f"\nWrote {table_path}")
        return

    if args.config is None:
        parser.error("a config is required unless --render is given")

    cfg = yaml.safe_load(args.config.read_text())
    model = cfg["model"]
    fmt = cfg.get("format", "fp8")
    extra_args = shlex.split(cfg.get("server_args", ""))
    # Forced rather than left to the config. Every point here pays engine
    # startup once and then answers a few hundred questions, so CUDA graph
    # capture and torch.compile are overhead the measurement never earns back —
    # and holding the execution path fixed keeps the comparison about the
    # accumulation. Upstream's own accuracy configs run eager for the same
    # reason.
    if "--enforce-eager" not in extra_args:
        extra_args.append("--enforce-eager")

    runs: list[Run] = []
    for spec in cfg["runs"]:
        runs += expand(spec, fmt, extra_args)

    print(f"{args.config}: {len(runs)} runs of {cfg['task']}", end="")
    if cfg.get("limit"):
        print(f" (limit {cfg['limit']})", end="")
    print(f" against {model}\n")
    for i, run in enumerate(runs, 1):
        print(f"  {i:>2}. {run.name}")
        print(f"      {run.describe(model)}")
    print()

    if args.dry_run:
        print("Dry run: nothing was served or evaluated.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = args.results or RESULTS_DIR / f"{args.config.stem}-{stamp}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    append(results_path, provenance(cfg, stamp))

    vllm = Path(sys.executable).parent / "vllm"
    records = []

    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run.name}: serving ...", flush=True)
        command = [
            str(vllm),
            "serve",
            *run.server_args(model),
            "--port",
            str(args.port),
        ]
        server = subprocess.Popen(command)
        try:
            wait_for_health(args.port, args.startup_timeout, server)
            print(f"[{i}/{len(runs)}] {run.name}: evaluating ...", flush=True)
            metrics = evaluate(model, args.port, cfg)
        finally:
            server.terminate()
            server.wait(timeout=120)

        record = {"name": run.name, "config": run.mma_emu, "metrics": metrics}
        records.append(record)
        append(results_path, record)
        print(f"[{i}/{len(runs)}] {run.name}: {metrics}\n", flush=True)

    table = render(records)
    print(table)

    table_path = results_path.with_suffix(".md")
    table_path.write_text(table)
    print(f"\nWrote {results_path}\n      {table_path}")


if __name__ == "__main__":
    main()
