#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The runner both examples call.

`eval/` and `cross_arch/` ask different questions — one sweeps accumulations on
one machine, the other runs a single accumulation on two — but they build the
same engine, evaluate the same way and write the same shape of result. So they
run this, and differ only in the configs they hold.

The keys a config may state:

    model, task or tasks, limit, num_fewshot, apply_chat_template
    max_model_len, gpu_memory_utilization, max_num_seqs, enable_chunked_prefill
    seed, log_samples, timestamped
    native, emulation
    anything common/engine.py resolves

An example directory calls `main(its own path)`; `configs/` and `results/` are
read and written under it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_ROOT))

# `engine` is the settings chat and eval hold fixed; `mma_emu` reads the
# checkpoint's format and asks the kernels whether they take a configuration.
from common import device, engine, mma_emu  # noqa: E402

# What `native` resolves to, which the checkpoint decides rather than the config.
NATIVE_KERNEL = {"fp8": "cutlass_scaled_mm", "nvfp4": "cutlass_scaled_fp4_mm"}


@dataclass
class Run:
    """One accumulation: a label and the engine that produces it."""

    name: str
    mma_emu: dict | None = None

    def kernel_config(self) -> dict:
        """The `kernel_config` this run builds its layers with. No
        accumulation means the baseline, named rather than left to auto:
        FlashInfer sits ahead of CUTLASS in the FP8 candidate list.
        """
        if self.mma_emu is None:
            return {"linear_backend": "cutlass"}
        return {"linear_backend": "mma_emu", "mma_emu": self.mma_emu}

    def describe(self, model: str) -> str:
        """The engine this run would build, verbatim, for --dry-run."""
        return f"LLM({model!r}, kernel_config={self.kernel_config()})"


@dataclass
class Plan:
    """One config file: the runs it expands to, and what they are measured on."""

    path: Path
    cfg: dict
    fmt: str
    runs: list[Run]


def specs(cfg: dict) -> list[dict]:
    """The `emulation` entries, in the one form the rest of this reads.

    A sweep states `{mma_emu: ..., sweep: ...}` per entry. A config asking for
    a single accumulation may say it directly, since wrapping one dictionary in
    a list to nest it under its own name reads like a mistake.
    """
    spec = cfg.get("emulation")
    if spec is None:
        return []
    if isinstance(spec, dict):
        return [spec if "mma_emu" in spec else {"mma_emu": spec}]
    return list(spec)


def expand(spec: dict, fmt: str) -> list[Run]:
    """The runs one `emulation` entry stands for, each checked by the kernels
    as it is made, so a wrong value costs a second rather than a load.
    """
    if "mma_emu" not in spec:
        raise SystemExit(f"An `emulation` entry needs an `mma_emu` field: {spec}")

    base = dict(spec["mma_emu"])
    sweep = spec.get("sweep", {})
    axes = list(sweep)
    points = itertools.product(*(sweep[axis] for axis in axes)) if axes else [()]

    runs = []
    for point in points:
        overrides = base | dict(zip(axes, point))
        algorithm = overrides.pop("algorithm", None)
        if algorithm not in mma_emu.ALGORITHMS:
            raise SystemExit(
                f"algorithm must be one of {sorted(mma_emu.ALGORITHMS)}, "
                f"got {algorithm!r}"
            )

        resolved = mma_emu.resolve(algorithm, **overrides)
        error = mma_emu.config_error(resolved, fmt)
        if error:
            raise SystemExit(f"{error}\nThe accepted values live in design_space.cuh.")

        fields = mma_emu.chosen(resolved)
        name = spec.get("name", "").format(**fields) or mma_emu.describe(fields)
        runs.append(Run(name=name, mma_emu=fields))
    return runs


def dedupe(runs: list[Run]) -> list[Run]:
    """One engine per distinct accumulation, however many names reached it.

    A named part and a point of a sweep can be the same arithmetic. Names are
    joined so the table still says everything the config asked about.
    """
    seen: dict[str, Run] = {}
    order: list[str] = []
    for run in runs:
        key = json.dumps(run.mma_emu, sort_keys=True)
        if key in seen:
            if run.name not in seen[key].name.split(" / "):
                seen[key].name += f" / {run.name}"
            continue
        seen[key] = run
        order.append(key)
    return [seen[key] for key in order]


def plan(path: Path, limit: int | None) -> Plan:
    """Every engine one config asks for, without loading a model.

    The format comes from the checkpoint rather than a key: it decides which
    values are legal, and a config cannot be wrong about what it never states.
    """
    cfg = yaml.safe_load(path.read_text())
    if limit is not None:
        cfg["limit"] = limit

    tasks = cfg.get("tasks", cfg.get("task"))
    cfg["tasks"] = [tasks] if isinstance(tasks, str) else tasks
    if not cfg["tasks"]:
        raise SystemExit(f"{path}: `tasks` is what this measures, and it is empty.")

    fmt = mma_emu.detect_format(cfg["model"])
    runs: list[Run] = []
    if cfg.get("native"):
        runs.append(Run(name=f"native ({NATIVE_KERNEL[fmt]})"))
    for spec in specs(cfg):
        runs += expand(spec, fmt)

    if not runs:
        raise SystemExit(f"{path}: neither `native` nor `emulation` asks for anything.")
    runs = dedupe(runs)

    # Samples are keyed by task alone, in the record and in the sidecar, so a
    # second accumulation in the same file would overwrite the first rather
    # than sit beside it. A comparison that reaches the logprobs is about one
    # accumulation by construction.
    if cfg.get("log_samples") and len(runs) > 1:
        raise SystemExit(
            f"{path}: `log_samples` records one accumulation per file, but this "
            f"asks for {len(runs)}."
        )
    return Plan(path=path, cfg=cfg, fmt=fmt, runs=runs)


def evaluate(cfg: dict, run: Run) -> Iterator[tuple[str, dict]]:
    """Build one engine and yield lm-eval's result for each task in turn.

    One engine for all the tasks: they differ in what they ask the model, not
    in how it accumulates, so a second model load would only cost an hour.

    But one lm-eval call per task rather than one for all of them. Handed
    several, lm-eval builds a single request set out of the lot and runs it as
    one batch, so a four-task config reports nothing until the last question of
    the last task is answered, and a failure late in the run loses the tasks
    that had already been answered. Called once per task, each finishes, is
    written, and is printed before the next one starts.
    """
    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM

    seed = cfg.get("seed")
    vllm_kwargs = dict(
        pretrained=cfg["model"],
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.70),
        max_model_len=cfg.get("max_model_len", 4096),
        batch_size="auto",
        # Compiling for a few thousand questions never pays back, and a fixed
        # execution path keeps the comparison about the accumulation.
        enforce_eager=True,
        kernel_config=run.kernel_config(),
        # bf16 around the linear layers; defaults and reasons in
        # common/engine.py, which a config may override.
        **engine.engine_kwargs(engine.resolve(cfg)),
    )
    for key in ("max_num_seqs", "enable_chunked_prefill"):
        if cfg.get(key) is not None:
            vllm_kwargs[key] = cfg[key]
    if seed is not None:
        vllm_kwargs["seed"] = seed

    # Absent, lm-eval's own seeds apply. A config that pins them is asking to
    # be reproducible rather than merely repeatable, which is what a comparison
    # across two machines needs and a sweep on one does not.
    seeds = (
        {
            "random_seed": seed,
            "numpy_random_seed": seed,
            "torch_random_seed": seed,
            "fewshot_random_seed": seed,
        }
        if seed is not None
        else {}
    )

    lm = VLLM(**vllm_kwargs)
    for i, task in enumerate(cfg["tasks"], 1):
        print(f"  [{i}/{len(cfg['tasks'])}] {task}: evaluating ...", flush=True)
        yield (
            task,
            evaluator.simple_evaluate(
                model=lm,
                tasks=[task],
                limit=cfg.get("limit"),
                # Absent, each task's own definition applies — gsm8k_cot asks
                # for eight shots itself, so restating it here is only a way to
                # disagree.
                num_fewshot=cfg.get("num_fewshot"),
                batch_size="auto",
                log_samples=cfg.get("log_samples", False),
                apply_chat_template=cfg.get("apply_chat_template", False),
                **seeds,
            ),
        )


def append(path: Path, record: dict) -> None:
    """Add one line, so an interrupted sweep keeps what it measured."""
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def provenance(item: Plan, stamp: str) -> dict:
    """The first line of a results file: what was measured, and on what.

    Baselines and sweeps are separate files, so the settings are recorded
    rather than implied. `max_model_len` is among them because under wikitext's
    rolling loglikelihood it is the window summed over, and the device because
    a cross-architecture comparison refuses to read that off a filename.

    A file holding one accumulation names it here, where it describes the whole
    file rather than one line of it.
    """
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None

    cfg = item.cfg
    record = {
        "kind": "provenance",
        "timestamp": stamp,
        "commit": commit,
        "device": device.info(),
        "model": cfg["model"],
        "format": item.fmt,
        "tasks": cfg["tasks"],
        "limit": cfg.get("limit"),
        "num_fewshot": cfg.get("num_fewshot"),
        "seed": cfg.get("seed"),
        "max_model_len": cfg.get("max_model_len", 4096),
        "max_num_seqs": cfg.get("max_num_seqs"),
        "engine": engine.resolve(cfg),
    }
    if len(item.runs) == 1:
        record |= {"name": item.runs[0].name, "accumulation": item.runs[0].mma_emu}
    return record


def _stderr_key(metric: str) -> str:
    """lm-eval names the error for `exact_match,foo` as `exact_match_stderr,foo`."""
    name, comma, suffix = metric.partition(",")
    return f"{name}_stderr{comma}{suffix}"


def render(records: list[dict]) -> str:
    """The results as a table, aligned so a terminal reads it.

    A column that would say the same thing on every row is left out, so one
    accumulation over four tasks and four accumulations over one task each read
    as the list they are.
    """
    if not records:
        return "No runs completed."

    # By the type of the value rather than by name: lm-eval mixes labels into
    # the same dict as the scores, and the set differs by task.
    metrics: list[str] = []
    for record in records:
        for key, value in record["metrics"].items():
            if "_stderr" in key or key in metrics:
                continue
            if isinstance(value, float):
                metrics.append(key)

    axes = [
        (label, key)
        for label, key in (("configuration", "name"), ("task", "task"))
        if len({record[key] for record in records}) > 1
    ] or [("configuration", "name")]

    header = [label for label, _ in axes] + metrics
    rows = []
    for record in records:
        row = [record[key] for _, key in axes]
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


def catalogue(configs_dir: Path) -> str:
    """The configs on hand and what each asks for. Counted from the YAML
    rather than resolved, so listing costs no checkpoint lookup.
    """
    rows = []
    for path in sorted(configs_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        points = sum(
            math.prod(len(values) for values in spec.get("sweep", {}).values())
            for spec in specs(cfg)
        )
        asks = ["native"] if cfg.get("native") else []
        if points:
            asks.append(f"{points} accumulation{'s' * (points > 1)}")
        tasks = cfg.get("tasks", cfg.get("task", "?"))
        shown = tasks if isinstance(tasks, str) else ", ".join(tasks)
        rows.append((path.name, shown, " + ".join(asks) or "—"))

    if not rows:
        return f"No configs in {configs_dir}."
    widths = [max(len(row[i]) for row in rows) for i in range(2)]
    return "\n".join(
        f"  {name:<{widths[0]}}  {task:<{widths[1]}}  {asks}"
        for name, task, asks in rows
    )


def paths(item: Plan, results_dir: Path, stamp: str, override: Path | None) -> Path:
    """Where one config's results go, emptied so a re-run replaces them.

    A config may drop the timestamp, which makes the name a place rather than
    an event: `cross_arch/` commits its results so two machines can exchange
    them through git, and there the history is git's rather than the
    directory's.
    """
    if override is not None:
        # Named by hand, so added to rather than replaced, as before.
        override.parent.mkdir(parents=True, exist_ok=True)
        return override

    stem = item.path.stem + (f"-{stamp}" if item.cfg.get("timestamped", True) else "")
    path = results_dir / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.with_name(f"{stem}-samples.jsonl").unlink(missing_ok=True)
    return path


def write_samples(path: Path, tasks: list[str], results: dict) -> Path:
    """lm-eval's sample records, one per line, tagged with the task they came
    from — which is what a comparison of logprobs reads.
    """
    samples = path.with_name(f"{path.stem}-samples.jsonl")
    with samples.open("a") as handle:
        for task in tasks:
            for sample in results.get("samples", {}).get(task, []):
                handle.write(json.dumps({"task": task, **sample}, default=str) + "\n")
    return samples


def main(experiment_dir: Path) -> None:
    """Run the configs an example directory holds."""
    configs_dir = experiment_dir / "configs"
    results_dir = experiment_dir / "results"

    parser = argparse.ArgumentParser(
        prog=f"{experiment_dir.name}/run.py",
        description=(
            "Evaluate a task under one or more MMA accumulations.\n"
            "\n"
            "The kernels read their bitwidths when a layer is built, so every\n"
            "accumulation is its own engine and its own model load. Results\n"
            "land under results/ as they arrive."
        ),
        epilog=(
            "examples:\n"
            "  run.py configs/<name>.yaml\n"
            "  run.py configs/*.yaml --dry-run\n"
            "  run.py --list\n"
            "\n"
            "Start with --dry-run, and --limit to check the plumbing: a config\n"
            "that expands to more points than you meant is an hour you do not\n"
            "get back.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "configs", type=Path, nargs="*", help="experiment YAML, one or more"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_configs",
        help="print the configs on hand, and stop",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the runs the configs expand to, and stop",
    )
    parser.add_argument("--limit", type=int, help="override every config's `limit`")
    parser.add_argument(
        "--results",
        type=Path,
        help="JSONL to write (default: one per config, under results/)",
    )
    args = parser.parse_args()

    if args.list_configs:
        print(catalogue(configs_dir))
        return

    if not args.configs:
        parser.error("a config is required")
    if args.results and len(args.configs) > 1:
        parser.error("--results names one file, so it takes one config")

    plans = [plan(path, args.limit) for path in args.configs]

    for item in plans:
        cfg = item.cfg
        count = len(item.runs)
        tasks = ", ".join(cfg["tasks"])
        print(f"{item.path}: {count} run{'s' * (count > 1)} of {tasks}", end="")
        if cfg.get("limit"):
            print(f" (limit {cfg['limit']})", end="")
        print(f" against {cfg['model']} [{item.fmt}]\n")
        for i, run in enumerate(item.runs, 1):
            print(f"  {i:>2}. {run.name}")
            print(f"      {run.describe(cfg['model'])}")
        print(f"  on {device.info()}\n")

    if args.dry_run:
        print("Dry run: nothing was built or evaluated.")
        return

    for item in plans:
        cfg = item.cfg

        # Read before an engine exists, so set rather than passed. See
        # common/engine.py.
        settings = engine.resolve(cfg)
        os.environ.update(engine.env_vars(settings))
        if not settings["batch_invariant"]:
            print("batch_invariant: false — scores will vary between runs.\n")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = paths(item, results_dir, stamp, args.results)
        append(results_path, provenance(item, stamp))

        records = []
        written = [results_path]
        total = len(cfg["tasks"])
        for i, run in enumerate(item.runs, 1):
            print(
                f"[{i}/{len(item.runs)}] {run.name}: building the engine ...",
                flush=True,
            )
            for j, (task, results) in enumerate(evaluate(cfg, run), 1):
                metrics = results["results"].get(task, {})
                if not metrics:
                    print(f"  ! {task}: no scores — lm-eval takes leaf tasks")
                record = {
                    "name": run.name,
                    "config": run.mma_emu,
                    "task": task,
                    "metrics": metrics,
                }
                records.append(record)
                append(results_path, record)
                if cfg.get("log_samples"):
                    samples = write_samples(results_path, [task], results)
                    if samples not in written:
                        written.append(samples)
                print(f"  [{j}/{total}] {task}: done", flush=True)
            print(f"[{i}/{len(item.runs)}] {run.name}: done\n", flush=True)

        print(render(records))
        print("\nWrote " + "\n      ".join(str(path) for path in written) + "\n")
