#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the two sides of the cross-architecture check.

Both sides produce a score per task and a logprob per answer choice. Scores
agreeing is the weaker claim, and the one a table of accuracies would stop at:
two runs can round to the same number while disagreeing about which questions
they got right. So the logprobs are compared too, and for equality rather than
closeness — the emulation reproduces the hardware's arithmetic, so anything
short of every bit is a result about something else.

No GPU is needed here; this only reads JSON. Given nothing to compare, it
compares what is in `results/`, which is the whole of it once both machines
have run.

    python cross_arch/compare.py
    python cross_arch/compare.py results/A.jsonl results/B.jsonl --out cmp.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# lm-eval mixes these counts in with the scores. They are worth comparing but
# are not results, so they are checked without being tabulated.
COUNTS = {"sample_len"}


def emulated(path: Path) -> bool:
    """Whether a result is the emulated side, read off its provenance line."""
    try:
        with path.open() as handle:
            return json.loads(handle.readline() or "{}").get("accumulation") is not None
    except (OSError, json.JSONDecodeError):
        return False


def discover() -> tuple[Path, Path]:
    """The pair in `results/`, for when the command was given no arguments.

    Two sides is the whole experiment, so anything else is a state worth
    naming rather than a guess worth making.
    """
    found = sorted(
        path
        for path in RESULTS_DIR.glob("*.jsonl")
        if not path.stem.endswith("-samples")
    )
    if not found:
        raise SystemExit(
            f"No results in {RESULTS_DIR}. Run one side on each machine, bring "
            f"both here, then compare — see README.md."
        )
    if len(found) == 1:
        raise SystemExit(
            f"Only {found[0].name} is in {RESULTS_DIR}. The other machine's side "
            f"has to be here too — see README.md."
        )
    if len(found) > 2:
        listed = "\n".join(f"  {path.name}" for path in found)
        raise SystemExit(
            f"{len(found)} results in {RESULTS_DIR}, so name the two to "
            f"compare:\n{listed}"
        )

    # The emulation first, since the table reads as what it reproduces.
    return tuple(sorted(found, key=emulated, reverse=True))  # type: ignore[return-value]


def load_scores(path: Path) -> tuple[dict, dict]:
    """`(provenance, {task: metrics})` from one side's result file."""
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run that side first, or copy it here."
        )

    head: dict = {}
    scores: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "provenance":
            head = record
        elif "task" in record:
            scores[record["task"]] = record.get("metrics", {})

    if not head:
        raise SystemExit(f"{path} has no provenance line; it is not a result file.")
    return head, scores


def logprobs_of(sample: dict) -> list[float]:
    """The per-choice logprobs of one lm-eval sample.

    `filtered_resps` holds one entry per request. Multiple-choice tasks store
    `[logprob, is_greedy]` per answer choice; a rolling loglikelihood such as
    wikitext stores the logprob bare. Both turn up here.
    """
    out = []
    for resp in sample.get("filtered_resps", []):
        if isinstance(resp, (list, tuple)) and resp:
            out.append(float(resp[0]))
        elif isinstance(resp, (int, float)):
            out.append(float(resp))
    return out


def load_samples(path: Path) -> dict | None:
    """`{task: {doc_id: (prompt_hash, logprobs)}}` from the samples sidecar.

    Read a line at a time and keep only what the comparison needs: the file
    holds every prompt and response, which is worth committing but not worth
    holding twice in memory.
    """
    sidecar = path.with_name(f"{path.stem}-samples.jsonl")
    if not sidecar.exists():
        return None

    index: dict[str, dict] = {}
    with sidecar.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            task = sample.get("task")
            doc_id = sample.get("doc_id")
            if task is None or doc_id is None:
                continue
            entry = (sample.get("prompt_hash"), logprobs_of(sample))
            index.setdefault(task, {})[doc_id] = entry
    return index


def task_stats(left: dict, right: dict) -> dict:
    """How far one task's logprobs agree, sample by sample."""
    shared = sorted(set(left) & set(right))
    stats = {
        "total": 0,
        "identical": 0,
        "max_diff": 0.0,
        "hash_mismatch": 0,
        "only_one_side": len(set(left) ^ set(right)),
    }
    for doc_id in shared:
        (left_hash, left_lp), (right_hash, right_lp) = left[doc_id], right[doc_id]
        if left_hash != right_hash:
            stats["hash_mismatch"] += 1
        for a, b in zip(left_lp, right_lp):
            stats["total"] += 1
            if a == b:
                stats["identical"] += 1
            else:
                stats["max_diff"] = max(stats["max_diff"], abs(a - b))
    return stats


def numeric(metrics: dict) -> dict:
    """The metrics that carry a number, which are the ones worth comparing."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def plural(count: int) -> str:
    """`n samples`, and `1 sample` when that is what it is."""
    return f"{count:,} sample{'' if count == 1 else 's'}"


def fmt(value: float) -> str:
    """Enough digits to show that agreement is exact rather than rounded."""
    return f"{value:.6f}" if abs(value) < 100 else f"{value:.4f}"


def side(head: dict, path: Path) -> str:
    """One line saying what a result is and where it came from."""
    device = head.get("device") or {}
    arch = device.get("arch") or "unknown"
    return (
        f"{head.get('name', '?')} on {arch} ({device.get('name', '?')}) — {path.name}"
    )


def compare(left_path: Path, right_path: Path) -> tuple[list[str], bool]:
    """Print both sides against each other; return the markdown and the verdict."""
    left_head, left_scores = load_scores(left_path)
    right_head, right_scores = load_scores(right_path)

    md = ["# Cross-architecture reproduction\n"]
    print(f"\n{'=' * 78}")
    print("Cross-architecture reproduction")
    print(f"{'=' * 78}")
    for head, path in ((left_head, left_path), (right_head, right_path)):
        print(f"  {side(head, path)}")
        md.append(f"- {side(head, path)}")

    left_arch = (left_head.get("device") or {}).get("arch")
    right_arch = (right_head.get("device") or {}).get("arch")
    if left_arch is not None and left_arch == right_arch:
        note = (
            f"both sides ran on {left_arch} — this is not a cross-architecture "
            f"comparison, and agreement here shows nothing"
        )
        print(f"\n  !! {note}")
        md.append(f"\n**Warning:** {note}\n")

    # Everything outside the accumulation is supposed to be pinned to the same
    # value on both sides; if it is not, a difference below means nothing.
    for key in ("model", "engine", "limit", "tasks"):
        if left_head.get(key) != right_head.get(key):
            note = f"the two sides disagree on `{key}`, so they are not comparable"
            print(f"  !! {note}")
            md.append(f"\n**Warning:** {note}\n")

    left_samples, right_samples = load_samples(left_path), load_samples(right_path)
    have_samples = left_samples is not None and right_samples is not None
    if not have_samples:
        missing = left_path if left_samples is None else right_path
        print(f"\n  (no samples sidecar for {missing.name}; comparing scores only)")
        md.append(f"\n_No samples sidecar for {missing.name}; scores only._\n")

    header = (
        f"  {'task':16}{'metric':18}{'score':>12}   "
        f"{'logprobs (identical/total)':>27}   match"
    )
    print(f"\n{header}\n  {'-' * (len(header) - 2)}")
    md.append("\n| task | metric | score | logprobs (identical/total) | match |")
    md.append("| --- | --- | ---: | ---: | :---: |")

    tasks = sorted(set(left_scores) | set(right_scores))
    ok_tasks = lp_identical = lp_total = 0
    details: list[str] = []

    for task in tasks:
        left_metrics = numeric(left_scores.get(task, {}))
        right_metrics = numeric(right_scores.get(task, {}))
        names = sorted(
            key
            for key in set(left_metrics) | set(right_metrics)
            if "_stderr" not in key
        )
        # `sample_len` is how many documents the task read. It is compared —
        # two sides that read different documents are not comparable — but it
        # is not a score, and a table saying one is 5.000000 reads as if it is.
        shown = [key for key in names if key not in COUNTS]

        stats = None
        if have_samples and task in left_samples and task in right_samples:
            stats = task_stats(left_samples[task], right_samples[task])
            lp_identical += stats["identical"]
            lp_total += stats["total"]
            samples_ok = (
                stats["identical"] == stats["total"]
                and stats["hash_mismatch"] == 0
                and stats["only_one_side"] == 0
            )
            cell = f"{stats['identical']:,} / {stats['total']:,}"
        else:
            samples_ok, cell = True, "—"

        scores_ok = all(
            left_metrics.get(key) == right_metrics.get(key) for key in names
        )
        task_ok = scores_ok and samples_ok
        ok_tasks += task_ok

        for i, key in enumerate(shown or ["—"]):
            value = left_metrics.get(key, right_metrics.get(key))
            shown = fmt(value) if value is not None else "—"
            mark = ("✓" if task_ok else "✗") if i == 0 else ""
            print(
                f"  {task:16}{key:18}{shown:>12}   {cell if not i else '':>27}   {mark}"
            )
            md.append(
                f"| {task if not i else ''} | {key} | {shown} | "
                f"{cell if not i else ''} | {mark} |"
            )

        for key in names:
            a, b = left_metrics.get(key), right_metrics.get(key)
            if a == b:
                continue
            if a is None or b is None:
                details.append(f"{task}/{key}: present on only one side")
            else:
                details.append(
                    f"{task}/{key}: {fmt(a)} vs {fmt(b)} (delta {a - b:+.3g})"
                )

        if stats is not None and not samples_ok:
            # Only what actually went wrong: a run that asked different
            # questions and a run that answered them differently are separate
            # failures, and saying both every time hides which one happened.
            parts = []
            differing = stats["total"] - stats["identical"]
            if differing:
                parts.append(
                    f"{differing:,}/{stats['total']:,} logprobs differ, "
                    f"max|delta| {stats['max_diff']:.3g}"
                )
            if stats["hash_mismatch"]:
                parts.append(
                    f"{plural(stats['hash_mismatch'])} with a different prompt"
                )
            if stats["only_one_side"]:
                parts.append(f"{plural(stats['only_one_side'])} on one side only")
            details.append(f"{task}: " + "; ".join(parts))

    print(f"  {'-' * (len(header) - 2)}")
    if have_samples:
        summary = (
            f"{ok_tasks}/{len(tasks)} tasks identical  ·  "
            f"{lp_identical:,}/{lp_total:,} logprobs bit-identical"
        )
    else:
        summary = (
            f"{ok_tasks}/{len(tasks)} tasks with identical scores "
            f"(logprobs not checked)"
        )
    print(f"  {summary}\n")
    md.append(f"\n**{summary}**\n")

    for detail in details:
        print(f"  ! {detail}")
        md.append(f"- {detail}")

    return md, ok_tasks == len(tasks) and bool(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the two sides of the cross-architecture check.\n"
            "\n"
            "Given no arguments, compares the pair in results/. Scores are\n"
            "compared for equality and logprobs bit for bit. Exits non-zero if\n"
            "any task disagrees, so a script can gate on it."
        ),
        epilog=(
            "examples:\n"
            "  compare.py                                  # the pair in results/\n"
            "  compare.py results/A.jsonl results/B.jsonl --out cmp.md\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sides",
        type=Path,
        nargs="*",
        help="the two results/<config>.jsonl (default: whichever pair is there)",
    )
    parser.add_argument(
        "--out", type=Path, help="also write the comparison as markdown to this file"
    )
    args = parser.parse_args()

    if len(args.sides) == 1:
        parser.error("name both sides or neither; one on its own has nothing to meet")
    if len(args.sides) > 2:
        parser.error("a comparison is between two sides")
    left, right = args.sides if args.sides else discover()

    md, agreed = compare(left, right)
    if args.out:
        args.out.write_text("\n".join(md) + "\n")
        print(f"\nMarkdown written to {args.out}")

    raise SystemExit(0 if agreed else 1)


if __name__ == "__main__":
    main()
