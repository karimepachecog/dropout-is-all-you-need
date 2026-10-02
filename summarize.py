"""Collect seed noise and the two ablation deltas from runs/log.csv.

A difference is treated as larger than run-to-run noise when its absolute
dev (and test, if present) delta exceeds the standard deviation of the
three main seeds for that mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from simcse.data import ROOT


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest: dict[str, dict] = {}
    for row in rows:
        latest[row["run_name"]] = row
    return list(latest.values())


def as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def is_default_lr(row: dict) -> bool:
    """Seed noise is the three runs at the mode's default rate, not the learning-rate sweep."""
    lr = float(row["lr"])
    if row["mode"] == "unsupervised":
        return abs(lr - 3e-5) < 1e-12
    return abs(lr - 5e-5) < 1e-12


def is_main(row: dict) -> bool:
    return (
        row["same_dropout"] in {"False", "false", "0"}
        and float(row["data_fraction"]) == 1.0
        and is_default_lr(row)
        and (
            (row["mode"] == "unsupervised")
            or (row["mode"] == "supervised" and row["hard_negatives"] in {"True", "true", "1"})
        )
    )


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values)


def best_main(rows: list[dict], mode: str) -> dict:
    chosen = [row for row in rows if row["mode"] == mode and is_main(row)]
    if not chosen:
        raise SystemExit(f"No main {mode} runs in runs/log.csv")
    return max(chosen, key=lambda row: as_float(row["best_dev_spearman"]) or -1)


def find_one(rows: list[dict], predicate) -> dict | None:
    matches = [row for row in rows if predicate(row)]
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize seeds and ablations.")
    parser.add_argument("--log", type=Path, default=ROOT / "runs" / "log.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "ablation_summary.json")
    args = parser.parse_args()
    rows = load_rows(args.log)
    summary: dict = {"modes": {}, "ablations": []}
    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        if is_main(row):
            by_mode.setdefault(row["mode"], []).append(row)
    for mode, group in by_mode.items():
        devs = [as_float(row["best_dev_spearman"]) for row in group]
        devs = [value for value in devs if value is not None]
        summary["modes"][mode] = {
            "seeds": [int(row["seed"]) for row in group],
            "dev": devs,
            "dev_mean": statistics.mean(devs) if devs else None,
            "dev_std": stdev(devs) if devs else None,
            "runs": [row["run_name"] for row in group],
        }

    def paired(mode: str, predicate) -> dict | None:
        mains = [row for row in rows if row["mode"] == mode and is_main(row) and row["seed"] == "42"]
        alts = [row for row in rows if row["mode"] == mode and predicate(row) and row["seed"] == "42"]
        if not mains or not alts:
            return None
        base, alt = mains[-1], alts[-1]
        base_dev, alt_dev = as_float(base["best_dev_spearman"]), as_float(alt["best_dev_spearman"])
        base_test, alt_test = as_float(base["test_spearman"]), as_float(alt["test_spearman"])
        noise = summary["modes"].get(mode, {}).get("dev_std")
        dev_delta = None if base_dev is None or alt_dev is None else alt_dev - base_dev
        test_delta = None if base_test is None or alt_test is None else alt_test - base_test
        larger = None if dev_delta is None or noise is None else abs(dev_delta) > noise
        return {
            "base": base["run_name"],
            "ablation": alt["run_name"],
            "base_dev": base_dev,
            "ablation_dev": alt_dev,
            "dev_delta": dev_delta,
            "base_test": base_test,
            "ablation_test": alt_test,
            "test_delta": test_delta,
            "seed_dev_std": noise,
            "larger_than_seed_noise": larger,
        }

    same_mask = paired(
        "unsupervised",
        lambda row: row["same_dropout"] in {"True", "true", "1"} and float(row["data_fraction"]) == 1.0,
    )
    if same_mask:
        same_mask["name"] = "unsupervised same dropout mask"
        summary["ablations"].append(same_mask)
    no_hard = paired(
        "supervised",
        lambda row: row["hard_negatives"] in {"False", "false", "0"}
        and row["same_dropout"] in {"False", "false", "0"}
        and float(row["data_fraction"]) == 1.0,
    )
    if no_hard:
        no_hard["name"] = "supervised hard negatives off"
        summary["ablations"].append(no_hard)

    fractions = []
    for row in rows:
        if float(row["data_fraction"]) < 1.0 and row["seed"] == "42" and row["same_dropout"] in {"False", "false", "0"}:
            if row["mode"] == "supervised" and row["hard_negatives"] in {"False", "false", "0"}:
                continue
            fractions.append(
                {
                    "mode": row["mode"],
                    "fraction": float(row["data_fraction"]),
                    "dev": as_float(row["best_dev_spearman"]),
                    "run": row["run_name"],
                }
            )
    summary["data_fractions"] = fractions
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
