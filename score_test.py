"""Score a chosen checkpoint on STS-B test, once.

Training never reads the test split. This script refuses to overwrite a
test number that is already stored, unless you pass --force.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from simcse.evaluate import evaluate_checkpoint


def update_log(log_path: Path, run_name: str, test_spearman: float, alignment: float, uniformity: float) -> None:
    if not log_path.exists():
        return
    with log_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    if not rows:
        return
    for row in rows:
        if row["run_name"] == run_name:
            row["test_spearman"] = f"{test_spearman:.4f}"
            row["alignment"] = f"{alignment:.4f}"
            row["uniformity"] = f"{uniformity:.4f}"
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot STS-B test evaluation.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("test_spearman") not in (None, "") and not args.force:
        raise SystemExit(
            f"{args.run_dir} already has test_spearman={config['test_spearman']}. "
            "The test split is used once. Pass --force only if that number was a dry run."
        )
    result = evaluate_checkpoint(args.run_dir / "best", ["test"], max_length=args.max_length)
    metrics = result["splits"]["test"]
    config["test_spearman"] = metrics["spearman"]
    config["test_alignment"] = metrics["alignment"]
    config["test_uniformity"] = metrics["uniformity"]
    config_path.write_text(json.dumps(config, indent=2))
    update_log(
        args.run_dir.parent / "log.csv",
        config["run_name"],
        metrics["spearman"],
        metrics["alignment"],
        metrics["uniformity"],
    )
    print(json.dumps({"run": config["run_name"], **metrics}, indent=2))


if __name__ == "__main__":
    main()
