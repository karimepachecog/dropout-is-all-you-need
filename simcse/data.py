"""Build the two training sets from the shared SNLI subset.

The assignment fixes the counts, so this module checks them:

- unsupervised: the 165,529 unique sentences (premise and hypothesis,
  first-seen order, no whitespace stripping)
- supervised: the 33,351 entailment records (label 0), not the 33,333
  unique pairs. Eighteen duplicate records are kept on purpose.
- hard negative: the first contradiction (label 2) written for that
  premise. 9,488 / 33,351 pairs have one (28.4%). Pairs without one
  stay in the set; the loss masks their hard-negative column.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNLI = ROOT / "data" / "snli_train_100k.jsonl"

EXPECTED_SENTENCES = 165_529
EXPECTED_PAIRS = 33_351
EXPECTED_WITH_HARD_NEG = 9_488


def load_snli_rows(path: Path | str = DEFAULT_SNLI) -> list[tuple[str, str, int]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append((record["premise"], record["hypothesis"], int(record["label"])))
    return rows


def unique_sentences(rows: list[tuple[str, str, int]]) -> list[str]:
    seen: set[str] = set()
    sentences: list[str] = []
    for premise, hypothesis, _label in rows:
        for sentence in (premise, hypothesis):
            if sentence not in seen:
                seen.add(sentence)
                sentences.append(sentence)
    return sentences


def supervised_pairs(rows: list[tuple[str, str, int]]) -> list[dict]:
    """One training example per entailment record.

    Hard negative is the first contradiction hypothesis for the same
    premise, in file order. ``hard_negative`` is None when there is none.
    """
    contradictions: dict[str, str] = {}
    for premise, hypothesis, label in rows:
        if label == 2 and premise not in contradictions:
            contradictions[premise] = hypothesis

    pairs = []
    for premise, hypothesis, label in rows:
        if label != 0:
            continue
        pairs.append(
            {
                "premise": premise,
                "positive": hypothesis,
                "hard_negative": contradictions.get(premise),
            }
        )
    return pairs


def dataset_stats(rows: list[tuple[str, str, int]] | None = None, path: Path | str = DEFAULT_SNLI) -> dict:
    if rows is None:
        rows = load_snli_rows(path)
    sentences = unique_sentences(rows)
    pairs = supervised_pairs(rows)
    with_hard = sum(pair["hard_negative"] is not None for pair in pairs)
    return {
        "records": len(rows),
        "unique_sentences": len(sentences),
        "entailment_pairs": len(pairs),
        "pairs_with_hard_negative": with_hard,
        "hard_negative_fraction": with_hard / len(pairs) if pairs else 0.0,
        "counts_match_assignment": (
            len(sentences) == EXPECTED_SENTENCES
            and len(pairs) == EXPECTED_PAIRS
            and with_hard == EXPECTED_WITH_HARD_NEG
        ),
    }


def take_fraction(items: list, fraction: float, seed: int) -> list:
    """Prefix of a seed-shuffled copy. 25% is a subset of 50% of 100%."""
    if fraction >= 1.0:
        return list(items)
    if not 0 < fraction < 1:
        raise ValueError(f"data_fraction must be in (0, 1], got {fraction}")
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    keep = max(1, int(len(items) * fraction))
    return [items[index] for index in order[:keep]]


def load_stsb(split: str):
    """STS-B from sentence-transformers/stsb.

    ``dev`` and ``validation`` both load the 1,500-pair development split.
    ``test`` is the 1,379-pair test split. The hub stores scores in [0, 1];
    they are multiplied by 5 so the rest of the project can use the
    original 0-5 ratings. Spearman does not change.
    """
    from datasets import load_dataset

    name = {"dev": "validation", "validation": "validation", "test": "test", "train": "train"}[split]
    dataset = load_dataset("sentence-transformers/stsb", split=name)
    sentence1 = list(dataset["sentence1"])
    sentence2 = list(dataset["sentence2"])
    # sentence-transformers/stsb stores the gold score in [0, 1]. The
    # assignment and the paper talk about the original 0-5 ratings, and
    # Spearman is unchanged by this rescaling.
    scores = [float(score) * 5.0 for score in dataset["score"]]
    return sentence1, sentence2, scores


def main() -> None:
    stats = dataset_stats()
    print(json.dumps(stats, indent=2))
    if not stats["counts_match_assignment"]:
        raise SystemExit(
            "Counts do not match the assignment "
            f"({EXPECTED_SENTENCES} sentences, {EXPECTED_PAIRS} pairs, "
            f"{EXPECTED_WITH_HARD_NEG} with a hard negative)."
        )


if __name__ == "__main__":
    main()
