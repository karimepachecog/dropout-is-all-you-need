"""The shared JSONL has to produce the counts in the assignment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simcse.data import (
    EXPECTED_PAIRS,
    EXPECTED_SENTENCES,
    EXPECTED_WITH_HARD_NEG,
    dataset_stats,
)


def test_assignment_counts():
    stats = dataset_stats()
    assert stats["records"] == 100_000
    assert stats["unique_sentences"] == EXPECTED_SENTENCES
    assert stats["entailment_pairs"] == EXPECTED_PAIRS
    assert stats["pairs_with_hard_negative"] == EXPECTED_WITH_HARD_NEG
    assert stats["counts_match_assignment"]


if __name__ == "__main__":
    test_assignment_counts()
    print("data tests passed", dataset_stats())
