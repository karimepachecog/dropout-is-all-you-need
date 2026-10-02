"""Plots and nearest-neighbor checks for the trained models.

Writes, under the output directory:

- alignment_uniformity.png
- similarity_by_rating.png
- retrievals.json, including one false friend and one missed paraphrase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from simcse.data import ROOT
from simcse.evaluate import (
    embed_texts,
    encode_transformer,
    evaluate_split,
    load_checkpoint_encoder,
    load_stsb,
    unique_in_order,
)

BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5.01)]
BUCKET_LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5"]


def embeddings_for_checkpoint(checkpoint: Path, texts: list[str], max_length: int, batch_size: int):
    encoder, tokenizer, pool, meta = load_checkpoint_encoder(checkpoint)
    unique, _index = unique_in_order(texts)
    vectors = encode_transformer(
        unique, encoder, tokenizer, pool=pool, batch_size=batch_size, max_length=max_length
    )
    return {text: vectors[index] for index, text in enumerate(unique)}, pool, meta


def plot_geometry(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        ax.scatter(row["uniformity"], row["alignment"], s=60)
        ax.annotate(row["name"], (row["uniformity"], row["alignment"]), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("uniformity (lower is more uniform)")
    ax.set_ylabel("alignment (lower is tighter)")
    ax.set_title("STS-B test: alignment vs uniformity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_buckets(named_cosines: dict[str, tuple[list[float], list[float]]], path: Path) -> None:
    names = list(named_cosines)
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cosines, scores = named_cosines[name]
        groups = []
        for low, high in BUCKETS:
            groups.append([cosine for cosine, score in zip(cosines, scores) if low <= score < high])
        ax.boxplot(groups, tick_labels=BUCKET_LABELS, showfliers=False)
        ax.set_title(name)
        ax.set_xlabel("human score")
        ax.set_ylabel("cosine")
        ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_neighbors(query: str, bank: list[str], vectors: torch.Tensor, index: dict[str, int], k: int = 5) -> list[dict]:
    query_vector = vectors[index[query]]
    scores = vectors @ query_vector
    scores[index[query]] = -2
    order = torch.argsort(scores, descending=True)[:k]
    return [{"sentence": bank[int(i)], "cosine": float(scores[int(i)])} for i in order]


def failure_cases(sentence1, sentence2, scores, cosines) -> dict:
    pairs = list(zip(sentence1, sentence2, scores, cosines))
    low_gold = [pair for pair in pairs if pair[2] < 1.0]
    high_gold = [pair for pair in pairs if pair[2] >= 4.0]
    false_friend = max(low_gold, key=lambda pair: pair[3])
    missed = min(high_gold, key=lambda pair: pair[3])
    return {
        "false_friend": {
            "sentence1": false_friend[0],
            "sentence2": false_friend[1],
            "gold": false_friend[2],
            "cosine": false_friend[3],
            "why": "Gold score is below 1, but this pair has the highest cosine in that bin.",
        },
        "missed_paraphrase": {
            "sentence1": missed[0],
            "sentence2": missed[1],
            "gold": missed[2],
            "cosine": missed[3],
            "why": "Gold score is at least 4, but this pair has the lowest cosine in that bin.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot alignment, uniformity, and retrievals.")
    parser.add_argument("--checkpoint", action="append", default=[], help="name=path/to/best")
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "analysis")
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    sentence1, sentence2, scores = load_stsb(args.split)
    texts = sentence1 + sentence2
    args.out.mkdir(parents=True, exist_ok=True)

    models: dict[str, dict[str, torch.Tensor]] = {}
    if args.include_references:
        models["BERT mean"] = embed_texts(
            texts, "transformer", "bert-base-uncased", "mean", args.max_length, args.batch_size
        )
        models["SBERT-2019"] = embed_texts(
            texts,
            "sbert",
            "sentence-transformers/bert-base-nli-mean-tokens",
            "mean",
            args.max_length,
            args.batch_size,
        )
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        embeddings, pool, _meta = embeddings_for_checkpoint(Path(path), texts, args.max_length, args.batch_size)
        models[f"{name} ({pool})"] = embeddings

    geometry_rows = []
    named_cosines = {}
    retrieval_report = {}
    for name, embeddings in models.items():
        metrics = evaluate_split(embeddings, sentence1, sentence2, scores)
        geometry_rows.append({"name": name, **metrics})
        left = [float((embeddings[a] * embeddings[b]).sum()) for a, b in zip(sentence1, sentence2)]
        named_cosines[name] = (left, scores)
        unique, index = unique_in_order(texts)
        bank = torch.stack([embeddings[text] for text in unique])
        queries = list(args.query)
        if not queries:
            queries = [sentence1[0], sentence1[len(sentence1) // 2]]
        retrieval_report[name] = {
            "metrics": metrics,
            "neighbors": {query: top_neighbors(query, unique, bank, index) for query in queries if query in index},
            "failures": failure_cases(sentence1, sentence2, scores, left),
        }
        print(f"{name}: spearman={metrics['spearman']:.2f} align={metrics['alignment']:.4f} uniform={metrics['uniformity']:.4f}")

    plot_geometry(geometry_rows, args.out / "alignment_uniformity.png")
    plot_buckets(named_cosines, args.out / "similarity_by_rating.png")
    payload = {"split": args.split, "models": retrieval_report}
    (args.out / "retrievals.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
