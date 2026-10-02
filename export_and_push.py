"""Export a checkpoint as a sentence-transformers model and optionally push it.

Unsupervised checkpoints are Transformer + CLS pooling, with the MLP removed.
Supervised checkpoints add the BERT pooler as a Dense(tanh) module.

After export, the script reloads the saved model, reruns STS-B, and checks
that the Spearman matches the in-project evaluator. ``--push`` then uploads
that same folder and reloads it from the Hub for the same check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from simcse.data import ROOT, load_stsb
from simcse.evaluate import evaluate_checkpoint, pair_cosine, spearman, unique_in_order


def build_sentence_transformer(checkpoint: Path, mode: str, max_seq_length: int):
    from sentence_transformers import SentenceTransformer, models
    from transformers import AutoModel

    transformer = models.Transformer(str(checkpoint), max_seq_length=max_seq_length)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
    )
    modules: list = [transformer, pooling]
    if mode == "supervised":
        encoder = AutoModel.from_pretrained(str(checkpoint))
        dense = models.Dense(
            in_features=encoder.config.hidden_size,
            out_features=encoder.config.hidden_size,
            bias=True,
            activation_function=nn.Tanh(),
        )
        with torch.no_grad():
            dense.linear.weight.copy_(encoder.pooler.dense.weight)
            dense.linear.bias.copy_(encoder.pooler.dense.bias)
        modules.append(dense)
    return SentenceTransformer(modules=modules)


def sbert_spearman(model, split: str, batch_size: int) -> float:
    sentence1, sentence2, scores = load_stsb(split)
    unique, index = unique_in_order(sentence1 + sentence2)
    vectors = model.encode(
        unique, batch_size=batch_size, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    ).float().cpu()
    by_text = {text: vectors[index[text]] for text in unique}
    left = torch.stack([by_text[text] for text in sentence1])
    right = torch.stack([by_text[text] for text in sentence2])
    return spearman(pair_cosine(left, right), scores)


def batch_limitation(meta: dict) -> str:
    """Unsupervised BERT-base uses batch 64 in the paper. Supervised uses 512."""
    batch = meta.get("batch_size")
    if meta.get("mode") == "unsupervised":
        return f"The unsupervised batch size is {batch}, the same as the paper's BERT-base setting."
    return (
        f"The supervised batch size is {batch}, not the paper's 512, "
        "and the learning rate was not retuned for that."
    )


def write_model_card(directory: Path, meta: dict, metrics: dict, repo_id: str | None) -> None:
    card = f"""---
language: en
license: apache-2.0
tags:
  - sentence-transformers
  - sentence-similarity
  - feature-extraction
  - dense
  - simcse
pipeline_tag: sentence-similarity
library_name: sentence-transformers
datasets:
  - stanfordnlp/snli
base_model: bert-base-uncased
---

# {repo_id or directory.name}

Sentence embedding model trained with SimCSE (Gao, Yao and Chen, 2021) from `bert-base-uncased`.

## Training data

A shared 100k-record subset of SNLI (`snli_train_100k.jsonl`), not the paper's Wikipedia or SNLI+MNLI training set.

- Mode: `{meta.get("mode")}`
- Examples used: {meta.get("n_examples")} ({meta.get("data_fraction")} of the constructed set)
- Unsupervised set: 165,529 unique SNLI sentences
- Supervised set: 33,351 entailment pairs; a contradiction hard negative exists for 9,488 of them (28.4%) and is masked out otherwise

## Recipe

| | |
| --- | --- |
| pooling at train | CLS + BERT pooler MLP |
| pooling at test | `{meta.get("pooling_eval")}` |
| batch size | {meta.get("batch_size")} |
| learning rate | {meta.get("lr")} |
| temperature | {meta.get("tau")} |
| epochs | {meta.get("epochs")} |
| max length | {meta.get("max_len")} |
| dropout | {meta.get("dropout")} |
| seed | {meta.get("seed")} |
| hard negatives | {meta.get("hard_negatives")} |
| hardware | {meta.get("hardware")} |

Optimizer: AdamW, weight decay {meta.get("weight_decay")}, warmup ratio {meta.get("warmup_ratio")}, linear decay.

## Metrics

STS-B Spearman x100, cosine similarity, no regressor.

| split | Spearman |
| --- | --- |
| dev | {meta.get("best_dev_spearman")} |
| test | {metrics.get("test_spearman")} |

Alignment (STS-B test, gold >= 4): {metrics.get("alignment")}

Uniformity (STS-B test, t=2): {metrics.get("uniformity")}

## Limitations

- English only, and the encoder is uncased.
- Trained on SNLI captions, which are shorter and more concrete than the Wikipedia sentences used for unsupervised SimCSE in the paper.
- The supervised set is about an order of magnitude smaller than SNLI+MNLI, and most pairs have no contradiction hard negative.
- {batch_limitation(meta)}
- Not a substitute for the published SimCSE checkpoints if you need the paper's STS-B numbers.
"""
    (directory / "README.md").write_text(card)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SimCSE checkpoint to sentence-transformers.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["unsupervised", "supervised"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="test")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    meta_path = args.checkpoint / "training_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"mode": args.mode}
    ours = evaluate_checkpoint(args.checkpoint, [args.split], max_length=args.max_seq_length, batch_size=args.batch_size)
    ours_score = ours["splits"][args.split]["spearman"]

    model = build_sentence_transformer(args.checkpoint, args.mode, args.max_seq_length)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))

    from sentence_transformers import SentenceTransformer

    reloaded = SentenceTransformer(str(args.out))
    local_score = sbert_spearman(reloaded, args.split, args.batch_size)
    print(f"evaluator {ours_score:.4f}  reloaded sentence-transformers {local_score:.4f}")
    if abs(local_score - ours_score) > 0.05:
        raise SystemExit("Reloaded model does not match the evaluator. Check pooling and the MLP.")

    metrics = {
        "test_spearman": ours_score,
        "reloaded_spearman": local_score,
        "alignment": ours["splits"][args.split]["alignment"],
        "uniformity": ours["splits"][args.split]["uniformity"],
    }
    write_model_card(args.out, meta, metrics, args.repo_id)
    (args.out / "export_metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.push:
        print(f"Saved to {args.out}. Re-run with --push to upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = args.repo_id or f"{user}/simcse-{args.mode}-snli100k"
    reloaded.push_to_hub(repo_id, exist_ok=True)
    # push_to_hub replaces README.md with the library's empty card.
    write_model_card(args.out, meta, metrics, repo_id)
    api.upload_file(
        path_or_fileobj=str(args.out / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Restore the SimCSE model card",
    )
    from_hub = SentenceTransformer(repo_id)
    hub_score = sbert_spearman(from_hub, args.split, args.batch_size)
    print(f"hub reload {hub_score:.4f} (local {local_score:.4f})")
    if abs(hub_score - local_score) > 0.05:
        raise SystemExit("Hub reload does not match the local export.")
    metrics["repo_id"] = repo_id
    metrics["hub_spearman"] = hub_score
    (args.out / "export_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "runs" / f"hub_{args.mode}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Pushed and verified {repo_id}")


if __name__ == "__main__":
    main()
