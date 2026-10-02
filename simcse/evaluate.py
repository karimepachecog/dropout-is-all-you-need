"""STS-B evaluation, alignment, and uniformity.

Protocol, matching the assignment and SimCSE Appendix B:

- embed both sentences
- L2-normalize
- cosine similarity (the dot product of the normalized vectors)
- Spearman correlation with the human scores, times 100
- no regressor

Alignment and uniformity follow Wang and Isola (2020), with features
L2-normalized. Alignment is the mean squared distance of STS-B pairs
whose gold score is at least 4 (out of 5). Uniformity uses t = 2 over
distinct sentences in the split:

    align = mean ||f(x) - f(y)||^2
    uniform = log mean exp(-2 ||f(x) - f(y)||^2)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from simcse.data import ROOT, load_stsb
from simcse.model import load_tokenizer, mean_pool

REFERENCE = {
    "bert-base-uncased-mean": {"validation": 59.31, "test": 47.29},
    "sbert-2019": {"validation": 80.77, "test": 76.98},
}


def spearman(predictions: list[float], gold: list[float]) -> float:
    from scipy.stats import spearmanr

    correlation, _pvalue = spearmanr(predictions, gold)
    return float(correlation) * 100.0


def alignment_uniformity_from_texts(
    embeddings: torch.Tensor,
    texts: list[str],
    pair_index_a: list[int],
    pair_index_b: list[int],
    scores: list[float],
    positive_threshold: float = 4.0,
) -> dict[str, float]:
    positive = [index for index, score in enumerate(scores) if score >= positive_threshold]
    if not positive:
        align = float("nan")
        n_positive = 0
    else:
        left = embeddings[pair_index_a][positive]
        right = embeddings[pair_index_b][positive]
        align = float((left - right).pow(2).sum(dim=-1).mean())
        n_positive = len(positive)
    similarity = embeddings @ embeddings.T
    squared = (2 - 2 * similarity).clamp(min=0)
    mask = ~torch.eye(embeddings.size(0), dtype=torch.bool)
    uniform = float(torch.log(torch.exp(-2 * squared[mask]).mean()))
    return {
        "alignment": align,
        "uniformity": uniform,
        "n_positive_pairs": n_positive,
        "n_uniformity_sentences": int(embeddings.size(0)),
    }


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_transformer(
    texts: list[str],
    encoder,
    tokenizer,
    pool: str,
    batch_size: int = 64,
    max_length: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or _device()
    encoder.eval()
    encoder.to(device)
    vectors = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        tokens = tokenizer(
            chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = encoder(**tokens)
        if pool == "mean":
            hidden = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        elif pool == "cls":
            hidden = outputs.pooler_output
        elif pool == "cls_before_pooler":
            hidden = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError(pool)
        vectors.append(F.normalize(hidden.float(), dim=-1).cpu())
    return torch.cat(vectors, dim=0)


@torch.no_grad()
def encode_sbert(texts: list[str], model_name: str, batch_size: int = 64) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=str(_device()))
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return encoded.float().cpu()


def pair_cosine(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    return (left * right).sum(dim=-1).tolist()


def unique_in_order(texts: list[str]) -> tuple[list[str], dict[str, int]]:
    index: dict[str, int] = {}
    unique: list[str] = []
    for text in texts:
        if text not in index:
            index[text] = len(unique)
            unique.append(text)
    return unique, index


def evaluate_split(
    embeddings_by_text: dict[str, torch.Tensor],
    sentence1: list[str],
    sentence2: list[str],
    scores: list[float],
) -> dict[str, float]:
    left = torch.stack([embeddings_by_text[text] for text in sentence1])
    right = torch.stack([embeddings_by_text[text] for text in sentence2])
    cosine = pair_cosine(left, right)
    unique_texts, index = unique_in_order(sentence1 + sentence2)
    bank = torch.stack([embeddings_by_text[text] for text in unique_texts])
    geometry = alignment_uniformity_from_texts(
        bank,
        unique_texts,
        [index[text] for text in sentence1],
        [index[text] for text in sentence2],
        scores,
    )
    return {
        "spearman": spearman(cosine, scores),
        "alignment": geometry["alignment"],
        "uniformity": geometry["uniformity"],
        "n_pairs": len(scores),
        "n_positive_pairs": geometry["n_positive_pairs"],
        "n_uniformity_sentences": geometry["n_uniformity_sentences"],
    }


def embed_texts(texts: list[str], kind: str, model_name: str, pool: str, max_length: int, batch_size: int) -> dict[str, torch.Tensor]:
    unique, _index = unique_in_order(texts)
    if kind == "sbert":
        vectors = encode_sbert(unique, model_name, batch_size=batch_size)
    else:
        from transformers import AutoModel

        tokenizer = load_tokenizer(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        vectors = encode_transformer(
            unique, encoder, tokenizer, pool=pool, batch_size=batch_size, max_length=max_length
        )
    return {text: vectors[index] for index, text in enumerate(unique)}


def evaluate_model(
    kind: str,
    model_name: str,
    pool: str,
    splits: list[str],
    max_length: int = 128,
    batch_size: int = 64,
) -> dict:
    results = {}
    for split in splits:
        sentence1, sentence2, scores = load_stsb(split)
        embeddings = embed_texts(sentence1 + sentence2, kind, model_name, pool, max_length, batch_size)
        metrics = evaluate_split(embeddings, sentence1, sentence2, scores)
        metrics["split"] = split
        results[split] = metrics
        print(
            f"{model_name} pool={pool} {split}: spearman={metrics['spearman']:.2f} "
            f"align={metrics['alignment']:.4f} uniform={metrics['uniformity']:.4f} "
            f"n={metrics['n_pairs']}"
        )
    return {"model": model_name, "kind": kind, "pool": pool, "max_length": max_length, "splits": results}


def load_checkpoint_encoder(checkpoint: Path):
    from transformers import AutoModel

    meta_path = checkpoint / "training_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tokenizer = load_tokenizer(str(checkpoint))
    encoder = AutoModel.from_pretrained(str(checkpoint))
    pool = meta.get("pooling_eval", "cls_before_pooler")
    return encoder, tokenizer, pool, meta


def evaluate_checkpoint(checkpoint: Path, splits: list[str], max_length: int = 128, batch_size: int = 64) -> dict:
    encoder, tokenizer, pool, meta = load_checkpoint_encoder(checkpoint)
    results = {}
    for split in splits:
        sentence1, sentence2, scores = load_stsb(split)
        unique, _index = unique_in_order(sentence1 + sentence2)
        vectors = encode_transformer(
            unique, encoder, tokenizer, pool=pool, batch_size=batch_size, max_length=max_length
        )
        embeddings = {text: vectors[index] for index, text in enumerate(unique)}
        metrics = evaluate_split(embeddings, sentence1, sentence2, scores)
        metrics["split"] = split
        results[split] = metrics
        print(
            f"{checkpoint} pool={pool} {split}: spearman={metrics['spearman']:.2f} "
            f"align={metrics['alignment']:.4f} uniform={metrics['uniformity']:.4f}"
        )
    return {"checkpoint": str(checkpoint), "pool": pool, "meta": meta, "splits": results}


def check_references(max_length: int = 128, batch_size: int = 64, tolerance: float = 0.2) -> dict:
    """Score the two fixed models. Fail if a number is far from the notes."""
    bert = evaluate_model(
        "transformer", "bert-base-uncased", "mean", ["validation", "test"], max_length, batch_size
    )
    sbert = evaluate_model(
        "sbert",
        "sentence-transformers/bert-base-nli-mean-tokens",
        "mean",
        ["validation", "test"],
        max_length,
        batch_size,
    )
    report = {"bert-base-uncased-mean": bert, "sbert-2019": sbert, "ok": True, "deltas": {}}
    for key, payload, ref_key in (
        ("bert-base-uncased-mean", bert, "bert-base-uncased-mean"),
        ("sbert-2019", sbert, "sbert-2019"),
    ):
        for split, expected in REFERENCE[ref_key].items():
            got = payload["splits"][split]["spearman"]
            delta = got - expected
            report["deltas"][f"{key}:{split}"] = {"got": got, "expected": expected, "delta": delta}
            if abs(delta) > tolerance:
                report["ok"] = False
    out = ROOT / "runs" / "reference_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["deltas"], indent=2))
    if not report["ok"]:
        raise SystemExit(
            f"Reference Spearman is more than {tolerance} away from the notes. "
            "The bug is in evaluation, not in training. See runs/reference_eval.json."
        )
    print(f"Reference check passed (tolerance {tolerance}). Wrote {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on STS-B.")
    parser.add_argument("--check-references", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--kind", choices=["transformer", "sbert"], default="transformer")
    parser.add_argument("--pool", choices=["mean", "cls", "cls_before_pooler"], default="mean")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=0.2)
    args = parser.parse_args()
    splits = args.splits or ["validation"]

    if args.check_references:
        check_references(args.max_length, args.batch_size, args.tolerance)
        return
    if args.checkpoint:
        payload = evaluate_checkpoint(args.checkpoint, splits, args.max_length, args.batch_size)
    else:
        payload = evaluate_model(args.kind, args.model, args.pool, splits, args.max_length, args.batch_size)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
