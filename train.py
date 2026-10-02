"""Train unsupervised or supervised SimCSE and keep the best dev checkpoint.

The test split is never loaded here. Score it once, after you have chosen
the checkpoint, with ``python -m simcse.evaluate --checkpoint ... --split test``.

Example:
    python train.py --mode unsupervised --seed 42
    python train.py --mode supervised --seed 42 --no_hard_negatives
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from simcse.data import ROOT, load_snli_rows, supervised_pairs, take_fraction, unique_sentences
from simcse.evaluate import encode_transformer, load_stsb, pair_cosine, spearman, unique_in_order
from simcse.model import SimCSE, load_tokenizer, pooling_for_eval


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_description(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return f"cuda:{name} ({memory_gb:.1f} GB)"
    if device.type == "mps":
        return f"mps:{platform.processor() or platform.machine()}"
    return f"cpu:{platform.processor() or platform.machine()}"


class SentenceDataset(Dataset):
    def __init__(self, sentences: list[str]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> str:
        return self.sentences[index]


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        return self.pairs[index]


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def make_collate(tokenizer, max_len: int, supervised: bool, hard_negatives: bool):
    def collate_sentences(sentences: list[str]) -> dict:
        tokens = tokenizer(
            sentences, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        return dict(tokens)

    def collate_pairs(pairs: list[dict]) -> dict:
        premises = [pair["premise"] for pair in pairs]
        positives = [pair["positive"] for pair in pairs]
        batch = {}
        premise_tokens = tokenizer(
            premises, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        positive_tokens = tokenizer(
            positives, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        batch["input_ids"] = premise_tokens["input_ids"]
        batch["attention_mask"] = premise_tokens["attention_mask"]
        if "token_type_ids" in premise_tokens:
            batch["token_type_ids"] = premise_tokens["token_type_ids"]
        batch["input_ids_pos"] = positive_tokens["input_ids"]
        batch["attention_mask_pos"] = positive_tokens["attention_mask"]
        if "token_type_ids" in positive_tokens:
            batch["token_type_ids_pos"] = positive_tokens["token_type_ids"]
        if hard_negatives:
            negatives = [
                pair["hard_negative"] if pair["hard_negative"] is not None else pair["positive"]
                for pair in pairs
            ]
            negative_tokens = tokenizer(
                negatives, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            )
            batch["input_ids_neg"] = negative_tokens["input_ids"]
            batch["attention_mask_neg"] = negative_tokens["attention_mask"]
            if "token_type_ids" in negative_tokens:
                batch["token_type_ids_neg"] = negative_tokens["token_type_ids"]
            batch["has_hard_negative"] = torch.tensor(
                [pair["hard_negative"] is not None for pair in pairs], dtype=torch.bool
            )
        return batch

    return collate_pairs if supervised else collate_sentences


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(model: SimCSE, tokenizer, directory: Path, meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    (directory / "training_meta.json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCSE.")
    parser.add_argument("--mode", choices=["unsupervised", "supervised"], required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "snli_train_100k.jsonl")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--eval_max_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--same_dropout", action="store_true")
    parser.add_argument("--no_hard_negatives", action="store_true")
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "unsupervised":
        args.batch_size = args.batch_size or 64
        args.lr = 3e-5 if args.lr is None else args.lr
        args.epochs = args.epochs or 1
        hard_negatives = False
    else:
        args.batch_size = args.batch_size or 128
        args.lr = 5e-5 if args.lr is None else args.lr
        args.epochs = args.epochs or 3
        hard_negatives = not args.no_hard_negatives

    set_seed(args.seed)
    device = pick_device(args.device)
    use_fp16 = device.type == "cuda"
    use_autocast = use_fp16

    rows = load_snli_rows(args.data)
    if args.mode == "unsupervised":
        data = take_fraction(unique_sentences(rows), args.data_fraction, args.seed)
        dataset = SentenceDataset(data)
    else:
        data = take_fraction(supervised_pairs(rows), args.data_fraction, args.seed)
        dataset = PairDataset(data)

    tokenizer = load_tokenizer(args.model_name)
    model = SimCSE(
        model_name=args.model_name,
        dropout=args.dropout,
        tau=args.tau,
        same_dropout=args.same_dropout,
        hard_negatives=hard_negatives,
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=make_collate(tokenizer, args.max_len, args.mode == "supervised", hard_negatives),
        generator=generator,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max(total_steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    fraction_tag = f"{args.data_fraction:g}"
    run_name = args.run_name or (
        f"{args.mode}_s{args.seed}_bs{args.batch_size}_lr{args.lr:g}_tau{args.tau:g}"
        f"_frac{fraction_tag}_samedrop{int(args.same_dropout)}_hardneg{int(hard_negatives)}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_name": run_name,
        "mode": args.mode,
        "model_name": args.model_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "tau": args.tau,
        "epochs": args.epochs,
        "max_len": args.max_len,
        "eval_max_len": args.eval_max_len,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "same_dropout": args.same_dropout,
        "hard_negatives": hard_negatives,
        "data_fraction": args.data_fraction,
        "n_examples": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "pooling_train": "cls",
        "pooling_eval": pooling_for_eval(args.mode),
        "hardware": hardware_description(device),
        "device": str(device),
        "torch_version": torch.__version__,
        "fp16": use_fp16,
        "autocast": use_autocast,
        "max_steps": args.max_steps,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))
    print(
        f"Expected initial loss near ln({args.batch_size}) = {torch.log(torch.tensor(float(args.batch_size))):.3f}"
        + (" plus the hard-negative columns" if hard_negatives else "")
    )

    dev_s1, dev_s2, dev_scores = load_stsb("validation")
    dev_texts, _dev_index = unique_in_order(dev_s1 + dev_s2)
    pool = pooling_for_eval(args.mode)

    best_dev = float("-inf")
    best_step = -1
    history = []
    step = 0
    started = time.time()
    model.train()
    done = False
    for epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps is not None and step >= args.max_steps:
                done = True
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type if use_autocast else "cpu", dtype=torch.float16, enabled=use_autocast):
                loss = model(batch, pool="cls")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            loss_value = float(loss.detach().cpu())
            if step == 1 or step % 50 == 0:
                print(f"step {step}/{total_steps} epoch {epoch} loss {loss_value:.4f}")
            if step == 1 and loss_value < 0.05 and not args.same_dropout:
                print(
                    "WARNING: loss is near zero on step 1. "
                    "The two views may be identical. Check that the model is in train() "
                    "and that dropout is not zero."
                )
            if step % args.eval_steps == 0 or step == total_steps:
                dev = dev_spearman(model, tokenizer, dev_texts, dev_s1, dev_s2, dev_scores, pool, args.eval_max_len, device)
                history.append({"step": step, "loss": loss_value, "dev_spearman": dev})
                print(f"dev spearman @ {step}: {dev:.2f}")
                if dev > best_dev:
                    best_dev = dev
                    best_step = step
                    save_checkpoint(
                        model,
                        tokenizer,
                        run_dir / "best",
                        {**config, "best_dev_spearman": best_dev, "best_step": best_step},
                    )
                    print(f"saved best checkpoint ({best_dev:.2f} at step {best_step})")
                model.train()
        if done:
            break

    if best_step < 0:
        save_checkpoint(model, tokenizer, run_dir / "best", {**config, "best_dev_spearman": None, "best_step": step})

    elapsed = time.time() - started
    summary = {
        **config,
        "best_dev_spearman": None if best_dev == float("-inf") else best_dev,
        "best_step": best_step,
        "elapsed_sec": elapsed,
        "history": history,
        "test_spearman": None,
    }
    (run_dir / "config.json").write_text(json.dumps(summary, indent=2))
    append_log(
        args.output_dir / "log.csv",
        {
            "run_name": run_name,
            "mode": args.mode,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "tau": args.tau,
            "epochs": args.epochs,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "hard_negatives": hard_negatives,
            "same_dropout": args.same_dropout,
            "data_fraction": args.data_fraction,
            "n_examples": len(dataset),
            "best_dev_spearman": summary["best_dev_spearman"],
            "best_step": best_step,
            "test_spearman": "",
            "alignment": "",
            "uniformity": "",
            "hardware": config["hardware"],
            "torch_version": torch.__version__,
            "elapsed_sec": round(elapsed, 1),
        },
    )
    print(f"Finished {run_name}. best dev {summary['best_dev_spearman']} at step {best_step}.")


@torch.no_grad()
def dev_spearman(model, tokenizer, texts, sentence1, sentence2, scores, pool, max_length, device) -> float:
    """Dev Spearman of the live encoder. Does not touch the test split."""
    vectors = encode_transformer(
        texts,
        model.encoder,
        tokenizer,
        pool=pool,
        batch_size=64,
        max_length=max_length,
        device=device,
    )
    by_text = {text: vectors[index] for index, text in enumerate(texts)}
    left = torch.stack([by_text[text] for text in sentence1])
    right = torch.stack([by_text[text] for text in sentence2])
    return spearman(pair_cosine(left, right), scores)


if __name__ == "__main__":
    main()
