"""SimCSE encoder and the contrastive loss.

Unsupervised (paper Eq. 1, section 3). The same sentence is encoded
twice in train mode. Dropout samples a new mask on each pass, so the
two vectors are a positive pair. Other sentences in the batch are the
negatives:

    L_i = -log exp(sim(h_i, h_i+) / tau) / sum_j exp(sim(h_i, h_j+) / tau)

With batch size N the denominator has N terms, so each step pushes the
positive away from N - 1 negatives. ``cross_entropy`` on the N x N
similarity matrix, labeled with arange(N), is exactly that loss.
A random batch has nearly equal similarities, so the loss starts near ln(N).

Supervised (paper Eq. 5, section 4). The positive is the entailment
hypothesis. The candidate list is the N positives concatenated with the
N contradiction hard negatives, so the denominator has 2N terms and
there are 2N - 1 negatives when every row has a hard negative. Rows
without a contradiction are masked to a large negative logit and do
not contribute.

The MLP is BERT's pooler (linear + tanh on the [CLS] state). Unsupervised
SimCSE trains with it and drops it at evaluation (``cls_before_pooler``).
Supervised SimCSE keeps it at evaluation (``cls``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token states, ignoring padding. Includes [CLS] and [SEP]."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def info_nce(anchors: torch.Tensor, positives: torch.Tensor, tau: float) -> torch.Tensor:
    """Eq. 1. Cosine similarity, then temperature, then in-batch cross-entropy."""
    anchors = F.normalize(anchors.float(), dim=-1)
    positives = F.normalize(positives.float(), dim=-1)
    logits = (anchors @ positives.T) / tau
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return F.cross_entropy(logits, labels)


def info_nce_with_hard_negatives(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    hard_negatives: torch.Tensor,
    has_hard_negative: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Eq. 5. Logits are N x 2N: all positives, then all hard negatives.

    ``has_hard_negative`` is a boolean vector of length N. A False column
    is filled with -1e4, so that candidate adds nothing to the softmax
    and receives no gradient.
    """
    anchors = F.normalize(anchors.float(), dim=-1)
    positives = F.normalize(positives.float(), dim=-1)
    hard_negatives = F.normalize(hard_negatives.float(), dim=-1)
    positive_logits = (anchors @ positives.T) / tau
    negative_logits = (anchors @ hard_negatives.T) / tau
    negative_logits = negative_logits.masked_fill(~has_hard_negative.unsqueeze(0), -1e4)
    logits = torch.cat([positive_logits, negative_logits], dim=1)
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return F.cross_entropy(logits, labels)


class SimCSE(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        tau: float = 0.05,
        same_dropout: bool = False,
        hard_negatives: bool = False,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        self.encoder = AutoModel.from_pretrained(model_name, config=config, attn_implementation="sdpa")
        self.tau = tau
        self.same_dropout = same_dropout
        self.hard_negatives = hard_negatives

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        pool: str = "cls",
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        if pool == "cls":
            return outputs.pooler_output
        if pool == "cls_before_pooler":
            return outputs.last_hidden_state[:, 0]
        if pool == "mean":
            return mean_pool(outputs.last_hidden_state, attention_mask)
        raise ValueError(f"Unknown pooling {pool!r}")

    def forward(self, batch: dict, pool: str = "cls") -> torch.Tensor:
        if "input_ids_pos" not in batch:
            first = self.encode(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"), pool=pool
            )
            if self.same_dropout:
                # Identical views: one mask, one vector. This is the
                # "fixed dropout" ablation (paper Table 3, 43.6 dev).
                second = first
            else:
                second = self.encode(
                    batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"), pool=pool
                )
            return info_nce(first, second, self.tau)

        anchors = self.encode(
            batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"), pool=pool
        )
        positives = self.encode(
            batch["input_ids_pos"], batch["attention_mask_pos"], batch.get("token_type_ids_pos"), pool=pool
        )
        if not self.hard_negatives:
            return info_nce(anchors, positives, self.tau)
        negatives = self.encode(
            batch["input_ids_neg"], batch["attention_mask_neg"], batch.get("token_type_ids_neg"), pool=pool
        )
        return info_nce_with_hard_negatives(
            anchors, positives, negatives, batch["has_hard_negative"], self.tau
        )


def load_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


def pooling_for_eval(mode: str) -> str:
    """Unsupervised drops the MLP at test time. Supervised keeps it."""
    if mode == "unsupervised":
        return "cls_before_pooler"
    if mode == "supervised":
        return "cls"
    raise ValueError(mode)
