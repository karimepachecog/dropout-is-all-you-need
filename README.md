# SimCSE on a 100k SNLI subset

Train unsupervised and supervised [SimCSE](https://arxiv.org/abs/2104.08821) (Gao, Yao, and Chen, 2021) from `bert-base-uncased` on a fixed 100,000-record subset of SNLI, then score sentence embeddings on STS-B with Spearman correlation of cosine similarity. No regressor.

Unsupervised SimCSE encodes each sentence twice. Dropout is the only difference between the two views, so the matching pair is the positive and the other sentences in the batch are negatives. Supervised SimCSE uses an SNLI entailment as the positive and, when one exists, a contradiction as a hard negative.

STS-B comes from `sentence-transformers/stsb`. The Hub stores scores in `[0, 1]`; the loader rescales them to the original `0–5` ratings. Spearman is unchanged by that rescaling.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

Check that `data/snli_train_100k.jsonl` produces the expected counts (100,000 records, 165,529 unique sentences, 33,351 entailment pairs, 9,488 with a hard negative):

```bash
python -m simcse.data
python tests/test_data.py
python tests/test_loss.py
```

## Train

`train.py` never loads the STS-B test split. It writes `runs/<run_name>/config.json`, keeps the best dev checkpoint in `runs/<run_name>/best`, and appends a row to `runs/log.csv`. Dev is STS-B validation, every 250 steps.

```bash
python train.py --mode unsupervised --seed 42
python train.py --mode supervised --seed 42
```

| | unsupervised | supervised |
| --- | --- | --- |
| batch size | 64 | 128 (paper: 512) |
| learning rate | 3e-5 | 5e-5 |
| epochs | 1 | 3 |
| temperature | 0.05 | 0.05 |
| max length | 32 | 32 |
| dropout | 0.1 | 0.1 |
| eval pooling | CLS before the pooler | CLS with the pooler |
| hard negatives | no | yes, masked when absent |

AdamW, weight decay 0, no warmup, linear decay. Training pools `[CLS]` through BERT's pooler. Unsupervised evaluation drops that MLP and uses the raw `[CLS]` state. Supervised evaluation keeps it.

Ablations, same seed so the data order matches:

```bash
python train.py --mode unsupervised --seed 42 --same_dropout
python train.py --mode supervised --seed 42 --no_hard_negatives
python train.py --mode unsupervised --seed 42 --data_fraction 0.25
```

The same runs can be launched from [colab_runner.ipynb](colab_runner.ipynb). Open it from this folder and run it top to bottom. A run is skipped when `runs/<name>/best` already exists.

## Evaluate and export

Score the test split once, after the checkpoint is chosen. The script refuses to overwrite a stored test number unless you pass `--force`.

```bash
python score_test.py --run-dir runs/<run_name>
```

Plots and nearest-neighbor checks go to `runs/analysis/`:

```bash
python analysis.py --checkpoint unsupervised=runs/<run_name>/best --include-references
```

Seed noise and ablation deltas:

```bash
python summarize.py
```

Export a checkpoint as a sentence-transformers model. The script reloads the saved folder and checks that the STS-B Spearman matches the in-project evaluator. `--push` uploads that folder and repeats the check from the Hub. You need to be logged in (`huggingface-cli login`) before pushing.

```bash
python export_and_push.py \
  --checkpoint runs/<run_name>/best \
  --mode unsupervised \
  --out exported_models/unsupervised
```

## Layout

| path | role |
| --- | --- |
| `simcse/data.py` | SNLI subsets and STS-B |
| `simcse/model.py` | encoder and contrastive loss |
| `simcse/evaluate.py` | Spearman, alignment, uniformity |
| `train.py` | training, dev selection, `runs/log.csv` |
| `score_test.py` | the single test evaluation |
| `analysis.py` | plots and retrievals |
| `summarize.py` | seed noise and ablation deltas |
| `export_and_push.py` | sentence-transformers export and Hub check |
| `data/snli_train_100k.jsonl` | the shared training file |
| `colab_runner.ipynb` | the same protocol on Colab |

Checkpoints in `runs/<run>/best` and exported weights stay on the machine that trained them. A BERT checkpoint is larger than GitHub's file limit. The run log, the ablation summary, and the plots in `runs/analysis/` are in the repo.

## Citation

```bibtex
@inproceedings{gao2021simcse,
  title = {{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
  author = {Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2021}
}
```
