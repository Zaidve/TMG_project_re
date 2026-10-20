# Final model: PhoBERT-base-v2 Vietnamese clickbait classifier

The model chosen for the browser extension, after the cross-validation experiments in this
repository. This folder holds its settings, metrics and test predictions; the weights are
too large for git (see [Weights](#weights)).

## What it does

Scores a Vietnamese news headline, optionally with its lead paragraph, and returns the
probability that it is clickbait. Labels: `clickbait` if the probability is at least 0.5,
otherwise `non-clickbait`.

```python
from utils.predict import ClickbaitPredictor

model = ClickbaitPredictor.from_dir("models/phobert_final")
model.predict("Sốc: 5 bí quyết này ai cũng nên biết!")            # headline only
model.predict(["Tiêu đề ..."], ["Đoạn mở đầu ..."])                # headline + lead
```

`ClickbaitPredictor` applies the same cleaning and `pyvi` word segmentation as training,
so raw text from a web page can be passed in directly.

## Training

| Setting | Value |
|---|---|
| Base model | `vinai/phobert-base-v2` with a linear head on the `<s>` token |
| Input | Title + lead paragraph as a sentence pair, word-segmented with `pyvi`, max 128 tokens |
| Lead dropout | The lead is hidden for 50% of training examples, so the model also judges headlines alone |
| Loss | Class-weighted cross-entropy (about 0.73 non-clickbait, 1.6 clickbait) |
| Optimiser | AdamW, learning rate 2e-5, layer-wise decay 0.9, weight decay 0.01, 10% warm-up |
| Schedule | 5 epochs, batch 32, best checkpoint on validation loss (epoch 5 was kept) |
| Data | `data/processed`: 2,726 train / 341 val / 341 test articles, 31% clickbait, stratified by label and source |
| Seed | 42 |
| Code | `trainer/train_phobert.py` defaults, run by `notebooks/kaggle_final.ipynb` on a Kaggle T4 |

Full settings: [config.json](config.json). Training curve: [history.json](history.json).

## Results

Test split (341 articles, never used for training or checkpoint selection):

| Input | F1 (clickbait) | Precision | Recall | Accuracy | AUC |
|---|---|---|---|---|---|
| Headline + lead | 0.805 | 0.750 | 0.869 | 0.868 | 0.945 |
| Headline only | 0.814 | 0.773 | 0.860 | 0.877 | 0.945 |

For reference, the same configuration without lead dropout scored 0.798 with the lead and
0.751 on headlines alone (precision 0.65).

This is one split of 341 articles. In 5-fold cross-validation the same configuration
averaged an out-of-fold F1 of 0.771, and single models ranged 0.76-0.81 on the test split.
Expect roughly 0.77-0.81 F1 on new articles from the same eight outlets.

## What was tried and did not beat it

All on the same 5 folds, out-of-fold F1 (this configuration: 0.771):

| Change | Out-of-fold F1 |
|---|---|
| Training recipes (R-Drop, fewer epochs, checkpoint on F1) | 0.766-0.774 |
| Loss functions (label smoothing, focal, supervised contrastive, generalised CE) | 0.724-0.768 |
| PhoW2V embeddings, alone or combined | 0.706-0.771 |
| Thumbnail image (CLIP), alone or fused with text | 0.571 / 0.753-0.765 |
| Start of the article body added to the input | 0.758 |
| Body-only model combined with the headline model | 0.767-0.771 |
| Hand-crafted features, alone or stacked on PhoBERT | 0.665 / 0.758 |
| Topic as an input or as a second training task | 0.753-0.770 |
| TF-IDF + logistic regression baseline | 0.659 |

## Limitations

- **Labels limit the score.** 8 of the 41 headlines that appear twice in the dataset carry
  conflicting labels, so a few points of error are likely irreducible.
- **Eight outlets, mostly June 2025.** Accuracy on other sites, other topics or later news
  is untested.
- **Weakest on outlets and topics with little clickbait** (VnExpress, "Tin tức tổng hợp"),
  where there were few clickbait examples to learn from.
- **Needs `pyvi`.** Input must be word-segmented the same way, which ties inference to Python.

## Weights

`best.pt` (537,706,122 bytes) and the tokenizer are kept in `models/phobert_final/`,
which is git-ignored.

SHA-256 of `best.pt`:
`651834768a5eea717f964975894662110b1ac304bdc640bb722edd84c016173e`

To rebuild them, run `notebooks/kaggle_final.ipynb` at the tagged commit; results can
differ slightly from GPU non-determinism.
