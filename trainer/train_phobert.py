"""Fine-tune PhoBERT-base-v2 on title + lead paragraph (text only).

    python -m trainer.train_phobert --data-dir data/processed --out-dir outputs/phobert
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from architecture.phobert import PHOBERT_NAME, PhoBERTClassifier, load_tokenizer
from trainer.trainer import Trainer
from utils.dataloader import (
    ID2LABEL,
    ROOT,
    class_weights,
    load_processed,
    make_dataloaders,
    segment_words,
)


@dataclass
class Config:
    data_dir: str = str(ROOT / "data" / "processed")
    out_dir: str = str(ROOT / "outputs" / "phobert")
    model_name: str = PHOBERT_NAME
    max_length: int = 128
    use_lead: bool = True
    batch_size: int = 32
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    weighted_loss: bool = True
    patience: int = 3
    monitor: str = "f1"  # validation metric for checkpoint selection; "loss" is minimised
    rdrop_alpha: float = 0.0  # > 0 enables R-Drop (doubles the training time)
    layer_decay: float = 1.0  # < 1 lowers the learning rate layer by layer towards the embeddings
    num_workers: int = 2
    seed: int = 42
    limit: int = 0  # use only the first N rows of each split (smoke tests)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(cfg: Config) -> dict:
    """Train, then evaluate the best checkpoint on the test split.

    Returns the trainer, history, test metrics and a per-article test prediction table.
    """
    # the slow PhoBERT tokenizer logs a truncation notice on every sentence-pair batch
    transformers.logging.set_verbosity_error()
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    splits = load_processed(cfg.data_dir)
    if cfg.limit:
        splits = {k: v.head(cfg.limit) for k, v in splits.items()}
    splits = {k: segment_words(v) for k, v in splits.items()}

    tokenizer = load_tokenizer(cfg.model_name)
    loaders = make_dataloaders(
        splits,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        use_lead=cfg.use_lead,
        use_image=False,
        num_workers=cfg.num_workers,
    )
    model = PhoBERTClassifier(cfg.model_name, dropout=cfg.dropout)
    trainer = Trainer(
        model,
        loaders["train"],
        loaders["val"],
        out_dir=out_dir,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        class_weights=class_weights(splits["train"]) if cfg.weighted_loss else None,
        patience=cfg.patience,
        monitor=cfg.monitor,
        rdrop_alpha=cfg.rdrop_alpha,
        layer_decay=cfg.layer_decay,
    )
    print(f"device: {trainer.device} | train batches: {len(loaders['train'])}")
    history = trainer.fit()

    test_metrics = trainer.evaluate(loaders["test"])
    print("test | " + " | ".join(f"{k} {v:.4f}" for k, v in test_metrics.items()))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    out = trainer.predict(loaders["test"])
    preds = pd.DataFrame({"id": out["id"], "prob_clickbait": out["prob"]})
    preds["pred"] = (preds["prob_clickbait"] >= 0.5).astype(int).map(ID2LABEL)
    preds = preds.merge(load_processed(cfg.data_dir)["test"], on="id")[
        ["id", "source", "category", "title", "label", "pred", "prob_clickbait"]
    ]
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    tokenizer.save_pretrained(out_dir / "tokenizer")

    return {
        "trainer": trainer,
        "history": history,
        "test_metrics": test_metrics,
        "test_predictions": preds,
    }


def parse_config(config_cls, description: str):
    """Build a config from command-line flags, one flag per dataclass field."""
    parser = argparse.ArgumentParser(description=description)
    for name, default in asdict(config_cls()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return config_cls(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_config(Config, __doc__.strip().splitlines()[0]))
