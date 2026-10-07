"""Model-agnostic training loop for the clickbait classifiers.

The model is called as `model(**batch)` with every tensor in the batch except
`labels`, and must return logits of shape (batch, num_labels), or a dict with the
main logits under "logits" (extra entries are passed on to the criterion).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Accuracy, precision/recall/F1 of the clickbait class (id 1) and macro F1."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": tp / (tp + fp) if tp else 0.0,
        "recall": tp / (tp + fn) if tp else 0.0,
        "f1": f1(tp, fp, fn),
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        out_dir: str | Path,
        epochs: int = 10,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        class_weights: torch.Tensor | None = None,
        criterion: nn.Module | None = None,
        grad_clip: float = 1.0,
        patience: int = 3,
        monitor: str = "f1",
        amp: bool = True,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.monitor = monitor
        self.amp = amp and self.device.type == "cuda"

        no_decay = ("bias", "LayerNorm.weight")
        params = list(model.named_parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in params if not any(k in n for k in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in params if any(k in n for k in no_decay)],
                    "weight_decay": 0.0,
                },
            ],
            lr=lr,
        )
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(total_steps * warmup_ratio), total_steps
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        # default: class-weighted cross-entropy; pass e.g. a FusionLoss to override
        self.criterion = (criterion or nn.CrossEntropyLoss(weight=class_weights)).to(self.device)
        self.history: list[dict] = []
        self.best_path = self.out_dir / "best.pt"

    def _forward(self, batch: dict) -> tuple[torch.Tensor | dict, torch.Tensor]:
        inputs = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        labels = inputs.pop("labels")
        with torch.autocast(self.device.type, enabled=self.amp):
            outputs = self.model(**inputs)
        if isinstance(outputs, dict):
            outputs = {k: v.float() for k, v in outputs.items() if v is not None}
        else:
            outputs = outputs.float()
        return outputs, labels

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, seen = 0.0, 0
        bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in bar:
            outputs, labels = self._forward(batch)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total += loss.item() * len(labels)
            seen += len(labels)
            bar.set_postfix(loss=f"{total / seen:.4f}")
        return total / seen

    @torch.no_grad()
    def predict(self, loader) -> dict:
        """Ids, labels, clickbait probabilities and mean loss over a loader."""
        self.model.eval()
        ids, labels_all, probs_all, total = [], [], [], 0.0
        for batch in loader:
            outputs, labels = self._forward(batch)
            total += self.criterion(outputs, labels).item() * len(labels)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            ids.extend(batch["id"])
            labels_all.append(labels.cpu().numpy())
            probs_all.append(logits.softmax(-1)[:, 1].cpu().numpy())
        labels_all, probs_all = np.concatenate(labels_all), np.concatenate(probs_all)
        return {
            "id": ids,
            "label": labels_all,
            "prob": probs_all,
            "loss": total / len(labels_all),
        }

    def evaluate(self, loader, threshold: float = 0.5) -> dict[str, float]:
        out = self.predict(loader)
        metrics = classification_metrics(out["label"], (out["prob"] >= threshold).astype(int))
        return {"loss": out["loss"], **metrics}

    def fit(self) -> list[dict]:
        """Train with early stopping on `monitor`; the best weights are reloaded at the end."""
        best, bad_epochs = -float("inf"), 0
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(epoch)
            val = self.evaluate(self.val_loader)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val.items()},
                "seconds": round(time.time() - start, 1),
            }
            self.history.append(row)
            print(
                f"epoch {epoch:>2} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} "
                f"| acc {val['accuracy']:.3f} | P {val['precision']:.3f} R {val['recall']:.3f} "
                f"F1 {val['f1']:.3f} | macro F1 {val['macro_f1']:.3f} | {row['seconds']}s"
            )
            if val[self.monitor] > best:
                best, bad_epochs = val[self.monitor], 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"early stop: no val {self.monitor} improvement for {self.patience} epochs")
                    break
        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device))
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        print(f"best val {self.monitor}: {best:.4f} -> {self.best_path}")
        return self.history
