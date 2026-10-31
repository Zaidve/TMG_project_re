"""Load a trained clickbait model and score headlines.

    from utils.predict import ClickbaitPredictor

    model = ClickbaitPredictor.from_dir("outputs/phobert_final")
    model.predict("Sốc: 5 bí quyết này ai cũng nên biết!")
    # [{'prob_clickbait': 0.93, 'label': 'clickbait'}]

The directory is what `trainer.train_phobert` writes: `config.json`, `best.pt` and
`tokenizer/`. Input goes through the same cleaning and word segmentation as in training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModel

from architecture.phobert import PhoBERTClassifier, load_tokenizer
from utils.dataloader import ID2LABEL, segment_words, split_location_prefix, strip_publisher_suffix


class ClickbaitPredictor:
    def __init__(self, model: PhoBERTClassifier, tokenizer, config: dict, threshold: float = 0.5):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.threshold = threshold
        self.device = next(model.parameters()).device

    @classmethod
    def from_dir(cls, path: str | Path, device: str | None = None, threshold: float = 0.5):
        transformers.logging.set_verbosity_error()  # base-model load report is expected noise
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = load_tokenizer(str(path / "tokenizer"))
        # Only the architecture is needed from the base model: every weight comes from best.pt.
        # `encoder_config/` (written by `save_encoder_config`) makes loading fully offline;
        # without it the base model's config is fetched from the Hugging Face hub.
        local = path / "encoder_config"
        encoder_config = AutoConfig.from_pretrained(str(local) if local.exists() else config["model_name"])
        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        model = PhoBERTClassifier(encoder=encoder, dropout=config.get("dropout", 0.1))
        model.load_state_dict(torch.load(path / "best.pt", map_location="cpu"))
        return cls(model.to(device), tokenizer, config, threshold)

    @staticmethod
    def clean(title: str, lead: str = "") -> tuple[str, str]:
        """The same light cleaning as `utils.dataloader.clean`, for text scraped from a page."""
        title = strip_publisher_suffix(" ".join(title.split()))
        lead = split_location_prefix(" ".join(lead.split())) if lead else ""
        return title, lead

    @torch.no_grad()
    def predict(self, titles: str | list[str], leads: str | list[str] | None = None, batch_size: int = 32) -> list[dict]:
        titles = [titles] if isinstance(titles, str) else list(titles)
        if leads is None:
            leads = [""] * len(titles)
        leads = [leads] if isinstance(leads, str) else list(leads)
        rows = pd.DataFrame([self.clean(t, l) for t, l in zip(titles, leads)], columns=["title", "lead_paragraph"])
        if self.config.get("segment", True):
            rows = segment_words(rows)
        use_lead = self.config.get("use_lead", True)
        results = []
        for start in range(0, len(rows), batch_size):
            part = rows.iloc[start : start + batch_size]
            leads_part = list(part["lead_paragraph"]) if use_lead else None
            batch = self.tokenizer(
                list(part["title"]), leads_part, padding=True, truncation=True,
                max_length=self.config.get("max_length", 128), return_tensors="pt",
            )
            logits = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            for p in logits.float().softmax(-1)[:, 1].tolist():
                results.append({"prob_clickbait": round(p, 4), "label": ID2LABEL[int(p >= self.threshold)]})
        return results


def save_encoder_config(model_dir: str | Path) -> None:
    """Store the base model's architecture config next to the weights, for offline loading."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    AutoConfig.from_pretrained(config["model_name"]).save_pretrained(model_dir / "encoder_config")
