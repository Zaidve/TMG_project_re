"""Loss for the text + image fusion model.

The fusion model may return either the fused logits alone, or a dict

    {"logits": fused, "text_logits": ..., "image_logits": ...}

where the two extra entries come from optional auxiliary heads on each branch.
The auxiliary terms keep a branch learning on its own, so the weaker modality
(here usually the image) is not simply ignored by the fusion head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FusionLoss(nn.Module):
    """Class-weighted cross-entropy on the fused logits plus weighted auxiliary terms.

        loss = CE(fused) + text_weight * CE(text_only) + image_weight * CE(image_only)

    An auxiliary term is skipped when its weight is 0 or the model does not return it.
    `gamma > 0` turns every term into focal loss, which down-weights easy examples.
    """

    AUX_KEYS = {"text": "text_logits", "image": "image_logits"}

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        text_weight: float = 0.3,
        image_weight: float = 0.3,
        label_smoothing: float = 0.0,
        gamma: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.aux_weights = {"text": text_weight, "image": image_weight}
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.last_terms: dict[str, float] = {}

    def _term(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.gamma == 0:
            return F.cross_entropy(
                logits, labels, weight=self.class_weights, label_smoothing=self.label_smoothing
            )
        ce = F.cross_entropy(
            logits, labels, weight=self.class_weights,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        p_true = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
        return ((1 - p_true) ** self.gamma * ce).mean()

    def forward(self, outputs: torch.Tensor | dict, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            outputs = {"logits": outputs}
        loss = self._term(outputs["logits"], labels)
        self.last_terms = {"fused": loss.item()}
        for name, key in self.AUX_KEYS.items():
            weight = self.aux_weights[name]
            if weight and outputs.get(key) is not None:
                term = self._term(outputs[key], labels)
                self.last_terms[name] = term.item()
                loss = loss + weight * term
        return loss
