"""BC8 with a lightweight supported-command head; historical factory unchanged."""
import torch
from torch import nn
from .broadcast import build_broadcast_model

NAME = "bc8_command_validity"


class ValidityModel(nn.Module):
    def __init__(self, n_way, slot_heads):
        super().__init__()
        self.backbone = build_broadcast_model(n_way, slot_heads)
        self.validity_head = nn.Linear(256, 1)

    def forward(self, x):
        embedding = self.backbone.embed(x)
        validity = self.validity_head(embedding).squeeze(1)
        self.validity_logits = validity
        return {"intent": self.backbone.intent_head(embedding),
                "slots": {k: head(embedding) for k, head in self.backbone.slot_head_map.items()},
                "validity": validity}


def balanced_validity_loss(logits, labels, oos_idx):
    target = (labels != oos_idx).float()
    values = nn.functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    means = [values[target == value].mean() for value in (0, 1) if (target == value).any()]
    return torch.stack(means).mean()
