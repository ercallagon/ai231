"""Verified canonical label/loss helpers extracted without training imports."""
from typing import Optional
import numpy as np


SLOT_VALUE_MAP = {
    "TIMER": {"10 seconds": "10s", "30 seconds": "30s", "1 minute": "1m"},
    "ALARM": {"6:00 AM": "6AM", "8:00 AM": "8AM", "9:00 PM": "9PM"},
    "TEMPERATURE": {"18 degrees": "18", "22 degrees": "22", "26 degrees": "26"},
    "BRIGHTNESS": {"20 percent": "20", "60 percent": "60", "100 percent": "100"},
    "COLOR": {"red": "red", "blue": "blue", "green": "green"},
    "CREATE_REMINDER": {"drink water": "drink_water", "study": "study", "exercise": "exercise"},
}


def _class_weights(labels: list, n_classes: int, device) -> "object":
    """Inverse-frequency class weights (helps rare intents + OOS)."""
    import torch
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _slot_label(intent: str, raw_slot: Optional[str], tax) -> Optional[int]:
    """Map a raw slot_value string to a slot-class index, or None.

    None means: no slot supervision for this row (non-slotted intent, empty
    slot, or an unsupported value — the latter only occurs with slot_reject
    off, where unsupported values are dropped rather than REJECT-labeled).
    """
    if intent not in tax.slotted_intents or raw_slot is None:
        return None
    classes = tax.slot_classes(intent)
    canon = SLOT_VALUE_MAP.get(intent, {}).get(str(raw_slot).strip().lower()) \
        or SLOT_VALUE_MAP.get(intent, {}).get(str(raw_slot).strip())
    if canon is None:
        # try case-insensitive direct membership
        low = {c.lower(): c for c in classes}
        canon = low.get(str(raw_slot).strip().lower())
    if canon is None or canon not in classes:
        return None
    return classes.index(canon)


def _slot_loss_and_stats(out, yb, ys, tax, loss_fn) -> tuple:
    """Compute slot loss/accuracy on rows whose ground-truth intent is slotted."""
    import torch
    slots = out.get("slots", {}) if isinstance(out, dict) else {}
    losses = []
    slot_n = 0
    slot_correct = 0
    for intent_name, logits in slots.items():
        if intent_name not in tax.intent_to_idx:
            continue
        mask = (yb == tax.intent_to_idx[intent_name]) & (ys >= 0)
        if not torch.any(mask):
            continue
        losses.append(loss_fn(logits[mask], ys[mask]))
        slot_correct += (logits[mask].argmax(1) == ys[mask]).sum().item()
        slot_n += int(mask.sum().item())
    if losses:
        return torch.stack(losses).mean(), slot_correct, slot_n
    return None, slot_correct, slot_n
