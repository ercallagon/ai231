"""VCM taxonomy — the single source of truth for intents, OOS, and slots.

Carried forward from the locked me2-vcm2 schema (see docs/decisions.md).
This module reads `configs/vcm.yaml` and exposes the *effective* label sets,
deriving the active slot classes from `slot_reject`:

    slot_reject = False  -> 3-way slot heads (supported values only)
    slot_reject = True   -> 4-way slot heads (3 supported + REJECT)

Nothing here depends on torch or audio — it is pure data, so it can be imported
anywhere (tests, manifests, deploy) without heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

# --- Canonical 19 intents (0..18) + OUT_OF_SCOPE (19) ------------------------
# Order is FIXED and is the model's class order. Do not reorder without a new
# experiment tag (see docs/decisions.md, "Schema is frozen per tag").
INTENTS: List[str] = [
    "PLAY_MUSIC",        # 0
    "WEATHER",           # 1
    "TIME",              # 2
    "LIGHT_ON",          # 3
    "LIGHT_OFF",         # 4
    "PAUSE",             # 5
    "LIST_REMINDERS",    # 6
    "STOP",              # 7
    "NEXT",              # 8
    "VOLUME_UP",         # 9
    "VOLUME_DOWN",       # 10
    "CALL",              # 11
    "MESSAGE",           # 12
    "TIMER",             # 13
    "ALARM",             # 14
    "TEMPERATURE",       # 15
    "BRIGHTNESS",        # 16
    "COLOR",             # 17
    "CREATE_REMINDER",   # 18
]
OOS_CLASS = "OUT_OF_SCOPE"  # 19
N_WAY_HEAD = len(INTENTS) + 1  # 20

# Slotted intents and their supported (non-REJECT) values.
_DEFAULT_SLOT_SETS: Dict[str, List[str]] = {
    "TIMER": ["10s", "30s", "1m"],
    "ALARM": ["6AM", "8AM", "9PM"],
    "TEMPERATURE": ["18", "22", "26"],
    "BRIGHTNESS": ["20", "60", "100"],
    "COLOR": ["red", "blue", "green"],
    "CREATE_REMINDER": ["drink_water", "study", "exercise"],
}
REJECT = "REJECT"


@dataclass(frozen=True)
class Taxonomy:
    """Resolved, immutable view of the VCM label space."""
    intents: Sequence[str]
    oos_class: str
    n_way_head: int
    slot_reject: bool
    slotted_intents: Sequence[str]
    slot_sets: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def intent_to_idx(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.intents)}

    @property
    def idx_to_intent(self) -> Dict[int, str]:
        return {i: name for i, name in enumerate(self.intents)}

    @property
    def oos_idx(self) -> int:
        return self.n_way_head - 1

    def slot_classes(self, intent: str) -> List[str]:
        """Effective slot classes for an intent (honors slot_reject)."""
        if intent not in self.slot_sets:
            return []
        vals = list(self.slot_sets[intent])
        if not self.slot_reject and REJECT in vals:
            vals = [v for v in vals if v != REJECT]
        return vals

    def slot_width(self, intent: str) -> int:
        return len(self.slot_classes(intent))

    def is_slotted(self, intent: str) -> bool:
        return intent in self.slot_sets


def load_taxonomy(config_path: str | Path = "configs/vcm.yaml") -> Taxonomy:
    """Build a Taxonomy from a config YAML.

    The config's `slot_sets` may include a trailing REJECT per intent; this
    function stores them verbatim and lets `slot_classes()` strip REJECT when
    `slot_reject` is False.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    schema = cfg.get("schema", {}) or {}
    intents = list(schema.get("intents", INTENTS))
    oos = schema.get("oos_class", OOS_CLASS)
    n_way = int(schema.get("n_way_head", len(intents) + 1))
    slot_reject = bool(schema.get("slot_reject", False))
    slotted = list(schema.get("slotted_intents", list(_DEFAULT_SLOT_SETS)))
    slot_sets = {k: list(v) for k, v in (schema.get("slot_sets") or _DEFAULT_SLOT_SETS).items()}

    return Taxonomy(
        intents=intents,
        oos_class=oos,
        n_way_head=n_way,
        slot_reject=slot_reject,
        slotted_intents=slotted,
        slot_sets=slot_sets,
    )


def default_taxonomy() -> Taxonomy:
    """A Taxonomy from the built-in defaults (no config file needed)."""
    return Taxonomy(
        intents=INTENTS,
        oos_class=OOS_CLASS,
        n_way_head=N_WAY_HEAD,
        slot_reject=False,
        slotted_intents=list(_DEFAULT_SLOT_SETS),
        slot_sets={k: list(v) for k, v in _DEFAULT_SLOT_SETS.items()},
    )
