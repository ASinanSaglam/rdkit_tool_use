"""Static, offline name->SMILES lookup used by oracle.py (train/eval) and
dataset.py (chain examples). Baked once by build_name_dict.py from PubChem --
no network access at train/eval time, so this is deterministic like every
other oracle ground truth."""
from __future__ import annotations
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "v3" / "name_to_smiles.json"
_DICT: dict[str, str] = json.loads(_PATH.read_text())

NAMES = sorted(_DICT)  # dataset.py samples chain-example names from this


def resolve_dict(name: str) -> str | None:
    """Canonical SMILES for a common name, or None if not in the curated dict.
    Case-insensitive; the dict itself is stored lowercase."""
    return _DICT.get(name.strip().lower())
