"""Live name->SMILES lookup via PubChem PUG-REST -- used only by chat_v3.py
(interactive use), never by oracle.py/train/eval, which need a network-free,
deterministic ground truth (see name_dict.py). Same canonicalization as the
offline dict so scoring stays consistent regardless of which backend resolved
the name."""
from __future__ import annotations
import requests
from rdkit import Chem

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,IsomericSMILES/JSON"
_CACHE: dict[str, str | None] = {}


def resolve_pubchem(name: str, timeout: float = 10.0) -> str | None:
    key = name.strip().lower()
    if key in _CACHE:
        return _CACHE[key]
    try:
        r = requests.get(PUG.format(name=key), timeout=timeout)
        if r.status_code != 200:
            _CACHE[key] = None
            return None
        props = r.json()["PropertyTable"]["Properties"][0]
        raw = props.get("CanonicalSMILES") or props.get("IsomericSMILES") \
            or props.get("ConnectivitySMILES")
        mol = Chem.MolFromSmiles(raw) if raw else None
        smi = Chem.MolToSmiles(mol) if mol else None
    except Exception:
        smi = None
    _CACHE[key] = smi
    return smi


def demo():
    assert resolve_pubchem("acetone") == "CC(C)=O"
    assert resolve_pubchem("not a real chemical name xyz qqq") is None
    print("name_dict_pubchem.py: all checks pass (network required)")


if __name__ == "__main__":
    demo()
