"""v3 oracle: everything v2 has (mw/logp/tpsa/hbd/hba/ring_count/lipinski/
validity/canonical/qed/formula/num_stereocenters/veber/pains_alert) plus
name->SMILES resolution for the name_to_smiles tool. v2's oracle.py is frozen
(kept as a shippable baseline) -- this is a superset, not an edit.

name_to_smiles has a different shape than the other properties (input is a
common name, not a SMILES string) so it gets its own compute_name/score_name
pair rather than being forced into compute(smiles, prop)/score(prop, ...)."""
from __future__ import annotations
from rdkit import Chem

from rdkit_qa_v2.oracle import PROPERTIES, QUESTION_TEMPLATES, compute, score
from . import name_dict

__all__ = ["PROPERTIES", "QUESTION_TEMPLATES", "compute", "score",
           "compute_name", "score_name"]


def compute_name(name: str) -> str | None:
    """Ground-truth canonical SMILES for a common name, or None if unresolved."""
    return name_dict.resolve_dict(name)


def score_name(predicted, name: str) -> bool:
    """Verifiable reward for a name_to_smiles call: does `predicted` denote the
    same molecule as the ground-truth SMILES for `name`? Canonical-SMILES
    equivalence, same comparison v1/v2 use for the `canonical` property."""
    truth = compute_name(name)
    if truth is None:
        return False
    pm = Chem.MolFromSmiles(str(predicted))
    return pm is not None and Chem.MolToSmiles(pm) == truth


def demo():
    assert compute_name("acetone") == "CC(C)=O"
    assert compute_name("not a real chemical name xyz") is None
    assert score_name("CC(C)=O", "acetone") is True
    assert score_name("C(C)(C)=O", "acetone") is True  # different SMILES, same molecule
    assert score_name("CCO", "acetone") is False  # wrong molecule
    assert score_name("not_smiles", "acetone") is False
    assert score_name("CC(C)=O", "not a real chemical name xyz") is False
    # v2 properties still resolve through the same interface
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert compute(aspirin, "formula") == "C9H8O4"
    assert score("mw", "180.16", aspirin) is True
    print("oracle.py (v3): all checks pass")


if __name__ == "__main__":
    demo()
