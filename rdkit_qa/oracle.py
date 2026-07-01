"""RDKit ground-truth oracle. Single source of truth for both label generation
and verifiable scoring. Train labels and eval scoring MUST import from here so
they can never drift."""
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence parse warnings; validity is a label here

# property -> (value type, tolerance). tolerance None => exact match.
PROPERTIES = {
    "mw":         ("float", 0.1),
    "logp":       ("float", 0.1),
    "tpsa":       ("float", 0.1),
    "hbd":        ("int",   None),
    "hba":        ("int",   None),
    "ring_count": ("int",   None),
    "lipinski":   ("bool",  None),
    "validity":   ("bool",  None),
    "canonical":  ("str",   None),
}

QUESTION_TEMPLATES = {
    "mw":         "What is the molecular weight of the molecule with SMILES {smiles}?",
    "logp":       "What is the Crippen logP of the molecule with SMILES {smiles}?",
    "tpsa":       "What is the TPSA of the molecule with SMILES {smiles}?",
    "hbd":        "How many hydrogen bond donors does {smiles} have?",
    "hba":        "How many hydrogen bond acceptors does {smiles} have?",
    "ring_count": "How many rings does {smiles} have?",
    "lipinski":   "Does {smiles} pass Lipinski's rule of five? Answer true or false.",
    "validity":   "Is {smiles} a valid SMILES string? Answer true or false.",
    "canonical":  "What is the canonical SMILES of {smiles}?",
}


def compute(smiles: str, prop: str):
    """Ground-truth value for (smiles, prop). Returns None if uncomputable
    (invalid SMILES) — except for `validity`, which is defined on bad input."""
    if prop not in PROPERTIES:
        raise ValueError(f"unknown property: {prop}")
    mol = Chem.MolFromSmiles(smiles)
    if prop == "validity":
        return mol is not None
    if mol is None:
        return None
    if prop == "mw":         return round(Descriptors.MolWt(mol), 4)
    if prop == "logp":       return round(Crippen.MolLogP(mol), 4)
    if prop == "tpsa":       return round(rdMolDescriptors.CalcTPSA(mol), 4)
    if prop == "hbd":        return Lipinski.NumHDonors(mol)
    if prop == "hba":        return Lipinski.NumHAcceptors(mol)
    if prop == "ring_count": return rdMolDescriptors.CalcNumRings(mol)
    if prop == "lipinski":   return _lipinski_pass(mol)
    if prop == "canonical":  return Chem.MolToSmiles(mol)


def _lipinski_pass(mol) -> bool:
    """Ro5: <=1 violation among MW<=500, logP<=5, HBD<=5, HBA<=10."""
    viol = (Descriptors.MolWt(mol) > 500) + (Crippen.MolLogP(mol) > 5) \
         + (Lipinski.NumHDonors(mol) > 5) + (Lipinski.NumHAcceptors(mol) > 10)
    return viol <= 1


def score(prop: str, predicted, smiles: str) -> bool:
    """Verifiable reward: does `predicted` match ground truth for (smiles, prop)?
    `predicted` is the parsed model answer (already coerced to right type by caller)."""
    vtype, tol = PROPERTIES[prop]
    truth = compute(smiles, prop)
    if truth is None:
        return False
    try:
        if vtype == "float":
            return abs(float(predicted) - truth) <= tol
        if vtype == "int":
            return int(predicted) == truth
        if vtype == "bool":
            return _as_bool(predicted) == truth
        if vtype == "str":
            # canonicalize prediction too: equal molecules, any input form.
            pm = Chem.MolFromSmiles(str(predicted))
            return pm is not None and Chem.MolToSmiles(pm) == truth
    except (ValueError, TypeError):
        return False
    return False


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "yes", "1", "pass", "passes"):
        return True
    if s in ("false", "no", "0", "fail", "fails"):
        return False
    raise ValueError(f"not a bool: {x}")


def demo():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert compute(aspirin, "validity") is True
    assert compute("not_a_smiles", "validity") is False
    assert compute("not_a_smiles", "mw") is None
    assert abs(compute(aspirin, "mw") - 180.16) < 0.1
    assert compute(aspirin, "hbd") == 1
    assert compute(aspirin, "ring_count") == 1
    assert compute(aspirin, "lipinski") is True
    # scoring: tolerance + type coercion + canonical equivalence
    assert score("mw", "180.16", aspirin) is True
    assert score("mw", "181.0", aspirin) is False
    assert score("hbd", 1, aspirin) is True
    assert score("lipinski", "true", aspirin) is True
    assert score("validity", "false", "not_a_smiles") is True
    assert score("canonical", "OC(=O)c1ccccc1OC(C)=O", aspirin) is True  # reordered
    assert score("canonical", aspirin, aspirin) is True
    print("oracle.py: all checks pass")


if __name__ == "__main__":
    demo()
