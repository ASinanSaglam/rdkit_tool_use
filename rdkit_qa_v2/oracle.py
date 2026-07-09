"""v2 oracle: everything v1 has (mw, logp, tpsa, hbd, hba, ring_count,
lipinski, validity, canonical) plus five more medchem-flavored properties
(qed, formula, num_stereocenters, veber, pains_alert). v1's oracle.py is
frozen (results already shipped to CV) -- this is a superset, not an edit,
so v1's numbers never move under it."""
from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from rdkit_qa.oracle import PROPERTIES as _V1_PROPERTIES
from rdkit_qa.oracle import QUESTION_TEMPLATES as _V1_TEMPLATES
from rdkit_qa.oracle import compute as _v1_compute
from rdkit_qa.oracle import _as_bool

_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_pains_params)

# property -> (value type, tolerance). "str" = canonical-SMILES equivalence
# (v1's "canonical"); "str_exact" = plain stripped/uppercased string match
# (formula isn't a SMILES, so canonicalizing it as one would be wrong).
PROPERTIES = {
    **_V1_PROPERTIES,
    "qed":               ("float", 0.02),
    "formula":           ("str_exact", None),
    "num_stereocenters": ("int",   None),
    "veber":             ("bool",  None),
    "pains_alert":       ("bool",  None),
}

QUESTION_TEMPLATES = {
    **_V1_TEMPLATES,
    "qed":               "What is the QED (quantitative estimate of druglikeness) of {smiles}?",
    "formula":           "What is the molecular formula of {smiles}?",
    "num_stereocenters": "How many stereocenters does {smiles} have?",
    "veber":             "Does {smiles} pass Veber's rule (<=10 rotatable bonds and TPSA <=140)? Answer true or false.",
    "pains_alert":       "Does {smiles} trigger a PAINS (pan-assay interference) structural alert? Answer true or false.",
}

_NEW_PROPS = {"qed", "formula", "num_stereocenters", "veber", "pains_alert"}


def compute(smiles: str, prop: str):
    if prop not in PROPERTIES:
        raise ValueError(f"unknown property: {prop}")
    if prop not in _NEW_PROPS:
        return _v1_compute(smiles, prop)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if prop == "qed":               return round(QED.qed(mol), 4)
    if prop == "formula":           return rdMolDescriptors.CalcMolFormula(mol)
    if prop == "num_stereocenters":
        return len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False))
    if prop == "veber":
        return (Descriptors.NumRotatableBonds(mol) <= 10
                and rdMolDescriptors.CalcTPSA(mol) <= 140)
    if prop == "pains_alert":       return _PAINS_CATALOG.HasMatch(mol)


def score(prop: str, predicted, smiles: str) -> bool:
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
        if vtype == "str_exact":
            return str(predicted).strip().upper() == truth.strip().upper()
        if vtype == "str":
            pm = Chem.MolFromSmiles(str(predicted))
            return pm is not None and Chem.MolToSmiles(pm) == truth
    except (ValueError, TypeError):
        return False
    return False


def demo():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert compute(aspirin, "formula") == "C9H8O4"
    assert 0.0 <= compute(aspirin, "qed") <= 1.0
    assert compute(aspirin, "veber") is True
    assert isinstance(compute(aspirin, "pains_alert"), bool)
    assert compute("not_a_smiles", "qed") is None
    assert score("formula", "c9h8o4", aspirin) is True  # case-insensitive
    assert score("formula", "C9H9O4", aspirin) is False
    # existing v1 properties still resolve through the same interface
    assert abs(compute(aspirin, "mw") - 180.16) < 0.1
    assert score("mw", "180.16", aspirin) is True
    print("oracle.py (v2): all checks pass")


if __name__ == "__main__":
    demo()
