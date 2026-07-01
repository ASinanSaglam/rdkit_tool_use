"""Generate verifiable molecular-property QA from a public SMILES set.

Sources:
  zinc  - sagawa/ZINC-canonicalized (HF, streamed; drug-like by construction)
  bbbp  - DeepChem BBBP csv (~2k mols, small/fast)

Scaffold split (Bemis-Murcko) => no train/test leakage.
Labels come from oracle.compute, so they are exact by construction.

Two presets via --name: build a tiny `dev` set for fast local smoke tests and a
big `full` set for training, side by side under data/<name>/."""
from __future__ import annotations
import argparse, csv, json, random, urllib.request
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from . import oracle

RDLogger.DisableLog("rdApp.*")
BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
DATA = Path(__file__).resolve().parent.parent / "data"

GOOD_PROPS = [p for p in oracle.PROPERTIES if p != "validity"]


def is_drug_like(mol) -> bool:
    """Lipinski Ro5 + MW window. Off by default; used for filtered benchmark sets."""
    return (150 <= Descriptors.MolWt(mol) <= 500 and Crippen.MolLogP(mol) <= 5
            and Lipinski.NumHDonors(mol) <= 5 and Lipinski.NumHAcceptors(mol) <= 10)


def _accept(mol, drug_like: bool) -> bool:
    return mol is not None and 3 <= mol.GetNumAtoms() <= 60 and \
        (not drug_like or is_drug_like(mol))


def get_smiles(source: str, n: int, drug_like=False, seed=0) -> list[str]:
    """Canonical, deduped, valid SMILES. Stops once `n` collected (n=0 => all)."""
    seen, out = set(), []

    def add(raw_smiles) -> bool:
        mol = Chem.MolFromSmiles(raw_smiles)
        if not _accept(mol, drug_like):
            return False
        cano = Chem.MolToSmiles(mol)
        if cano in seen:
            return False
        seen.add(cano)
        out.append(cano)
        return len(out) >= n > 0  # True => enough, stop

    if source == "bbbp":
        raw = DATA / "raw_bbbp.csv"
        if not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(urllib.request.urlopen(BBBP_URL, timeout=60).read())
        for row in csv.DictReader(raw.open()):
            if add(row["smiles"]):
                break
    elif source == "zinc":
        from datasets import load_dataset
        ds = load_dataset("sagawa/ZINC-canonicalized", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=20000)  # diversity, not just first shard
        for row in ds:
            if add(row["smiles"]):
                break
    else:
        raise ValueError(f"unknown source: {source}")
    return out


def scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles)
    except Exception:
        return ""


def scaffold_split(smiles, fracs=(0.8, 0.1, 0.1), seed=0):
    """Split by scaffold group so no scaffold spans two sets."""
    groups = defaultdict(list)
    for s in smiles:
        groups[scaffold(s)].append(s)
    buckets = list(groups.values())
    random.Random(seed).shuffle(buckets)
    n = len(smiles)
    train, val, test = [], [], []
    for b in buckets:
        if len(train) < fracs[0] * n:
            train += b
        elif len(train) + len(val) < (fracs[0] + fracs[1]) * n:
            val += b
        else:
            test += b
    return train, val, test


def corrupt(smiles: str, rng: random.Random) -> str:
    """Make a syntactically-broken SMILES for validity negatives."""
    i = rng.randrange(len(smiles))
    return smiles[:i] + rng.choice("()[]@123") + smiles[i + 1:]


def make_examples(smiles_list, seed=0):
    """One QA example per (smiles, property). Plus corrupted-SMILES validity negatives."""
    rng = random.Random(seed)
    ex = []
    for s in smiles_list:
        for prop in GOOD_PROPS + ["validity"]:
            ex.append(_row(s, prop, oracle.compute(s, prop)))
        if rng.random() < 0.25:
            bad = corrupt(s, rng)
            ex.append(_row(bad, "validity", oracle.compute(bad, "validity")))
    rng.shuffle(ex)
    return ex


def _row(smiles, prop, answer):
    return {"smiles": smiles, "property": prop,
            "question": oracle.QUESTION_TEMPLATES[prop].format(smiles=smiles),
            "answer": answer}


PRESETS = {"dev": dict(source="zinc", n=200), "full": dict(source="zinc", n=10000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dev", help="output dir under data/ (also picks preset)")
    ap.add_argument("--source", default=None, choices=["zinc", "bbbp"])
    ap.add_argument("--n-mols", type=int, default=0, help="override preset target count")
    ap.add_argument("--drug-like", action="store_true", help="apply Lipinski/MW filter")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    preset = PRESETS.get(args.name, PRESETS["dev"])
    source = args.source or preset["source"]
    n = args.n_mols or preset["n"]

    sm = get_smiles(source, n, drug_like=args.drug_like, seed=args.seed)
    print(f"collected {len(sm)} unique mols from {source} (drug_like={args.drug_like})")
    tr, va, te = scaffold_split(sm, seed=args.seed)

    outdir = DATA / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    for split_name, split in [("train", tr), ("val", va), ("test", te)]:
        rows = make_examples(split, seed=args.seed)
        with (outdir / f"{split_name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {split_name}: {len(split)} mols -> {len(rows)} QA")

    s_tr, s_te = {scaffold(s) for s in tr}, {scaffold(s) for s in te}
    assert s_tr.isdisjoint(s_te), "scaffold leakage between train and test!"
    print("scaffold split clean (no train/test overlap)")


if __name__ == "__main__":
    main()
