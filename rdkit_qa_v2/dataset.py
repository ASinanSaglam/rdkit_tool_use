"""Generate v2 training/eval data: single-property, multi-property,
tool-restraint (no-tool-needed), clarify (missing SMILES), and tool-error
(invalid molecule) examples -- five behaviors, one combined dataset, trained
in a single run (not sequential fine-tunes -- avoids catastrophic
forgetting between behaviors).

Reuses v1's SMILES sourcing/scaffold-split (rdkit_qa.dataset) rather than
duplicating it -- only the question/trace shape differs between v1 and v2,
not how molecules are sourced or split."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

from rdkit_qa.dataset import get_smiles, scaffold_split, scaffold, corrupt
from . import oracle

DATA = Path(__file__).resolve().parent.parent / "data"
GOOD_PROPS = [p for p in oracle.PROPERTIES if p != "validity"]

# Extra phrasings per property, folded into the *same* rows as the canonical
# template (picked randomly per row) rather than kept as a separate
# category -- teaches paraphrase robustness without inflating dataset size.
PARAPHRASES = {
    "mw":         ["How heavy is one mole of {smiles}, in grams?"],
    "logp":       ["Roughly how lipophilic (Crippen logP) is {smiles}?"],
    "tpsa":       ["Compute the polar surface area for {smiles}, please."],
    "hbd":        ["For the compound {smiles}, count its hydrogen bond donor groups."],
    "hba":        ["For the compound {smiles}, count its hydrogen bond acceptor groups."],
    "ring_count": ["{smiles} -- how many ring systems does it contain?"],
    "lipinski":   ["Would {smiles} be considered orally druglike by Lipinski's rule of five (true/false)?"],
    "validity":   ["Can you parse {smiles} as a real molecule (true/false)?"],
    "canonical":  ["Rewrite {smiles} in RDKit canonical form."],
}

MULTI_PAIRS = [("mw", "logp"), ("hbd", "hba"), ("ring_count", "validity"),
               ("mw", "tpsa"), ("lipinski", "logp"), ("tpsa", "hba")]

# Arithmetic template + a curated bank: general trivia (no chemistry content
# at all) and chemistry-*adjacent* questions that still don't need RDKit --
# the harder, more valuable negative, since surface-level topic overlap is
# exactly what could make a model reach for the tool reflexively.
NO_TOOL_TRIVIA = [
    ("Spell the two-letter chemical symbol for gold.", "au"),
    ("How many legs does a spider have?", "8"),
    ("What color is chlorophyll?", "green"),
    ("Name the capital of France.", "paris"),
    ("What is the boiling point of water in Celsius, at sea level?", "100"),
]
NO_TOOL_CHEM_ADJACENT = [
    ("What does the acronym SMILES stand for in chemistry?",
     "simplified molecular input line entry system"),
    ("What family of compounds do alkanes belong to?", "hydrocarbons"),
    ("Is water polar or nonpolar?", "polar"),
    ("What is the atomic number of carbon?", "6"),
    ("What does logP measure, conceptually?", "lipophilicity"),
    ("Name the rule that estimates oral druglikeness from four simple thresholds.",
     "lipinski"),
]

CLARIFY_QUESTIONS = [
    "What is its molecular weight?",
    "How many hydrogen bond donors does it have?",
    "Is it druglike by Lipinski's rule?",
    "What's the logP?",
    "Can you give me the canonical SMILES?",
]


def _question(prop: str, smiles: str, rng: random.Random) -> str:
    choices = [oracle.QUESTION_TEMPLATES[prop]] + PARAPHRASES.get(prop, [])
    return rng.choice(choices).format(smiles=smiles)


def make_examples(smiles_list, seed=0):
    rng = random.Random(seed)
    ex = []

    for s in smiles_list:
        for prop in GOOD_PROPS + ["validity"]:
            ex.append({"kind": "single", "question": _question(prop, s, rng),
                       "smiles": s, "prop": prop})
        if rng.random() < 0.15:  # tool-error: corrupted SMILES on a non-validity property
            prop = rng.choice(GOOD_PROPS)
            bad = corrupt(s, rng)
            ex.append({"kind": "error", "question": _question(prop, bad, rng),
                       "smiles": bad, "prop": prop})
        if rng.random() < 0.4:  # multi-property composition
            p1, p2 = rng.choice(MULTI_PAIRS)
            q = (f"For the molecule {s}, what is its "
                 f"{p1.replace('_', ' ')} and its {p2.replace('_', ' ')}?")
            ex.append({"kind": "multi", "question": q, "pairs": [[s, p1], [s, p2]]})
        if rng.random() < 0.15:  # clarify: needs a molecule, none given
            q = rng.choice(CLARIFY_QUESTIONS)
            ex.append({"kind": "clarify", "question": q,
                       "clarify_text": "Which molecule? Please provide a SMILES string."})

    for q, a in NO_TOOL_TRIVIA + NO_TOOL_CHEM_ADJACENT:
        ex.append({"kind": "no_tool", "question": q, "answer": a})
    # scale arithmetic no_tool volume with the dataset so restraint stays a
    # meaningful fraction of the mix (~1 per molecule -> roughly 10% of
    # total rows, given ~9 single-kind rows generated per molecule above)
    for _ in range(len(smiles_list)):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        ex.append({"kind": "no_tool", "question": f"What is {a} plus {b}?", "answer": str(a + b)})

    rng.shuffle(ex)
    return ex


PRESETS = {"dev": dict(source="zinc", n=200), "full": dict(source="zinc", n=10000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dev", help="output dir under data/v2/")
    ap.add_argument("--source", default=None, choices=["zinc", "bbbp"])
    ap.add_argument("--n-mols", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    preset = PRESETS.get(args.name, PRESETS["dev"])
    source = args.source or preset["source"]
    n = args.n_mols or preset["n"]

    sm = get_smiles(source, n, seed=args.seed)
    print(f"collected {len(sm)} unique mols from {source}")
    tr, va, te = scaffold_split(sm, seed=args.seed)

    outdir = DATA / "v2" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    for split_name, split in [("train", tr), ("val", va), ("test", te)]:
        rows = make_examples(split, seed=args.seed)
        with (outdir / f"{split_name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"  {split_name}: {len(split)} mols -> {len(rows)} QA rows {kinds}")

    s_tr, s_te = {scaffold(s) for s in tr}, {scaffold(s) for s in te}
    assert s_tr.isdisjoint(s_te), "scaffold leakage between train and test!"
    print("scaffold split clean (no train/test overlap)")


if __name__ == "__main__":
    main()
