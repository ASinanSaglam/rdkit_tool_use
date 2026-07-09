"""Generate v3 training/eval data: v2's five behaviors (single, multi-property,
tool-restraint, clarify, tool-error) plus a sixth, "chain": a common-name
question that requires name_to_smiles -> rdkit_compute in sequence. One
combined dataset, trained in a single run (avoids catastrophic forgetting
between behaviors, same rationale as v2).

Reuses v1's SMILES sourcing/scaffold-split (rdkit_qa.dataset) and v2's
paraphrase/no-tool/clarify banks -- only the question/trace shape differs."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

from rdkit_qa.dataset import get_smiles, scaffold_split, scaffold, corrupt
from rdkit_qa_v2.dataset import (PARAPHRASES, MULTI_PAIRS, NO_TOOL_TRIVIA,
                                  NO_TOOL_CHEM_ADJACENT, NO_TOOL_META,
                                  CLARIFY_QUESTIONS, _split_bank)
from . import oracle, name_dict

DATA = Path(__file__).resolve().parent.parent / "data"
GOOD_PROPS = [p for p in oracle.PROPERTIES if p != "validity"]

NAME_QUESTION_TEMPLATES = {
    "mw":                "What is the molecular weight of {name}?",
    "logp":              "What is the Crippen logP of {name}?",
    "tpsa":              "What is the TPSA of {name}?",
    "hbd":               "How many hydrogen bond donors does {name} have?",
    "hba":               "How many hydrogen bond acceptors does {name} have?",
    "ring_count":        "How many rings does {name} have?",
    "lipinski":          "Does {name} pass Lipinski's rule of five? Answer true or false.",
    "canonical":         "What is the canonical SMILES of {name}?",
    "qed":               "What is the QED (quantitative estimate of druglikeness) of {name}?",
    "formula":           "What is the molecular formula of {name}?",
    "num_stereocenters": "How many stereocenters does {name} have?",
    "veber":             "Does {name} pass Veber's rule (<=10 rotatable bonds and TPSA <=140)? Answer true or false.",
    "pains_alert":       "Does {name} trigger a PAINS (pan-assay interference) structural alert? Answer true or false.",
}


def _question(prop: str, smiles: str, rng: random.Random) -> str:
    choices = [oracle.QUESTION_TEMPLATES[prop]] + PARAPHRASES.get(prop, [])
    return rng.choice(choices).format(smiles=smiles)


def make_examples(smiles_list, seed=0, split_name="train"):
    rng = random.Random(seed)
    ex = []
    clarify_bank = _split_bank(CLARIFY_QUESTIONS, split_name)
    trivia_bank = _split_bank(NO_TOOL_TRIVIA + NO_TOOL_CHEM_ADJACENT + NO_TOOL_META, split_name)
    chain_names = _split_bank(name_dict.NAMES, split_name)

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
            q = rng.choice(clarify_bank)
            ex.append({"kind": "clarify", "question": q,
                       "clarify_text": "Which molecule? Please provide a SMILES string."})

    # curated trivia/chem-adjacent bank, replicated to stay a visible slice
    # of a large dataset without letting ~10 unique strings dominate it
    reps = max(1, min(5, len(smiles_list) // 500))
    for q, a in trivia_bank * reps:
        ex.append({"kind": "no_tool", "question": q, "answer": a})
    # scale arithmetic no_tool volume with the dataset so restraint stays a
    # meaningful fraction of the mix (~1 per molecule -> roughly 10% of
    # total rows, given ~9 single-kind rows generated per molecule above)
    for _ in range(len(smiles_list)):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        ex.append({"kind": "no_tool", "question": f"What is {a} plus {b}?", "answer": str(a + b)})

    # chain: common-name question -> name_to_smiles then rdkit_compute. Volume
    # is capped by the curated name dict (~50 names/split after partitioning),
    # not by molecule count, so scale via reps like the trivia bank instead of
    # one-per-molecule.
    for name in chain_names:
        for _ in range(reps):
            prop = rng.choice(GOOD_PROPS)
            ex.append({"kind": "chain",
                       "question": NAME_QUESTION_TEMPLATES[prop].format(name=name),
                       "name": name, "prop": prop})

    rng.shuffle(ex)
    return ex


PRESETS = {"dev": dict(source="zinc", n=200), "full": dict(source="zinc", n=10000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dev", help="output dir under data/v3/")
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

    outdir = DATA / "v3" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    for split_name, split in [("train", tr), ("val", va), ("test", te)]:
        rows = make_examples(split, seed=args.seed, split_name=split_name)
        with (outdir / f"{split_name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"  {split_name}: {len(split)} mols -> {len(rows)} QA rows {kinds}")

    s_tr, s_te = {scaffold(s) for s in tr}, {scaffold(s) for s in te}
    assert s_tr.isdisjoint(s_te), "scaffold leakage between train and test!"

    # chain rows: every name must resolve through the offline dict (dataset
    # generation and the oracle must never diverge), and no name should be
    # shared between train/test given the disjoint _split_bank partition.
    all_rows = {}
    for split_name, split in [("train", tr), ("val", va), ("test", te)]:
        with (outdir / f"{split_name}.jsonl").open() as f:
            rows = [json.loads(line) for line in f]
        all_rows[split_name] = rows
        for r in rows:
            if r["kind"] == "chain":
                assert oracle.compute_name(r["name"]) is not None, \
                    f"chain row name unresolved: {r['name']}"
    names_tr = {r["name"] for r in all_rows["train"] if r["kind"] == "chain"}
    names_te = {r["name"] for r in all_rows["test"] if r["kind"] == "chain"}
    assert names_tr.isdisjoint(names_te), "chain name leakage between train and test!"

    print("scaffold split clean (no train/test overlap); chain names clean too")


if __name__ == "__main__":
    main()
