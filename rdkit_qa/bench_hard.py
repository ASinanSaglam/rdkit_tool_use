"""Harder, out-of-training-distribution benchmark: paraphrased questions,
tool-restraint (no-tool-needed) questions, and multi-property composition in
one episode. Deliberately separate from data/, dataset.py, and train.py --
this only evaluates generalization of an already-trained base/ft model, and
never touches the main pipeline's data or artifacts.

Uses harness.run_episode_general (loop-until-FINAL, N tool calls) instead of
the fixed one-call harness.run_episode that the main pipeline trains/evals
against -- the model was never trained on this protocol variation, so this
measures zero-shot generalization, not memorized behavior."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from . import harness
from . import eval as ev

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"

# Reworded versions of oracle.QUESTION_TEMPLATES -- same property, different
# phrasing, to test whether the model generalized the protocol or memorized
# the exact training question templates.
PARAPHRASES = {
    "mw":         "How heavy is one mole of {smiles}, in grams?",
    "logp":       "Roughly how lipophilic (Crippen logP) is {smiles}?",
    "tpsa":       "Compute the polar surface area for {smiles}, please.",
    "hbd":        "For the compound {smiles}, count its hydrogen bond donor groups.",
    "hba":        "For the compound {smiles}, count its hydrogen bond acceptor groups.",
    "ring_count": "{smiles} -- how many ring systems does it contain?",
    "lipinski":   "Would {smiles} be considered orally druglike by Lipinski's rule of five (true/false)?",
    "validity":   "Can you parse {smiles} as a real molecule (true/false)?",
    "canonical":  "Rewrite {smiles} in RDKit canonical form.",
}

# Trivia/arithmetic questions with no molecule or RDKit content at all --
# correct behavior is answering directly, never emitting a TOOL_CALL.
NO_TOOL = [
    {"question": "What is 12 plus 7?", "answer": "19"},
    {"question": "Spell the two-letter chemical symbol for gold.", "answer": "au"},
    {"question": "How many legs does a spider have?", "answer": "8"},
    {"question": "What color is chlorophyll?", "answer": "green"},
    {"question": "Name the capital of France.", "answer": "paris"},
]

# Property pairs asked about the same molecule in one question, to test
# whether the model chains multiple tool calls instead of answering only
# the first property (the trained protocol only ever demonstrates one call).
MULTI_PAIRS = [("mw", "logp"), ("hbd", "hba"), ("ring_count", "validity"),
               ("mw", "tpsa"), ("lipinski", "logp")]


def _sample_smiles(n, name="full", split="test"):
    """Reuse real molecules from the existing scaffold-split test set instead
    of hitting the network again -- keeps this benchmark deterministic and
    free of a new data source."""
    seen, out = set(), []
    for line in (DATA / name / f"{split}.jsonl").open():
        s = json.loads(line)["smiles"]
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    return out


def build(n_mols=15, name="full"):
    smiles = _sample_smiles(n_mols, name=name)
    rows = []
    for s in smiles:
        for prop, tmpl in PARAPHRASES.items():
            rows.append({"kind": "paraphrase", "question": tmpl.format(smiles=s),
                         "expected": [[s, prop]], "expects_tool": True})
    for s in smiles:
        for p1, p2 in MULTI_PAIRS:
            q = (f"For the molecule {s}, what is its "
                 f"{p1.replace('_', ' ')} and its {p2.replace('_', ' ')}?")
            rows.append({"kind": "multi", "question": q,
                         "expected": [[s, p1], [s, p2]], "expects_tool": True})
    for item in NO_TOOL:
        rows.append({"kind": "no_tool", "question": item["question"],
                     "expected": [], "expects_tool": False, "answer": item["answer"]})
    return rows


def evaluate(generate, rows):
    agg = defaultdict(lambda: {"n": 0, "correct": 0, "tool_valid": 0})
    fails = []
    for r in tqdm(rows):
        expected = [tuple(e) for e in r["expected"]]
        res = harness.run_episode_general(generate, r["question"], expected,
                                          expects_tool=r["expects_tool"])
        if r["kind"] == "no_tool" and res["final"] is not None:
            res["correct"] = res["correct"] and r["answer"] in res["final"].strip().lower()
        for key in (r["kind"], "_all"):
            agg[key]["n"] += 1
            agg[key]["correct"] += res["correct"]
            agg[key]["tool_valid"] += res["tool_valid"]
        if not res["correct"]:
            fails.append({"kind": r["kind"], "question": r["question"],
                          "final": res["final"], "tool_calls": res["tool_calls"]})
    return agg, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default="full", help="dataset dir to sample molecules from")
    ap.add_argument("--n-mols", type=int, default=15)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    rows = build(n_mols=args.n_mols, name=args.name)
    print(f"hard benchmark: {len(rows)} episodes "
          f"({sum(r['kind'] == 'paraphrase' for r in rows)} paraphrase, "
          f"{sum(r['kind'] == 'multi' for r in rows)} multi-call, "
          f"{sum(r['kind'] == 'no_tool' for r in rows)} tool-restraint)")

    model, tok = ev.load(args.model, args.adapter)
    agg, fails = evaluate(ev.make_generate(model, tok), rows)
    table = ev.report(agg, args.tag)
    print(table)
    if fails:
        print(f"\n{len(fails)} failures:")
        for f in fails:
            print(f)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_hard_{args.tag}.md").write_text(table + "\n")
    (RESULTS / f"eval_hard_{args.tag}.json").write_text(json.dumps(dict(agg), indent=2))
    (RESULTS / f"eval_hard_{args.tag}_fails.json").write_text(json.dumps(fails, indent=2))


if __name__ == "__main__":
    main()
