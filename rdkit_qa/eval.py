"""Run the verifiable benchmark on a model (base or LoRA-adapted).
Reports overall accuracy, tool-call validity rate, and per-property breakdown.
Fits a 6GB GPU via 4-bit load + short generations."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import harness

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"


def load(model_id: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def make_generate(model, tok):
    @torch.no_grad()
    def generate(messages):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=96, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def evaluate(generate, rows):
    agg = defaultdict(lambda: {"n": 0, "correct": 0, "tool_valid": 0})
    fails = []
    for r in tqdm(rows):
        res = harness.run_episode(generate, r["question"], r["smiles"], r["property"])
        for key in (r["property"], "_all"):
            agg[key]["n"] += 1
            agg[key]["correct"] += res["correct"]
            agg[key]["tool_valid"] += res["tool_valid"]
        if not res["correct"]:
            fails.append({"property": r["property"], "smiles": r["smiles"],
                          "final": res["final"], "tool_valid": res["tool_valid"]})
    return agg, fails


def report(agg, tag):
    lines = [f"# Eval: {tag}\n",
             "| property | n | accuracy | tool-valid |",
             "|---|---|---|---|"]
    for k in sorted(agg, key=lambda x: (x != "_all", x)):
        a = agg[k]
        lines.append(f"| {k} | {a['n']} | {a['correct']/a['n']:.3f} | "
                     f"{a['tool_valid']/a['n']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default="dev", help="dataset dir under data/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--property", default=None, help="only eval one property, e.g. logp")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA / args.name / f"{args.split}.jsonl").open()]
    if args.property:
        rows = [r for r in rows if r["property"] == args.property]
    if args.limit:
        rows = rows[: args.limit]
    model, tok = load(args.model, args.adapter)
    agg, fails = evaluate(make_generate(model, tok), rows)
    table = report(agg, args.tag)
    print(table)
    if fails:
        print(f"\n{len(fails)} failures:")
        for f in fails:
            print(f)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{args.tag}.md").write_text(table + "\n")
    (RESULTS / f"eval_{args.tag}.json").write_text(json.dumps(dict(agg), indent=2))
    (RESULTS / f"eval_{args.tag}_fails.json").write_text(json.dumps(fails, indent=2))


if __name__ == "__main__":
    main()
