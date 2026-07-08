"""Benchmark a v2 model (base or LoRA-adapted) across all five behaviors:
single-property, multi-property, tool-restraint, clarify, and tool-error
honesty. Same model-loading pattern as v1's eval.py (deprecated `torch_dtype=`
kwarg fix included -- this bit us twice already, not repeating it)."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import harness

DATA = Path(__file__).resolve().parent.parent / "data" / "v2"
RESULTS = Path(__file__).resolve().parent.parent / "results" / "v2"


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
        enc = tok.apply_chat_template(messages, tools=[harness.TOOL_SCHEMA],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def _row_expected_and_kind(r):
    kind = r["kind"]
    if kind == "single":
        return [(r["smiles"], r["prop"])], True, kind
    if kind == "multi":
        return [tuple(p) for p in r["pairs"]], True, kind
    if kind == "error":
        return [(r["smiles"], r["prop"])], True, kind
    if kind in ("no_tool", "clarify"):
        return [], False, kind
    raise ValueError(f"unknown kind: {kind}")


def evaluate(generate, rows):
    agg = defaultdict(lambda: {"n": 0, "correct": 0, "tool_valid": 0})
    fails = []
    for r in tqdm(rows):
        expected, expects_tool, kind = _row_expected_and_kind(r)
        res = harness.run_episode(generate, r["question"], expected,
                                  expects_tool=expects_tool, kind=kind)
        for key in (kind, "_all"):
            agg[key]["n"] += 1
            agg[key]["correct"] += res["correct"]
            agg[key]["tool_valid"] += res["tool_valid"]
        if not res["correct"]:
            fails.append({"kind": kind, "question": r["question"],
                          "final": res["final"], "tool_calls": res["tool_calls"]})
    return agg, fails


def report(agg, tag):
    lines = [f"# Eval (v2): {tag}\n", "| kind | n | accuracy | tool-valid |", "|---|---|---|---|"]
    for k in sorted(agg, key=lambda x: (x != "_all", x)):
        a = agg[k]
        lines.append(f"| {k} | {a['n']} | {a['correct']/a['n']:.3f} | {a['tool_valid']/a['n']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", default="dev", help="dataset dir under data/v2/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (DATA / args.name / f"{args.split}.jsonl").open()]
    if args.limit:
        rows = rows[: args.limit]
    model, tok = load(args.model, args.adapter)
    agg, fails = evaluate(make_generate(model, tok), rows)
    table = report(agg, args.tag)
    print(table)
    if fails:
        print(f"\n{len(fails)} failures (showing up to 10):")
        for f in fails[:10]:
            print(f)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"eval_{args.tag}.md").write_text(table + "\n")
    (RESULTS / f"eval_{args.tag}.json").write_text(json.dumps(dict(agg), indent=2))
    (RESULTS / f"eval_{args.tag}_fails.json").write_text(json.dumps(fails, indent=2))


if __name__ == "__main__":
    main()
