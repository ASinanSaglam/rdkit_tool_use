"""GRPO RL stage on top of the v3 SFT adapter (train.py) -- the payoff for
adding a second tool. SFT gold traces force one rigid call order
(name_to_smiles -> rdkit_compute); GRPO instead samples G completions per
prompt and rewards whichever call sequence actually gets the right answer,
which is the point of RL for multi-tool routing (see README/PROJECT.md
"Stretch: GRPO").

Uses trl's GRPOTrainer NATIVE `tools=` support (trl>=1.7, verified against
the installed version -- see rdkit_v3 env, NOT the repo's pinned
trl==0.24.0, which lacks any multi-turn rollout mechanism at all). Passing
plain Python callables as `tools=[...]` gets you the full generate -> parse
tool call -> execute -> inject result -> continue loop for free: no custom
rollout_func, no hand-rolled multi-turn generation loop. The `oracle`-backed
tool functions below are intentionally thin wrappers with the SAME name/
arg-name/type shape as harness.RDKIT_SCHEMA/NAME_SCHEMA, because the model
was SFT-warm-started on that exact schema -- GRPOTrainer auto-derives the
tool schema from the function signature+docstring (Google style), and a
mismatched shape would confuse a model that already learned the SFT one.

Scope (deliberately narrow first pass, see plan): trains on "chain" and
"single" rows only. Both reduce to the same reward check (does the final
answer match oracle.compute(target_smiles, prop)?) plus a penalty for
skipping/duplicating tool calls -- "multi" (two-property fan-out) and
no_tool/clarify/error (already SFT-exact-match-saturated, nothing for RL to
improve) are out of scope for v1 of this script.

Needs the `rdkit_v3` env (trl>=1.7, transformers>=4.56, accelerate>=1.4),
not `def` -- def's pinned trl==0.24.0 can't run this at all. See
requirements.txt comment / README for the two-environment split."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from . import oracle

DATA = Path(__file__).resolve().parent.parent / "data" / "v3"
MODELS = Path(__file__).resolve().parent.parent / "artifacts" / "models" / "v3_grpo"

SYSTEM = (
    "You answer questions, calling tools when you need them: rdkit_compute for "
    "molecular properties (requires a SMILES string), name_to_smiles to look up "
    "the SMILES for a molecule given its common name. If a question gives a "
    "common name instead of a SMILES string, call name_to_smiles first and use "
    "its result as the input to rdkit_compute."
)

MISSING_CALL_PENALTY = 0.3
EXTRA_CALL_PENALTY = 0.1


# --- tools: plain callables, schema auto-derived by GRPOTrainer from the
# signature + this docstring. Must mirror harness.RDKIT_SCHEMA/NAME_SCHEMA's
# function name and argument names exactly -- the SFT-warm-started model was
# trained to call THAT shape.

def rdkit_compute(smiles: str, property: str) -> str:
    """Compute a molecular property for a molecule using RDKit.

    Args:
        smiles: SMILES string of the molecule.
        property: Which property to compute.

    Returns:
        A JSON string: {"result": value} on success, {"error": msg} on failure.
    """
    val = oracle.compute(smiles, property)
    return json.dumps({"error": "invalid molecule"} if val is None else {"result": val})


def name_to_smiles(name: str) -> str:
    """Look up the SMILES string for a molecule given its common or trade name.

    Args:
        name: Common or trade name of the molecule, e.g. "acetone".

    Returns:
        A JSON string: {"result": smiles} on success, {"error": msg} on failure.
    """
    smi = oracle.compute_name(name)
    return json.dumps({"error": "unknown name"} if smi is None else {"result": smi})


def _final_covers(final_text: str, smiles: str, prop: str) -> bool:
    if oracle.score(prop, final_text, smiles):
        return True
    return any(oracle.score(prop, tok, smiles)
               for tok in re.split(r"[\s,;:=]+", final_text) if tok)


def _tool_call_names(completion: list[dict]) -> list[str]:
    names = []
    for msg in completion:
        for tc in (msg.get("tool_calls") or []):
            names.append(tc["function"]["name"])
    return names


def make_reward_func():
    def reward_func(prompts, completions, kind, smiles, prop, name, **kwargs) -> list[float]:
        rewards = []
        for completion, k, s, p, nm in zip(completions, kind, smiles, prop, name):
            final = None
            for msg in reversed(completion):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final = msg["content"]
                    break
            if final is None:
                rewards.append(0.0)
                continue

            correct = _final_covers(final, s, p)
            calls = _tool_call_names(completion)
            expected_n_calls = 2 if k == "chain" else 1
            penalty = 0.0
            if k == "chain" and "name_to_smiles" not in calls:
                penalty += MISSING_CALL_PENALTY
            if "rdkit_compute" not in calls:
                penalty += MISSING_CALL_PENALTY
            penalty += EXTRA_CALL_PENALTY * max(0, len(calls) - expected_n_calls)

            rewards.append(float(correct) - penalty)
        return rewards
    return reward_func


def load_rows(name: str, split: str, kinds: set[str]):
    path = DATA / name / f"{split}.jsonl"
    rows = [json.loads(l) for l in path.open() if json.loads(l)["kind"] in kinds]
    out = []
    for r in rows:
        if r["kind"] == "single":
            smi, prop, nm = r["smiles"], r["prop"], ""
        elif r["kind"] == "chain":
            smi, prop, nm = oracle.compute_name(r["name"]), r["prop"], r["name"]
            if smi is None:
                continue  # dataset.py already asserts every chain name resolves; skip defensively
        else:
            continue
        out.append({
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}],
            "kind": r["kind"], "smiles": smi, "prop": prop, "name": nm,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full", help="dataset dir under data/v3/")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sft-adapter", required=True, help="path to the v3 SFT adapter (train.py output)")
    ap.add_argument("--kinds", default="chain,single", help="comma-separated row kinds to train on")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--num-generations", type=int, default=4, help="G in the GRPO paper")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    kinds = set(args.kinds.split(","))

    from datasets import Dataset
    rows = load_rows(args.name, "train", kinds)
    if args.limit_train:
        rows = rows[: args.limit_train]
    train_ds = Dataset.from_list(rows)
    print(f"GRPO train prompts: {len(train_ds)}  kinds={kinds}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    print(f"loaded SFT adapter from {args.sft_adapter} as GRPO starting policy")

    out = args.out or str(MODELS / args.name)
    cfg = GRPOConfig(
        output_dir=out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_func(),
        args=cfg,
        train_dataset=train_ds,
        processing_class=tok,
        tools=[rdkit_compute, name_to_smiles],
    )
    result = trainer.train()
    print(f"train() metrics: {result.metrics}")
    trainer.save_model(out)
    print(f"saved GRPO-tuned adapter -> {out}")


if __name__ == "__main__":
    main()
