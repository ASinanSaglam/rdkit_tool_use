#!/usr/bin/env python
"""Interactive local chat with the v3 (two-tool) fine-tuned adapter.

Run from the repo root: python chat_v3.py [--adapter PATH]

Loads base Qwen2.5-3B-Instruct (4-bit) + a LoRA adapter, drives the native
tool_calls/tool_response loop (rdkit_qa_v3.harness) with BOTH tools declared
(rdkit_compute, name_to_smiles) -- and unlike training/eval, name_to_smiles
resolves through live PubChem (name_dict_pubchem.resolve_pubchem) instead of
the offline curated dict, so chat gets real-world coverage beyond the ~150
names baked into training data.

Type a question, or /quit to exit. 4-bit inference only, fits a 6GB 2060.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

import argparse, json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rdkit_qa_v3 import harness, name_dict_pubchem

# results live outside the repo (too large to commit) -- see memory:
# rdkit-tool-use-compute-split
DEFAULT_ADAPTER = ("/home/zhedd/ml_playground/rdkit_tool_use_res/v3/rdkit_v3_results/"
                    "models/v3/full/checkpoint-100")
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_TURNS = 6


def load(adapter):
    print(f"loading {MODEL_ID} (4-bit) + adapter from {adapter} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    print("ready.\n")
    return model, tok


def make_generate(model, tok):
    @torch.no_grad()
    def generate(messages):
        enc = tok.apply_chat_template(messages, tools=harness.TOOL_SCHEMAS,
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def chat_turn(generate, msgs, question):
    """Appends to and returns the running conversation `msgs` -- carries
    history across turns so follow-ups ("now do logP") can refer back to a
    molecule from an earlier turn, instead of each turn starting fresh."""
    msgs = msgs + [{"role": "user", "content": question}]
    for _ in range(MAX_TURNS):
        turn = generate(msgs)
        print(f"  model> {turn}")
        call = harness.parse_tool_call(turn)
        if call is None:
            if "<tool_call>" in turn:
                print(f"  final> [malformed tool call, treated as final] {turn.strip()}")
            else:
                print(f"  final> {turn.strip()}")
            msgs.append({"role": "assistant", "content": turn.strip()})
            return msgs
        result = harness.run_tool(call, name_resolver=name_dict_pubchem.resolve_pubchem)
        print(f"  tool[{call['tool']}]> {result}")
        msgs += [harness.tool_call_message(call),
                 {"role": "tool", "content": json.dumps(result)}]
    print("  (hit MAX_TURNS without a final answer -- stopping)")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = ap.parse_args()

    model, tok = load(args.adapter)
    generate = make_generate(model, tok)
    print("Type a question (molecular-property, common-name lookup, or "
          "otherwise). /reset clears conversation history, /quit exits.")
    msgs = [{"role": "system", "content": harness.SYSTEM}]
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q in ("/quit", "/exit"):
            break
        if q == "/reset":
            msgs = [{"role": "system", "content": harness.SYSTEM}]
            print("  (conversation reset)")
            continue
        msgs = chat_turn(generate, msgs, q)


if __name__ == "__main__":
    main()
