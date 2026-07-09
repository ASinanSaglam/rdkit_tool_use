#!/usr/bin/env python
"""Interactive local chat with the v2 (native tool-calling) fine-tuned adapter.

Run from the repo root: python chat_v2.py [--adapter PATH]

Loads base Qwen2.5-3B-Instruct (4-bit) + a LoRA adapter, drives the native
tool_calls/tool_response loop (rdkit_qa_v2.harness), printing every turn --
raw model output, parsed call, tool result -- so you can see what the model
actually does, not just its final answer.

Type a question, or /quit to exit. 4-bit inference only, fits a 6GB 2060.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rdkit_qa_v2 import harness

# results live outside the repo (too large to commit) -- see memory:
# rdkit-tool-use-compute-split
DEFAULT_ADAPTER = ("/home/zhedd/ml_playground/rdkit_tool_use_res/v2/rdkit_v2_results/"
                    "models/v2/full/checkpoint-100")
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
        enc = tok.apply_chat_template(messages, tools=[harness.TOOL_SCHEMA],
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
        result = harness.run_tool(call)
        print(f"  tool>  {result}")
        msgs += [harness.tool_call_message(call), harness.tool_result_message(call)]
    print("  (hit MAX_TURNS without a final answer -- stopping)")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = ap.parse_args()

    model, tok = load(args.adapter)
    generate = make_generate(model, tok)
    print("Type a question (molecular-property or otherwise). /reset clears "
          "conversation history, /quit exits.")
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
