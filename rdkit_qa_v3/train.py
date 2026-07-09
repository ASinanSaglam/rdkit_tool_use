"""QLoRA SFT on gold native-tool-calling traces (v3). Single combined
training run across all six behaviors (single/multi/chain/no_tool/clarify/
error) -- not sequential fine-tunes, same catastrophic-forgetting rationale
as v2. This is the SFT warm-start for v3's GRPO stage (train_grpo.py), which
loads this adapter as its starting policy rather than the base model.

Identical QLoRA setup to v2's train.py (dtype pinned at load via `dtype=`,
non-reentrant gradient checkpointing, `autocast_adapter_dtype=False`, SDPA) --
only load_traces() and the `tools=` column differ, since v3 declares two tool
schemas instead of one."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from . import harness

DATA = Path(__file__).resolve().parent.parent / "data" / "v3"
MODELS = Path(__file__).resolve().parent.parent / "artifacts" / "models" / "v3"


def load_traces(name: str, split: str = "train", limit: int | None = None):
    rows = [json.loads(l) for l in (DATA / name / f"{split}.jsonl").open()]
    if limit is not None:
        rows = rows[:limit]
    out = []
    for r in rows:
        kind = r["kind"]
        if kind == "single":
            msgs = harness.make_sft_trace(r["question"], "single", smiles=r["smiles"], prop=r["prop"])
        elif kind == "error":
            msgs = harness.make_sft_trace(r["question"], "error", smiles=r["smiles"], prop=r["prop"])
        elif kind == "multi":
            msgs = harness.make_sft_trace(r["question"], "multi",
                                          pairs=[tuple(p) for p in r["pairs"]])
        elif kind == "chain":
            msgs = harness.make_sft_trace(r["question"], "chain", name=r["name"], prop=r["prop"])
        elif kind == "no_tool":
            msgs = harness.make_sft_trace(r["question"], "no_tool", answer=r["answer"])
        elif kind == "clarify":
            msgs = harness.make_sft_trace(r["question"], "clarify", clarify_text=r["clarify_text"])
        else:
            raise ValueError(f"unknown kind: {kind}")
        out.append({"messages": msgs, "tools": harness.TOOL_SCHEMAS})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full", help="dataset dir under data/v3/")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    print(f"using compute dtype: {compute_dtype}")

    from datasets import Dataset
    t0 = time.time()
    train_ds = Dataset.from_list(load_traces(args.name, "train", limit=args.limit_train))
    print(f"train traces: {len(train_ds)} (loaded in {time.time() - t0:.1f}s)")
    kinds = {}
    for row in load_traces(args.name, "train", limit=args.limit_train):
        n_calls = sum(1 for m in row["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
        kinds[n_calls] = kinds.get(n_calls, 0) + 1
    print(f"tool-calls-per-episode distribution: {kinds}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype,
                             bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", dtype=compute_dtype,
        attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"model loaded in {time.time() - t0:.1f}s")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg, autocast_adapter_dtype=False)

    out = args.out or str(MODELS / args.name)
    cfg = SFTConfig(
        output_dir=out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=not args.fp16,
        fp16=args.fp16,
        max_length=896,           # longer than v2 -- chain traces add a 4th turn
        packing=False,
        report_to="none",
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok)
    t0 = time.time()
    result = trainer.train()
    print(f"train() wall time: {time.time() - t0:.1f}s -- {result.metrics}")
    trainer.save_model(out)
    print(f"saved final adapter -> {out}")


if __name__ == "__main__":
    main()
