"""QLoRA SFT on gold tool-use traces. Built for a single 16GB GPU (Colab T4).

Trains the model to emit a valid TOOL_CALL then faithfully report TOOL_RESULT.
Saves intermediate checkpoints (--save-steps) so eval can chart the
base -> checkpoint -> final progression.

Local 6GB note: this will NOT fit a 6GB 2060 for a 3B model. Run on Colab/cloud.
Everything else in the repo (data, harness, eval at --limit) runs locally."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from . import harness

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = Path(__file__).resolve().parent.parent / "artifacts" / "models"


def load_traces(name: str, split: str = "train"):
    """Each QA row -> a full gold conversation (list of chat messages)."""
    rows = [json.loads(l) for l in (DATA / name / f"{split}.jsonl").open()]
    return [{"messages": harness.make_sft_trace(r["question"], r["smiles"], r["property"])}
            for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full", help="dataset dir under data/")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from datasets import Dataset
    train_ds = Dataset.from_list(load_traces(args.name, "train"))
    print(f"train traces: {len(train_ds)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    out = args.out or str(MODELS / args.name)
    cfg = SFTConfig(
        output_dir=out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        max_length=512,           # traces are short; caps memory
        packing=False,
        report_to="none",
        # only learn the assistant turns (TOOL_CALL / FINAL), not system+user text
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(out)
    print(f"saved final adapter -> {out}")


if __name__ == "__main__":
    main()
