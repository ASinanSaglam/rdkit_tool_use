"""QLoRA SFT on gold tool-use traces. Built for a single 16GB GPU (Colab T4).

Trains the model to emit a valid TOOL_CALL then faithfully report TOOL_RESULT.
Saves intermediate checkpoints (--save-steps) so eval can chart the
base -> checkpoint -> final progression.

Local 6GB note: this will NOT fit a 6GB 2060 for a 3B model. Run on Colab/cloud.
Everything else in the repo (data, harness, eval at --limit) runs locally."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from . import harness

DATA = Path(__file__).resolve().parent.parent / "data"
MODELS = Path(__file__).resolve().parent.parent / "artifacts" / "models"


def load_traces(name: str, split: str = "train", limit: int | None = None):
    """Each QA row -> a full gold conversation (list of chat messages)."""
    rows = [json.loads(l) for l in (DATA / name / f"{split}.jsonl").open()]
    if limit is not None:
        rows = rows[:limit]
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
    ap.add_argument("--max-steps", type=int, default=-1,
                     help="stop after N optimizer steps (smoke tests); -1 = full epochs")
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--fp16", action="store_true",
                     help="use fp16 instead of bf16 (T4 has no native bf16 tensor cores)")
    ap.add_argument("--packing", action="store_true",
                     help="pack multiple short traces per 512-token sequence")
    ap.add_argument("--no-checkpointing", action="store_true",
                     help="disable gradient checkpointing (more VRAM, less recompute)")
    ap.add_argument("--limit-train", type=int, default=None,
                     help="use only the first N train traces (task saturates well under full set)")
    args = ap.parse_args()

    print(f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a'}")
    compute_dtype = torch.float16 if args.fp16 else torch.bfloat16
    print(f"using compute dtype: {compute_dtype}")

    from datasets import Dataset
    t0 = time.time()
    train_ds = Dataset.from_list(load_traces(args.name, "train", limit=args.limit_train))
    print(f"train traces: {len(train_ds)} (loaded in {time.time() - t0:.1f}s)")

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
    for name, mod in model.named_modules():
        if hasattr(mod, "compute_dtype"):
            print(f"quantized layer {name}: compute_dtype={mod.compute_dtype}, "
                  f"weight.dtype={mod.weight.dtype}")
            break
    # use_reentrant=False: reentrant checkpointing recomputes backward in a fresh
    # autocast context that can silently default to bf16 on a GPU that merely
    # self-reports bf16 support (T4 does) -- breaks fp16 GradScaler downstream.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=not args.no_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"model loaded in {time.time() - t0:.1f}s")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg, autocast_adapter_dtype=False)

    trainable_dtypes = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_dtypes.setdefault(str(p.dtype), []).append(n)
    for dt, names in trainable_dtypes.items():
        print(f"trainable dtype {dt}: {len(names)} params, e.g. {names[:3]}")

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
        max_length=512,           # traces are short; caps memory
        packing=args.packing,
        report_to="none",
        # only learn the assistant turns (TOOL_CALL / FINAL), not system+user text
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         processing_class=tok)
    print(f"accelerator mixed_precision={trainer.accelerator.mixed_precision}, "
          f"scaler={getattr(trainer.accelerator, 'scaler', None)}")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    print(f"train() wall time: {elapsed:.1f}s -- {result.metrics}")
    trainer.save_model(out)
    print(f"saved final adapter -> {out}")


if __name__ == "__main__":
    main()
