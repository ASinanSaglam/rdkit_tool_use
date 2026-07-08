# RDKit Tool-Use QA — QLoRA fine-tuning with a verifiable-reward benchmark

Fine-tune a small open LLM (Qwen2.5-3B-Instruct) to answer molecular-property
questions **by calling RDKit as a tool**, then measure accuracy against an
automatic ground truth (RDKit computes the true value). Report base vs
fine-tuned, per property.

Every number in the results table comes from a run actually executed in this
repo — no estimated or hand-waved metrics.

## The task

Input: a SMILES + a property question. The model must emit a tool call, the
harness runs RDKit, the model reports the result:

```
USER:      What is the molecular weight of CCO?
ASSISTANT: TOOL_CALL: {"smiles": "CCO", "property": "mw"}
TOOL:      TOOL_RESULT: 46.069
ASSISTANT: FINAL: 46.069
```

**Verifiable reward:** `oracle.score` independently recomputes the truth and
checks the final answer (float tolerance / exact int / bool / canonical-SMILES
equivalence). The measured skills, reported separately:

- **accuracy** — is the final answer right?
- **tool-call validity** — did the model call the tool with the *correct*
  molecule + property? (Scoring the wrong molecule can't be rewarded.)

Properties: `mw, logp, tpsa, hbd, hba, ring_count, lipinski, validity, canonical`.

## Data

Molecules streamed from [ZINC-canonicalized](https://huggingface.co/datasets/sagawa/ZINC-canonicalized)
(drug-like by construction), deduped on canonical SMILES, **scaffold-split**
(Bemis-Murcko) so no scaffold spans train/test — no leakage. Labels are computed
by the same oracle used for scoring, so train and eval can never drift.

- `data/dev/`  — 200 mols, fast local smoke tests
- `data/full/` — 10k mols, training

## Layout

```
rdkit_qa/
  oracle.py    # RDKit ground truth + verifiable scoring (single source of truth)
  dataset.py   # QA generation + scaffold split
  harness.py   # agentic tool-call loop + episode scoring (model-agnostic)
  eval.py      # benchmark a model (base or +LoRA), per-property table
  bench_hard.py  # generalization eval: paraphrase, tool-restraint, multi-call
  train.py     # QLoRA SFT on gold tool-use traces (Colab T4)
data/           # generated splits (gitignored)
artifacts/models/  # LoRA adapters + checkpoints
results/        # eval tables (base vs ft vs checkpoints)
```

## Reproduce

```bash
make setup        # pinned deps
make test         # oracle + harness self-checks (no GPU, no network-heavy)
make data-dev     # tiny dataset
make data-full    # 10k training dataset
make baseline     # base-model smoke on a 6GB GPU (5 examples, memory-capped)
```

### Training on Colab (T4, 16GB)

The 3B QLoRA train step does **not** fit a 6GB local GPU — run `train.py` on
Colab. Colab ships its own torch; install the rest:

```bash
pip install transformers peft trl datasets bitsandbytes accelerate rdkit
python -m rdkit_qa.train --name full --save-steps 100
```

Then evaluate base vs fine-tuned (and each checkpoint) with `eval.py`:

```bash
python -m rdkit_qa.eval --name full --tag base
python -m rdkit_qa.eval --name full --adapter artifacts/models/full --tag ft
```

## Results

QLoRA fine-tune (63 steps, 1,000 train examples — see write-up for why that's
enough) takes overall accuracy from **63.6% → 99.6%** and tool-call validity
from **59.8% → 99.0%** against zero-shot Qwen2.5-3B-Instruct, `--limit 500` on
held-out scaffold-split test data.

| property | base acc | ft acc | base tool-valid | ft tool-valid |
|---|---|---|---|---|
| _all_ | 0.636 | 0.996 | 0.598 | 0.990 |

Full per-property breakdown, and honest caveats (class-imbalance inflation on
lipinski/validity, the one open logp gap, planned harder benchmarks): see
**[WRITE_UP.md](WRITE_UP.md)**.

## v2: native tool-calling, five behaviors

`rdkit_qa_v2/` (new, `rdkit_qa/` above is frozen as-is) rebuilds the same
idea on Qwen's real chat-template tool-calling instead of a hand-rolled
plain-text protocol, and trains one combined skill set instead of one:
single-property lookup, multi-property composition, tool-restraint
(no-tool-needed questions), asking for missing required info instead of
guessing, and honestly reporting a failed tool call instead of hallucinating
a value. Motivated directly by `bench_hard`'s v1 finding that fine-tuning on
tool-necessary examples alone made restraint *worse*, not better. See
**[WRITE_UP_V2.md](WRITE_UP_V2.md)** (in progress).

## Scope & honesty

- QLoRA (not full FT), standard stack: PyTorch, transformers, peft/TRL, RDKit.
- Ship the minimal working benchmark first; optimize only after a baseline exists.
- Stretch (only if time): rejection-sampling / GRPO on the verifiable reward.
  Clearly marked, cut first if the timeline slips.
