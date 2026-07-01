# PROJECT.md — build log & decisions

Internal log for the RDKit tool-use QLoRA project. What's built, what's decided,
what's next. README is the outward-facing pitch; this is the honest working log.
Original brief preserved at the bottom.

## Decisions locked

| Decision | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-3B-Instruct | Small, strong tool-use, iterate fast on limited compute |
| Fine-tune | QLoRA (4-bit NF4, r16/α32) | Fits Colab T4; standard stack |
| Properties (8) | mw, logp, tpsa, hbd, hba, ring_count, lipinski, validity, canonical | All exactly RDKit-verifiable |
| Data source | ZINC-canonicalized (HF, streamed) | Drug-like by construction; download scales with target |
| Filter | none by default | ZINC already drug-like; `--drug-like` (Lipinski+MW) kept for later checkpoint experiments |
| Split | Bemis-Murcko scaffold, 80/10/10 | No train/test leakage |
| Sizes | dev 200 mols, full 10k mols | dev = fast local smoke; full = training |
| Reward | float ±0.1 tol; int/bool/str exact; canonical = SMILES equivalence | Robust to formatting, still real |
| Metrics | accuracy + tool-call validity (separate) | Split "called right tool" vs "reported right answer" |

## Compute split (hard constraint)

Local machine = **RTX 2060, 6GB, WSL2**. A model load once crashed WSL.

- **Local (2060):** data gen, oracle/harness self-tests, inference-only baseline eval
  at small `--limit`. All local GPU runs use `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
- **Colab T4 (16GB):** all QLoRA training; full evals. 3B QLoRA training does not fit 6GB.

## Architecture

`oracle.py` is the single source of truth: both dataset labels and eval scoring import
it, so train/test labels can never drift. `harness.py` is model-agnostic — it takes a
`generate(messages)->text` fn, so the same tool-call loop drives the fake test agent,
the HF eval model, and (via `make_sft_trace`) the SFT data. Protocol is plain text
(`TOOL_CALL:` / `TOOL_RESULT:` / `FINAL:`) not native function-calling JSON — ports
across models and is trivial to SFT on.

## Status

Done:
- [x] `oracle.py` — ground truth + verifiable scoring (self-test passing)
- [x] `dataset.py` — ZINC streaming, scaffold split, validity corruption (self-checked)
- [x] `harness.py` — agentic loop + episode scoring (self-test passing)
- [x] `eval.py` — benchmark runner (base/+LoRA), per-property table
- [x] `train.py` — QLoRA SFT, checkpoint saving (written; runs on Colab)
- [x] Repo scaffold: README, Makefile, pinned requirements, .gitignore
- [x] `data/dev/` built (160/20/20 mols)
- [ ] `data/full/` (10k) — building

Next:
- [ ] Local baseline smoke (`make baseline`, memory-capped) — first base number
- [ ] git init + push to GitHub (ASinanSaglam), repo name `rdkit_tool_use`
- [ ] Colab: QLoRA train on `full`, save checkpoints
- [ ] Eval base vs FT vs checkpoints → fill README results table
- [ ] Writeup + honest résumé bullet (only after numbers exist)

## Open risks / to verify

- **`assistant_only_loss=True`** needs Qwen's chat template to mark generation regions.
  If TRL errors on Colab: fall back to completion-only collator or full-sequence loss.
- **Baseline truncation:** `max_new_tokens=96`, greedy. If base model rambles past it
  before `FINAL:`, scores as format failure (fair, but bump tokens if it looks unfair).
- **Reward saturation:** the tool returns the exact answer, so a correct call + faithful
  copy = correct. That's the intended skill, but means accuracy tracks tool-use fidelity,
  not calculation ability. State plainly in the writeup — no overclaim.

## Stretch (cut first if timeline slips)

Rejection-sampling / GRPO on the verifiable reward. Clearly marked as stretch.

---

## Original brief

```
Project: LoRA fine-tune an LLM to use RDKit for molecular-property QA, with a verifiable-reward benchmark

Context. I'm a computational chemist / scientific software dev (PhD comp chem; ML for molecular
property prediction; production Python). Building a small, real, benchmarked project to earn an
honest résumé bullet covering: LLM post-training (SFT/LoRA), RL/eval from verifiable rewards,
agentic tool-use, ML-for-biology eval design. Target: Research Scientist, Life Sciences at Anthropic.
Done in ~1–2 weeks, measurable result stated truthfully. No overclaiming — every number from a run
actually executed.

Goal. Fine-tune a small open model (Qwen2.5-3B/7B-Instruct or Llama-3.1-8B) with QLoRA to answer
molecular-property questions by calling RDKit as a tool, measure accuracy vs automatic verifiable
ground truth. Report fine-tuned vs base.

Task: SMILES + property question (MW, logP, TPSA, HBD/HBA, ring count, Lipinski, validity,
canonicalization). Agentic: model tool call → harness runs RDKit → final answer. Reward = match vs
RDKit truth. Dataset from public SMILES set, scaffold split, ~5–20k QA.

Phases: (1) data + harness + baseline; (2) SFT/QLoRA on tool-use traces; (3) eval base vs FT,
per-property. Stretch: RLVR/reward-filtered pass (GRPO or rejection-sampling).

Deliverables: clean repo (README, make/scripts, pinned env) → GitHub (ASinanSaglam); results table
+ writeup; one honest résumé bullet + CV paragraph. No claim beyond runs.

Constraints: QLoRA experience already. Standard stack (PyTorch, transformers, peft/TRL, RDKit,
datasets). QLoRA not full FT. Ship minimal first, benchmark before optimizing, cut stretch if
timeline slips. Ask about GPU/compute, base model, property scope before heavy work.
```
