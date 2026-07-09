# Write-up: v2 — native tool-calling, five behaviors

In progress. See [WRITE_UP.md](WRITE_UP.md) for the v1 results this builds on
and the series framing (v1 established the tool-use mechanic without relying
on native chat-template support; v2 below adds that support plus restraint;
v3 adds a second tool, tool chaining, and GRPO on top).

## Why v2 exists

Two gaps `bench_hard.py` found in v1, both real, neither fixable by training
more of the same data:

1. **Protocol was hand-rolled plain text** (`TOOL_CALL:`/`TOOL_RESULT:`/
   `FINAL:`), not the model's real tool-calling interface. Fine for a
   controlled from-scratch benchmark, not representative of production
   tool-use.
2. **Fine-tuning regressed tool-restraint**: v1's ft model scored 0/5 on
   no-tool-needed questions, down from 2/5 in the base model (see
   WRITE_UP.md's `bench_hard` section) — training exclusively on
   tool-necessary examples taught the model to always reach for the tool.

## What changed

- **Protocol**: Qwen2.5's native tool-calling, verified directly against the
  actual `chat_template.jinja` shipped with the checkpoint (not assumed from
  docs) — tools declared via `tools=`, assistant tool calls as a
  `tool_calls` message field, results as `role: tool` messages. No more
  `FINAL:` marker: a final answer is just an assistant message without
  `tool_calls`, which the native format already signals structurally.
- **Five trained behaviors in one combined dataset**, not five sequential
  fine-tunes (avoids catastrophic forgetting between them):
  - `single` — one property, one molecule (v1's core skill)
  - `multi` — two properties, one molecule, chained tool calls
  - `no_tool` — arithmetic/trivia/chemistry-adjacent questions that don't
    need RDKit; correct behavior is answering directly
  - `clarify` — a question needs a molecule but no SMILES was given; correct
    behavior is asking for it, not guessing
  - `error` — tool is called correctly but the SMILES is invalid; correct
    behavior is honestly reporting the failure, not hallucinating a value
- Paraphrase robustness folded into `single`/`multi` rows directly (random
  phrasing per row) rather than kept as a separate category — v1's
  `bench_hard` already showed this generalizes well without needing to be a
  distinct training signal.
- Reuses v1's `oracle.py` (thin re-export, ground truth doesn't change) and
  SMILES sourcing/scaffold-split (`rdkit_qa.dataset.get_smiles`,
  `scaffold_split`) rather than duplicating either.

## Open / unverified

- ~~Whether TRL's `SFTTrainer` respects a per-example `tools` dataset column~~
  **Confirmed.** Smoke test (A100, 30 traces, 10 steps): loss 0.673 → 0.0060,
  `mean_token_accuracy` 0.967 → 0.998 over 5 epochs — a normal SFT curve, not
  the flat/random signature that'd show if `tools` were silently dropped.
- Dataset generation (`rdkit_qa_v2/dataset.py`) is written and smoke-tested
  locally at small `--n-mols` (structural correctness, kind distribution
  scales reasonably with dataset size) but not yet run at training scale.
- No `bench_hard`-equivalent built for v2 yet — the five kinds now live
  directly in the main train/test split, so the main eval (`eval.py`)
  already covers what `bench_hard` covered separately in v1. Worth
  confirming that's sufficient once real results exist, rather than
  assuming it replaces the harder-benchmark role entirely.
- No results yet. Numbers to be filled in after a real training run.
