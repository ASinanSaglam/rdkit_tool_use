# Write-up: RDKit Tool-Use QA — QLoRA fine-tune

Full methodology and honest caveats behind the numbers in the README. 
See README for the quick pitch; this is the detailed log.

## Task recap

Qwen2.5-3B-Instruct answers a molecular-property question by emitting a
`TOOL_CALL`, receiving a `TOOL_RESULT` from RDKit (the oracle, ground truth),
then reporting `FINAL: <value>`. Reward is independently recomputed by
`oracle.score` — never trusts the model's own arithmetic.

## Data & training scope

- 74,042 train / 9,252 val / 9,252 test rows generated from ZINC-canonicalized,
  Bemis-Murcko scaffold split (no leakage).
- **Trained on 1,000 of the 74k train rows (~63 optimizer steps, ~1/12 epoch),
  not the full set.** This was a deliberate call, not a shortcut: a 30-step
  smoke test showed `mean_token_accuracy` already at 0.9885 by step 5 and
  0.9984 by step 10 (160 examples seen). The task — emit a well-formed tool
  call, then faithfully copy the tool's result — saturates almost immediately;
  full-loss curve confirms it:

  | step | epoch | loss | token accuracy |
  |---|---|---|---|
  | 20 | 0.32 | 0.0376 | 99.57% |
  | 40 | 0.64 | 0.0022 | 99.96% |
  | 60 | 0.96 | 0.0019 | 99.94% |

  Loss drops ~20x and token accuracy crosses 99.5% within the first 320
  examples — full-dataset training would cost far more for no measurable
  benefit over ~63 steps.

## Results

`--limit 500` on `data/full/test.jsonl` (of 9,252 total), base = zero-shot
Qwen2.5-3B-Instruct, ft = QLoRA adapter after 63 steps on 1,000 examples.

| property | n | base acc | ft acc | base tool-valid | ft tool-valid |
|---|---|---|---|---|---|
| _all | 500 | 0.636 | 0.996 | 0.598 | 0.990 |
| mw | 63 | 0.698 | 1.000 | 0.698 | 0.984 |
| logp | 56 | 0.679 | 0.964 | 0.696 | 0.964 |
| hbd | 55 | 0.618 | 1.000 | 0.582 | 1.000 |
| hba | 51 | 0.647 | 1.000 | 0.588 | 1.000 |
| ring_count | 54 | 0.759 | 1.000 | 0.704 | 1.000 |
| lipinski | 55 | 0.982 | 1.000 | 0.691 | 1.000 |
| validity | 57 | 0.877 | 1.000 | 0.789 | 1.000 |
| canonical | 49 | 0.490 | 1.000 | 0.673 | 1.000 |

### Reading the numbers honestly

- **Continuous properties track tool-valid closely in the base model**
  (mw 0.698/0.698, logp 0.679/0.696, ring_count 0.759/0.704) — when the base
  model calls the tool correctly, it almost always copies the result
  faithfully. The bottleneck is *calling* the tool right, not reading the
  answer back. This is exactly the skill the fine-tune targets.
- **`lipinski` base accuracy (0.982) is inflated by class imbalance**, not
  reasoning skill — ZINC molecules are drug-like by construction, so most
  pass Lipinski's rule, and a model can score high by guessing "true"
  regardless of tool-call correctness (tool-valid is only 0.691). Same
  pattern, smaller, on `validity` (0.877 acc vs 0.789 valid). These two
  properties are less informative for measuring the fine-tune's actual effect
  than the continuous ones.
- **`canonical` had a base-model-specific failure mode distinct from
  tool-calling**: tool-valid 0.673 but accuracy only 0.490 — the model often
  identified the right molecule but garbled reproducing the SMILES string
  itself (a generation-fidelity gap, not a tool-use gap). Fully resolved
  post-fine-tune (1.000/1.000).
- **`logp` is the only ft property below 1.000** (0.964, 2/56 misses).
  Sample size is small enough (n=2) that this could be tolerance-boundary
  float noise (`|predicted - truth| <= 0.1`) rather than a systematic gap —
  pending failure-dump inspection (`eval.py --property logp` records
  `results/eval_<tag>_fails.json`) before concluding anything.
- **Scope of the claim**: near-100% ft accuracy means the model learned to
  call the tool correctly and copy its result faithfully — a real, useful
  agentic skill. It does **not** mean the model can compute molecular
  properties itself; the tool does all the actual chemistry. Stated plainly
  so the number doesn't read as an overclaim.

## Harder benchmark: generalization beyond the trained protocol

The original benchmark only tests "one SMILES, one known property, exactly
one tool call is always correct." That's necessary but not sufficient to
claim general tool-use competence — a model can get 99.6% on that by
memorizing a rigid two-turn template. `bench_hard.py` (new, separate from
the main pipeline so it never touches `data/`, `dataset.py`, or the trained
artifacts) probes three specific failure modes the base benchmark can't see:

1. **Paraphrase robustness** — the same 9 properties, asked with reworded,
   informal phrasing instead of the exact training templates (e.g. "How
   heavy is one mole of X, in grams?" instead of "What is the molecular
   weight of X?"). Tests whether the model generalized the protocol or
   pattern-matched on exact question wording.
2. **Tool-restraint** — 5 plain trivia/arithmetic questions with no
   molecule or RDKit content at all. Correct behavior is answering directly;
   emitting a `TOOL_CALL` here is a false-positive tool-use failure the
   original benchmark never exercises (every question in it genuinely needs
   the tool).
3. **Multi-tool-call composition** — questions asking for 2 properties of
   one molecule in a single turn (e.g. "what's the MW and logP of X?").
   Requires the model to chain two tool calls before giving one combined
   final answer — a behavior the training data never demonstrates (every SFT
   trace is exactly one call, one result, one final line).

**This needed a harness change**, not just new data: the original
`harness.run_episode` hardcodes a fixed one-call, one-final exchange (the
exact shape of the training protocol). Added `harness.run_episode_general`
alongside it (existing pipeline untouched) — loops up to `MAX_TURNS=6`
tool-call/result round-trips until the model emits `FINAL:` or stops
producing a parseable call, and scores against a *list* of expected
`(smiles, property)` pairs rather than one. Grading a multi-property final
answer is necessarily looser (`_final_covers` checks the whole line, then
each token split on whitespace/punctuation/`=`, since the model packs
several values into one line) — this is a real scoring approximation, not
exact-match like the main benchmark, and should be read that way.

Run it (reuses `eval.py`'s model loading, no duplicated code):

```bash
python -m rdkit_qa.bench_hard --name full --tag base                       # zero-shot
python -m rdkit_qa.bench_hard --name full --adapter artifacts/models/full --tag ft
```

Samples molecules from the existing `data/full/test.jsonl` (no new data
source) — for this run, 15 molecules × 9 paraphrase templates + 5
multi-property pairs + 5 fixed trivia questions = 215 episodes.

### Results

| kind | n | base acc | ft acc | base tool-valid | ft tool-valid |
|---|---|---|---|---|---|
| paraphrase | 135 | 0.637 | 0.941 | 0.970 | 1.000 |
| multi (2-call) | 75 | 0.080 | 0.120 | 0.173 | 0.387 |
| no_tool (restraint) | 5 | 0.400 | 0.000 | 0.600 | 0.400 |

**Paraphrase robustness held up well post-fine-tune** (63.7% → 94.1%
accuracy, tool-valid already near-ceiling at 97.0% even in the base model) —
the model generalized the *protocol* rather than memorizing exact training
question wording. This is the one result here that's genuinely reassuring
about what the fine-tune learned.

**Multi-call composition is weak in both models**, as expected — the SFT
data never demonstrates more than one tool call per episode, so there was no
reason to expect the fine-tune to teach chaining. Fine-tuning does help some
(8.0% → 12.0% accuracy, 17.3% → 38.7% tool-valid — more of the *first* call
in a 2-call sequence lands correctly) but the skill was never trained,
only tested, and the numbers say exactly that.

**Fine-tuning made tool-restraint *worse*, not better** — 0/5 correct on
the no-tool trivia questions post-ft, down from 2/5 in the base model
(tool-valid, i.e. correctly not calling the tool at all, also drops: 60% →
40%). This is the most useful single result in this benchmark: training
exclusively on tool-necessary examples plausibly taught the model to
*always* reach for the tool, a real regression the main property-QA
benchmark has no way to see (every question in it genuinely needs RDKit).
Direct evidence, not a guess, for why any future expansion of this project
needs negative/no-tool-needed examples in the training mix — see the
roadmap note above.

Sample size caveat: `no_tool` is only 5 questions — the direction (regression)
is worth taking seriously, the exact magnitude isn't precise at n=5.

**Not yet built**: new oracle-verifiable RDKit skills beyond the current 9
properties (substructure match via SMARTS, Tanimoto similarity on Morgan
fingerprints, functional-group counting). These extend *what* can be asked
rather than *how* it's asked, and need their own oracle/dataset schema work
— scoped as a follow-up, not started.
