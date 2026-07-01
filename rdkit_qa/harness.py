"""Agentic tool-call loop + verifiable scoring.

Protocol (plain text so it ports across models and is trivial to SFT on):
  assistant turn 1:  TOOL_CALL: {"smiles": "...", "property": "..."}
  tool reply:        TOOL_RESULT: <value>
  assistant turn 2:  FINAL: <answer>

The tool IS the RDKit oracle (ground truth). The measured skill: emit a valid
call with the right smiles+property, then faithfully report the result.
Verifiable reward = oracle.score(final answer) — computed independently."""
from __future__ import annotations
import json, re
from . import oracle

PROPS = list(oracle.PROPERTIES)

SYSTEM = (
    "You answer molecular-property questions using a tool. You have exactly one "
    "tool:\n"
    '  rdkit_compute(smiles: str, property: str) -> value\n'
    f"  property must be one of: {', '.join(PROPS)}\n"
    "First reply with ONE line, nothing else:\n"
    '  TOOL_CALL: {"smiles": "<smiles>", "property": "<property>"}\n'
    "You will then receive a line `TOOL_RESULT: <value>`. After that, reply with "
    "ONE final line:\n"
    "  FINAL: <value>\n"
    "Report the tool result directly. Do not compute it yourself."
)

_CALL_RE = re.compile(r"TOOL_CALL:\s*(\{.*?\})", re.S)
_FINAL_RE = re.compile(r"FINAL:\s*(.+?)\s*$", re.S | re.M)


def build_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]


def parse_tool_call(text: str):
    """Return dict with smiles+property, or None if not a valid call."""
    m = _CALL_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "smiles" not in d or d.get("property") not in oracle.PROPERTIES:
        return None
    return {"smiles": str(d["smiles"]), "property": d["property"]}


def run_tool(call: dict) -> str:
    """Execute the oracle and format the result line the model sees back."""
    val = oracle.compute(call["smiles"], call["property"])
    return f"TOOL_RESULT: {val}"


def parse_final(text: str):
    m = _FINAL_RE.search(text)
    return m.group(1).strip() if m else None


def run_episode(generate, question: str, smiles: str, prop: str) -> dict:
    """generate(messages) -> assistant text. Returns scoring breakdown.
    Note: scoring uses the episode's true (smiles, prop), not what the model
    called — calling the wrong molecule cannot be rewarded."""
    msgs = build_prompt(question)
    turn1 = generate(msgs)
    call = parse_tool_call(turn1)
    tool_valid = call is not None and call["smiles"] == smiles and call["property"] == prop
    if call is None:
        return {"tool_valid": False, "correct": False, "final": None}

    msgs = msgs + [
        {"role": "assistant", "content": turn1},
        {"role": "user", "content": run_tool(call)},
    ]
    turn2 = generate(msgs)
    final = parse_final(turn2)
    correct = final is not None and oracle.score(prop, final, smiles)
    return {"tool_valid": tool_valid, "correct": correct, "final": final}


def make_sft_trace(question, smiles, prop):
    """Gold two-turn trace for SFT. The assistant copies the tool result, which
    is the oracle ground truth — so the target answer is always correct."""
    call = {"smiles": smiles, "property": prop}
    result = oracle.compute(smiles, prop)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"TOOL_CALL: {json.dumps(call)}"},
        {"role": "user", "content": f"TOOL_RESULT: {result}"},
        {"role": "assistant", "content": f"FINAL: {result}"},
    ]


def demo():
    from . import oracle as _o
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    q = _o.QUESTION_TEMPLATES["mw"].format(smiles=aspirin)

    # perfect agent: plays the gold trace
    def good(messages):
        if messages[-1]["role"] == "user" and messages[-1]["content"].startswith("TOOL_RESULT"):
            return "FINAL: " + messages[-1]["content"].split("TOOL_RESULT:")[1].strip()
        return 'TOOL_CALL: {"smiles": "%s", "property": "mw"}' % aspirin

    r = run_episode(good, q, aspirin, "mw")
    assert r["tool_valid"] and r["correct"], r

    # broken-format agent: no tool call
    assert run_episode(lambda m: "the weight is 180", q, aspirin, "mw") == \
        {"tool_valid": False, "correct": False, "final": None}

    # wrong-molecule call: tool_valid False, and cannot be scored correct
    def wrong_mol(messages):
        if messages[-1]["content"].startswith("TOOL_RESULT"):
            return "FINAL: " + messages[-1]["content"].split(":", 1)[1].strip()
        return 'TOOL_CALL: {"smiles": "CCO", "property": "mw"}'
    r = run_episode(wrong_mol, q, aspirin, "mw")
    assert not r["tool_valid"] and not r["correct"], r

    # make_sft_trace is self-consistent (final scores correct)
    tr = make_sft_trace(q, aspirin, "mw")
    assert oracle.score("mw", parse_final(tr[-1]["content"]), aspirin)
    print("harness.py: all checks pass")


if __name__ == "__main__":
    demo()
