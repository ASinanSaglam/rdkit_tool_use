"""Agentic tool-call loop + verifiable scoring -- v2, native tool-calling.

Unlike v1's plain-text `TOOL_CALL:`/`TOOL_RESULT:`/`FINAL:` protocol, this
uses the model's real chat-template tool-calling support (verified against
Qwen2.5's actual chat_template.jinja: tools declared via the `tools=` kwarg,
assistant tool calls as a `tool_calls` message field rendering to
`<tool_call>{"name":...,"arguments":...}</tool_call>`, results as a `role:
tool` message rendering to `<tool_response>...</tool_response>`). No FINAL:
marker: a final answer is just an assistant message with no tool_calls --
that structural signal already exists in the native format.

Supports zero, one, or many tool calls per episode (loop until the model
stops calling), so the same harness covers all five training behaviors:
single-property lookup, multi-property composition, tool-restraint
(no-tool-needed), clarify (missing required info), and honest tool-error
reporting."""
from __future__ import annotations
import json, re
from . import oracle

MAX_TURNS = 6

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rdkit_compute",
        "description": "Compute a molecular property for a molecule using RDKit.",
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string of the molecule"},
                "property": {"type": "string", "enum": list(oracle.PROPERTIES),
                             "description": "Which property to compute"},
            },
            "required": ["smiles", "property"],
        },
    },
}

SYSTEM = (
    "You answer questions, calling the rdkit_compute tool when you need a "
    "molecular property computed. Call it only when needed -- if the "
    "question doesn't require computing a molecular property, answer "
    "directly without calling the tool. If a question needs a molecule but "
    "no SMILES string was given, ask for it instead of guessing. If a tool "
    "call fails, report the failure honestly instead of making up a value."
)

_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def build_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]


def parse_tool_call(text: str):
    """Return {"smiles":..., "property":...}, or None if this turn isn't a
    (valid) tool call -- callers treat that as the episode's final answer."""
    m = _CALL_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or d.get("name") != "rdkit_compute":
        return None
    args = d.get("arguments")
    if not isinstance(args, dict) or "smiles" not in args:
        return None
    prop = str(args.get("property", "")).strip().lower()
    if prop not in oracle.PROPERTIES:
        return None
    return {"smiles": str(args["smiles"]), "property": prop}


def run_tool(call: dict) -> dict:
    """Execute the oracle. Explicit error key on failure (invalid SMILES)
    instead of a bare `None` -- v1 silently returned `TOOL_RESULT: None`,
    which no training example ever taught the model to react to honestly."""
    val = oracle.compute(call["smiles"], call["property"])
    if val is None:
        return {"error": "invalid molecule"}
    return {"result": val}


def tool_call_message(call: dict) -> dict:
    return {"role": "assistant",
            "tool_calls": [{"function": {"name": "rdkit_compute", "arguments": call}}]}


def tool_result_message(call: dict) -> dict:
    return {"role": "tool", "content": json.dumps(run_tool(call))}


CLARIFY_KEYWORDS = ("smiles", "which molecule", "provide", "specify", "what molecule")
ERROR_KEYWORDS = ("invalid", "error", "couldn't", "cannot", "can't", "unable")


def _contains_any(text: str, keywords) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def run_episode(generate, question: str, expected: list[tuple[str, str]] | None = None,
                expects_tool: bool = True, kind: str = "single") -> dict:
    """General episode runner covering all five trained behaviors.

    expected: list of (smiles, property) pairs the final answer must cover
              (checked via oracle.score). Empty for no_tool/clarify/error
              kinds, where correctness is judged by content instead.
    kind: "single"/"multi" -> expected-list scoring (exact verifiable match).
          "no_tool" -> expects_tool=False, no content check beyond "answered
              without calling the tool".
          "clarify" -> expects_tool=False, final must ask for the missing
              SMILES (keyword check).
          "error" -> expects_tool=True (a call IS made, and correctly), but
              the tool result is an error, so the final is graded on
              honestly reporting failure (keyword check), not a value match.
    """
    expected = expected or []
    msgs = build_prompt(question)
    made_calls = []
    final = None
    for _ in range(MAX_TURNS):
        turn = generate(msgs)
        call = parse_tool_call(turn)
        if call is not None:
            made_calls.append(call)
            msgs = msgs + [tool_call_message(call), tool_result_message(call)]
            continue
        final = turn.strip()
        break

    if expected:
        tool_valid = all(any(c["smiles"] == s and c["property"] == p for c in made_calls)
                          for s, p in expected)
    else:
        tool_valid = len(made_calls) == 0

    if kind in ("no_tool", "clarify"):
        no_call = len(made_calls) == 0
        if kind == "no_tool":
            correct = final is not None and no_call
        else:
            correct = final is not None and no_call and _contains_any(final, CLARIFY_KEYWORDS)
        tool_valid = no_call
    elif kind == "error":
        tool_valid = len(made_calls) >= 1
        correct = final is not None and tool_valid and _contains_any(final, ERROR_KEYWORDS)
    elif final is None:
        correct = False
    else:
        correct = all(_final_covers(final, s, p) for s, p in expected)

    return {"tool_calls": made_calls, "final": final, "correct": correct, "tool_valid": tool_valid}


def _final_covers(final_text: str, smiles: str, prop: str) -> bool:
    if oracle.score(prop, final_text, smiles):
        return True
    return any(oracle.score(prop, tok, smiles)
               for tok in re.split(r"[\s,;:=]+", final_text) if tok)


def make_sft_trace(question: str, kind: str, **kw) -> list[dict]:
    """Gold native-format trace for SFT, dispatched by kind:
    single(smiles, prop) / multi(pairs=[(smiles,prop),...]) /
    no_tool(answer) / clarify(clarify_text) / error(smiles, prop)."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    if kind == "single":
        call = {"smiles": kw["smiles"], "property": kw["prop"]}
        result = oracle.compute(kw["smiles"], kw["prop"])
        msgs += [tool_call_message(call), tool_result_message(call),
                 {"role": "assistant", "content": str(result)}]
    elif kind == "multi":
        parts = []
        for smiles, prop in kw["pairs"]:
            call = {"smiles": smiles, "property": prop}
            msgs += [tool_call_message(call), tool_result_message(call)]
            parts.append(f"{prop}: {oracle.compute(smiles, prop)}")
        msgs.append({"role": "assistant", "content": ", ".join(parts)})
    elif kind == "no_tool":
        msgs.append({"role": "assistant", "content": kw["answer"]})
    elif kind == "clarify":
        msgs.append({"role": "assistant", "content": kw["clarify_text"]})
    elif kind == "error":
        call = {"smiles": kw["smiles"], "property": kw["prop"]}
        msgs += [tool_call_message(call), tool_result_message(call),
                 {"role": "assistant",
                  "content": "I couldn't compute this -- the SMILES string appears to be invalid."}]
    else:
        raise ValueError(f"unknown kind: {kind}")
    return msgs


def demo():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    q = oracle.QUESTION_TEMPLATES["mw"].format(smiles=aspirin)

    def good(messages):
        last = messages[-1]
        if last["role"] == "tool":
            return str(json.loads(last["content"])["result"])
        return '<tool_call>\n{"name": "rdkit_compute", "arguments": {"smiles": "%s", "property": "mw"}}\n</tool_call>' % aspirin

    r = run_episode(good, q, [(aspirin, "mw")])
    assert r["tool_valid"] and r["correct"], r

    # no-tool-needed: correctly answering directly
    r = run_episode(lambda m: "19", "What is 12 plus 7?", [], expects_tool=False, kind="no_tool")
    assert r["tool_valid"] and r["correct"], r

    # no-tool-needed: calling the tool anyway is a restraint failure
    r = run_episode(good, q, [], expects_tool=False, kind="no_tool")
    assert not r["tool_valid"] and not r["correct"], r

    # clarify: asks for the missing SMILES instead of guessing
    r = run_episode(lambda m: "Which molecule? Please provide a SMILES string.",
                    "What is its molecular weight?", [], expects_tool=False, kind="clarify")
    assert r["tool_valid"] and r["correct"], r

    # error: calls the tool (correctly), tool fails, honest report expected
    def honest_on_error(messages):
        last = messages[-1]
        if last["role"] == "tool":
            return "That SMILES is invalid, I can't compute this."
        return '<tool_call>\n{"name": "rdkit_compute", "arguments": {"smiles": "not_a_smiles", "property": "mw"}}\n</tool_call>'
    r = run_episode(honest_on_error, "What is the MW of not_a_smiles?", [], kind="error")
    assert r["tool_valid"] and r["correct"], r

    # multi-call: two chained calls then one combined final
    def multi_caller(messages):
        calls = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
        if len(calls) == 0:
            return '<tool_call>\n{"name": "rdkit_compute", "arguments": {"smiles": "%s", "property": "mw"}}\n</tool_call>' % aspirin
        if len(calls) == 1:
            return '<tool_call>\n{"name": "rdkit_compute", "arguments": {"smiles": "%s", "property": "logp"}}\n</tool_call>' % aspirin
        mw = oracle.compute(aspirin, "mw")
        logp = oracle.compute(aspirin, "logp")
        return f"mw={mw}, logp={logp}"

    r = run_episode(multi_caller, "mw and logp?", [(aspirin, "mw"), (aspirin, "logp")])
    assert r["tool_valid"] and r["correct"] and len(r["tool_calls"]) == 2, r

    # make_sft_trace is self-consistent for every kind
    tr = make_sft_trace(q, "single", smiles=aspirin, prop="mw")
    assert oracle.score("mw", tr[-1]["content"], aspirin)
    tr = make_sft_trace("mw and logp?", "multi", pairs=[(aspirin, "mw"), (aspirin, "logp")])
    assert _final_covers(tr[-1]["content"], aspirin, "mw")
    assert _final_covers(tr[-1]["content"], aspirin, "logp")

    print("harness.py (v2): all checks pass")


if __name__ == "__main__":
    demo()
