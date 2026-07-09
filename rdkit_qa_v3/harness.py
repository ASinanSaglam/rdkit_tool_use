"""Agentic tool-call loop + verifiable scoring -- v3, two tools.

Superset of v2's harness.py: same native tool-calling protocol
(<tool_call>{"name":...,"arguments":...}</tool_call> / <tool_response>...),
same MAX_TURNS loop, same five v2 behaviors -- but a call can now name either
of two tools, so parse_tool_call/run_tool/tool_call_message all carry an
explicit "tool" field instead of v2's implicit single-tool assumption. Adds a
sixth behavior, "chain": resolve a common name via name_to_smiles, then feed
the result into rdkit_compute.

run_tool takes a `name_resolver` callable so the SAME harness serves both
training/eval (resolver = name_dict.resolve_dict, offline/deterministic) and
live chat (resolver = name_dict_pubchem.resolve_pubchem, network) -- the
harness itself never hardcodes which backend is in use."""
from __future__ import annotations
import json, re
from . import oracle, name_dict

MAX_TURNS = 6

RDKIT_SCHEMA = {
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

NAME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "name_to_smiles",
        "description": "Look up the SMILES string for a molecule given its common "
                        "or trade name (e.g. 'acetone', 'aspirin').",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Common/trade name of the molecule"},
            },
            "required": ["name"],
        },
    },
}

TOOL_SCHEMAS = [RDKIT_SCHEMA, NAME_SCHEMA]

SYSTEM = (
    "You answer questions, calling tools when you need them: rdkit_compute for "
    "molecular properties (requires a SMILES string), name_to_smiles to look up "
    "the SMILES for a molecule given its common name. If a question gives a "
    "common name instead of a SMILES string, call name_to_smiles first and use "
    "its result as the input to rdkit_compute. Call tools only when needed -- if "
    "the question doesn't require computing a molecular property, answer "
    "directly without calling a tool. If a question needs a molecule but "
    "neither a SMILES string nor a resolvable name was given, ask for it "
    "instead of guessing. If a tool call fails, report the failure honestly "
    "instead of making up a value."
)

_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def build_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]


def parse_tool_call(text: str):
    """Return {"tool": "rdkit_compute"|"name_to_smiles", "arguments": {...}},
    or None if this turn isn't a (valid) tool call -- callers treat that as
    the episode's final answer."""
    m = _CALL_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    name = d.get("name")
    args = d.get("arguments")
    if not isinstance(args, dict):
        return None
    if name == "rdkit_compute":
        if "smiles" not in args:
            return None
        prop = str(args.get("property", "")).strip().lower()
        if prop not in oracle.PROPERTIES:
            return None
        return {"tool": "rdkit_compute",
                "arguments": {"smiles": str(args["smiles"]), "property": prop}}
    if name == "name_to_smiles":
        if "name" not in args:
            return None
        return {"tool": "name_to_smiles", "arguments": {"name": str(args["name"])}}
    return None


def run_tool(call: dict, name_resolver=name_dict.resolve_dict) -> dict:
    """Execute the named tool. Explicit error key on failure (invalid SMILES /
    unresolvable name) instead of a bare `None`."""
    if call["tool"] == "rdkit_compute":
        val = oracle.compute(call["arguments"]["smiles"], call["arguments"]["property"])
        return {"error": "invalid molecule"} if val is None else {"result": val}
    if call["tool"] == "name_to_smiles":
        smi = name_resolver(call["arguments"]["name"])
        return {"error": "unknown name"} if smi is None else {"result": smi}
    raise ValueError(f"unknown tool: {call['tool']}")


def tool_call_message(call: dict) -> dict:
    return {"role": "assistant",
            "tool_calls": [{"function": {"name": call["tool"], "arguments": call["arguments"]}}]}


def tool_result_message(call: dict, name_resolver=name_dict.resolve_dict) -> dict:
    return {"role": "tool", "content": json.dumps(run_tool(call, name_resolver))}


CLARIFY_KEYWORDS = ("smiles", "which molecule", "provide", "specify", "what molecule", "name")
ERROR_KEYWORDS = ("invalid", "error", "couldn't", "cannot", "can't", "unable", "unknown")


def _contains_any(text: str, keywords) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def run_episode(generate, question: str, expected: list[tuple[str, str]] | None = None,
                expects_tool: bool = True, kind: str = "single", name: str | None = None,
                name_resolver=name_dict.resolve_dict) -> dict:
    """General episode runner covering all six trained behaviors.

    expected: list of (smiles, property) pairs the final answer must cover
              (checked via oracle.score). Empty for no_tool/clarify/error
              kinds, where correctness is judged by content instead.
    name: for kind="chain" only -- the common name a correct name_to_smiles
          call must have been made with.
    kind: "single"/"multi" -> expected-list scoring (exact verifiable match).
          "chain" -> like single, but ALSO requires a correct name_to_smiles
              call (for `name`) before the rdkit_compute call it feeds.
          "no_tool" -> expects_tool=False, no content check beyond "answered
              without calling a tool".
          "clarify" -> expects_tool=False, final must ask for the missing
              SMILES/name (keyword check).
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
            msgs = msgs + [tool_call_message(call), tool_result_message(call, name_resolver)]
            continue
        final = turn.strip()
        break

    rdkit_calls = [c for c in made_calls if c["tool"] == "rdkit_compute"]
    name_calls = [c for c in made_calls if c["tool"] == "name_to_smiles"]

    if expected:
        tool_valid = all(any(c["arguments"]["smiles"] == s and c["arguments"]["property"] == p
                              for c in rdkit_calls) for s, p in expected)
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
        tool_valid = len(rdkit_calls) >= 1
        correct = final is not None and tool_valid and _contains_any(final, ERROR_KEYWORDS)
    elif kind == "chain":
        name_ok = name is not None and any(
            c["arguments"]["name"].strip().lower() == name.strip().lower() for c in name_calls)
        tool_valid = tool_valid and name_ok
        correct = final is not None and tool_valid and all(
            _final_covers(final, s, p) for s, p in expected)
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
    single(smiles, prop) / multi(pairs=[(smiles,prop),...]) / chain(name, prop)
    / no_tool(answer) / clarify(clarify_text) / error(smiles, prop)."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]

    if kind == "single":
        call = {"tool": "rdkit_compute", "arguments": {"smiles": kw["smiles"], "property": kw["prop"]}}
        result = oracle.compute(kw["smiles"], kw["prop"])
        msgs += [tool_call_message(call), tool_result_message(call),
                 {"role": "assistant", "content": str(result)}]
    elif kind == "multi":
        parts = []
        for smiles, prop in kw["pairs"]:
            call = {"tool": "rdkit_compute", "arguments": {"smiles": smiles, "property": prop}}
            msgs += [tool_call_message(call), tool_result_message(call)]
            parts.append(f"{prop}: {oracle.compute(smiles, prop)}")
        msgs.append({"role": "assistant", "content": ", ".join(parts)})
    elif kind == "chain":
        smi = oracle.compute_name(kw["name"])
        call1 = {"tool": "name_to_smiles", "arguments": {"name": kw["name"]}}
        call2 = {"tool": "rdkit_compute", "arguments": {"smiles": smi, "property": kw["prop"]}}
        result = oracle.compute(smi, kw["prop"])
        msgs += [tool_call_message(call1), tool_result_message(call1),
                 tool_call_message(call2), tool_result_message(call2),
                 {"role": "assistant", "content": str(result)}]
    elif kind == "no_tool":
        msgs.append({"role": "assistant", "content": kw["answer"]})
    elif kind == "clarify":
        msgs.append({"role": "assistant", "content": kw["clarify_text"]})
    elif kind == "error":
        call = {"tool": "rdkit_compute", "arguments": {"smiles": kw["smiles"], "property": kw["prop"]}}
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
        return ('<tool_call>\n{"name": "rdkit_compute", "arguments": '
                '{"smiles": "%s", "property": "mw"}}\n</tool_call>' % aspirin)

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
        return ('<tool_call>\n{"name": "rdkit_compute", "arguments": '
                '{"smiles": "not_a_smiles", "property": "mw"}}\n</tool_call>')
    r = run_episode(honest_on_error, "What is the MW of not_a_smiles?", [], kind="error")
    assert r["tool_valid"] and r["correct"], r

    # multi-call: two chained rdkit_compute calls then one combined final
    def multi_caller(messages):
        calls = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
        if len(calls) == 0:
            return ('<tool_call>\n{"name": "rdkit_compute", "arguments": '
                     '{"smiles": "%s", "property": "mw"}}\n</tool_call>' % aspirin)
        if len(calls) == 1:
            return ('<tool_call>\n{"name": "rdkit_compute", "arguments": '
                     '{"smiles": "%s", "property": "logp"}}\n</tool_call>' % aspirin)
        mw = oracle.compute(aspirin, "mw")
        logp = oracle.compute(aspirin, "logp")
        return f"mw={mw}, logp={logp}"

    r = run_episode(multi_caller, "mw and logp?", [(aspirin, "mw"), (aspirin, "logp")])
    assert r["tool_valid"] and r["correct"] and len(r["tool_calls"]) == 2, r

    # chain: name_to_smiles then rdkit_compute
    acetone_smi = oracle.compute_name("acetone")

    def chain_caller(messages):
        calls = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
        if len(calls) == 0:
            return ('<tool_call>\n{"name": "name_to_smiles", '
                     '"arguments": {"name": "acetone"}}\n</tool_call>')
        if len(calls) == 1:
            return ('<tool_call>\n{"name": "rdkit_compute", "arguments": '
                     '{"smiles": "%s", "property": "mw"}}\n</tool_call>' % acetone_smi)
        return str(oracle.compute(acetone_smi, "mw"))

    r = run_episode(chain_caller, "What is the molecular weight of acetone?",
                     [(acetone_smi, "mw")], kind="chain", name="acetone")
    assert r["tool_valid"] and r["correct"], r

    # chain: skipping name_to_smiles and guessing a SMILES is a restraint/correctness failure
    def skip_lookup(messages):
        calls = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
        if len(calls) == 0:
            return ('<tool_call>\n{"name": "rdkit_compute", "arguments": '
                     '{"smiles": "%s", "property": "mw"}}\n</tool_call>' % acetone_smi)
        return str(oracle.compute(acetone_smi, "mw"))

    r = run_episode(skip_lookup, "What is the molecular weight of acetone?",
                     [(acetone_smi, "mw")], kind="chain", name="acetone")
    assert not r["tool_valid"] and not r["correct"], r

    # make_sft_trace is self-consistent for every kind
    tr = make_sft_trace(q, "single", smiles=aspirin, prop="mw")
    assert oracle.score("mw", tr[-1]["content"], aspirin)
    tr = make_sft_trace("mw and logp?", "multi", pairs=[(aspirin, "mw"), (aspirin, "logp")])
    assert _final_covers(tr[-1]["content"], aspirin, "mw")
    assert _final_covers(tr[-1]["content"], aspirin, "logp")
    tr = make_sft_trace("What is the molecular weight of acetone?", "chain", name="acetone", prop="mw")
    assert _final_covers(tr[-1]["content"], acetone_smi, "mw")
    assert tr[2]["tool_calls"][0]["function"]["name"] == "name_to_smiles"
    assert tr[4]["tool_calls"][0]["function"]["name"] == "rdkit_compute"

    print("harness.py (v3): all checks pass")


if __name__ == "__main__":
    demo()
