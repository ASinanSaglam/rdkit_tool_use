"""Re-exports rdkit_qa.oracle. Ground-truth RDKit computation is protocol-
agnostic (v1's plain-text TOOL_CALL: vs v2's native tool-calling don't change
what "correct" means), so v2 shares the exact same single source of truth
instead of duplicating it."""
from rdkit_qa.oracle import PROPERTIES, QUESTION_TEMPLATES, compute, score

__all__ = ["PROPERTIES", "QUESTION_TEMPLATES", "compute", "score"]
