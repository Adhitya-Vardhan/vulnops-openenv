"""Deterministic graders for the vulnerability triage benchmark."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

try:
    from ..models import TriageDraft
    from .cases import CASE_DEFINITIONS, CaseDefinition, get_case_definition
except ImportError:
    from models import TriageDraft
    from server.cases import CASE_DEFINITIONS, CaseDefinition, get_case_definition


WEIGHTS: Dict[str, float] = {
    "validity": 0.20,
    "affected_package": 0.10,
    "affected_versions": 0.10,
    "severity": 0.20,
    "exploitability": 0.15,
    "next_action": 0.15,
    "missing_information": 0.10,
}

TERMINAL_SCORE_EPSILON = 0.0001


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def normalize_list(values: Iterable[str]) -> List[str]:
    return sorted({normalize_text(value) for value in values if normalize_text(value)})


def set_similarity(actual: Iterable[str], expected: Iterable[str]) -> float:
    actual_set = set(normalize_list(actual))
    expected_set = set(normalize_list(expected))
    if not actual_set and not expected_set:
        return 1.0
    if not actual_set or not expected_set:
        return 0.0
    union = actual_set | expected_set
    return len(actual_set & expected_set) / len(union)


def field_match(actual: str, expected: str) -> float:
    return 1.0 if normalize_text(actual) == normalize_text(expected) else 0.0


def _normalize_version_range(value: str) -> str:
    """Canonicalize a version range string for flexible comparison.

    Two representations that are treated as equivalent:
    - A trivial lower bound ``>=0`` / ``>=0.0`` / ``>=0.0.0`` followed by a
      comma is stripped, so ``>=0,<0.1.5`` compares equal to ``<0.1.5``.
    - Semicolon-separated multi-branch segments are sorted so submission
      order does not matter.
    """
    text = normalize_text(value)
    segments = [seg.strip() for seg in text.split(";") if seg.strip()]
    normalized: List[str] = []
    for seg in segments:
        # Remove trivial lower-bound prefix: >=0, >=0.0, >=0.0.0 before comma
        seg = re.sub(r">=\s*0(?:\.0)*\s*,\s*", "", seg)
        # Collapse whitespace around comparison operators
        seg = re.sub(r"\s*([><=!]+)\s*", r"\1", seg).strip()
        if seg:
            normalized.append(seg)
    return " ; ".join(sorted(normalized))


def version_range_match(actual: str, expected: str) -> float:
    """Score 1.0 when version ranges are semantically equivalent."""
    return (
        1.0
        if _normalize_version_range(actual) == _normalize_version_range(expected)
        else 0.0
    )


def normalize_terminal_score(score: float) -> float:
    """Clamp terminal task scores into the validator's required open interval."""
    rounded = round(min(max(float(score), 0.0), 1.0), 4)
    if rounded <= 0.0:
        return TERMINAL_SCORE_EPSILON
    if rounded >= 1.0:
        return round(1.0 - TERMINAL_SCORE_EPSILON, 4)
    return rounded


def grade_case(case: CaseDefinition, draft: TriageDraft) -> Dict[str, float]:
    breakdown = {
        "validity": field_match(draft.validity, case.truth.validity),
        "affected_package": field_match(
            draft.affected_package, case.truth.affected_package
        ),
        "affected_versions": version_range_match(
            draft.affected_versions, case.truth.affected_versions
        ),
        "severity": field_match(draft.severity, case.truth.severity),
        "exploitability": field_match(
            draft.exploitability, case.truth.exploitability
        ),
        "next_action": field_match(draft.next_action, case.truth.next_action),
        "missing_information": set_similarity(
            draft.missing_information, case.truth.missing_information
        ),
    }
    breakdown["total"] = round(
        sum(breakdown[name] * WEIGHTS[name] for name in WEIGHTS), 4
    )
    
    # Cascading Difficulty Penalties
    difficulty = getattr(case, 'difficulty', 'hard')
    
    # HARD Penalty: if truth action is request_info (meaning a patch is missing)
    # and the model tries to publish an advisory or patch, it's a critical logic failure.
    if difficulty == "hard" and case.truth.next_action == "request_info" and draft.next_action in {"publish_advisory", "patch"}:
        breakdown["total"] = round(breakdown["total"] * 0.5, 4)
        
    # MEDIUM Penalty: Medium tasks specifically test the ability to decode
    # raw intelligence (code diffs, conflicting EPSS/honeypot numbers) against decoys.
    # If a model fails to extract both severity and exploitability accurately, it failed the core challenge.
    elif difficulty == "medium" and breakdown["severity"] == 0.0 and breakdown["exploitability"] == 0.0:
        breakdown["total"] = round(breakdown["total"] * 0.75, 4)
        
    return breakdown


def grade_task(task_id: str, draft: TriageDraft) -> Dict[str, float]:
    return grade_case(get_case_definition(task_id), draft)
