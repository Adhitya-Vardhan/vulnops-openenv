"""OpenEnv environment implementation for vulnerability triage."""

from __future__ import annotations

import random
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import EvidenceItem, TriageDraft, VulnTriageAction, VulnTriageObservation, VulnTriageState
    from .cases import CASE_DEFINITIONS, SEEDS, TASK_ORDER, CaseDefinition, choose_balanced_task_id, get_case_definition
    from .graders import grade_case, normalize_terminal_score, normalize_text
except ImportError:
    from models import EvidenceItem, TriageDraft, VulnTriageAction, VulnTriageObservation, VulnTriageState
    from server.cases import CASE_DEFINITIONS, SEEDS, TASK_ORDER, CaseDefinition, choose_balanced_task_id, get_case_definition
    from server.graders import grade_case, normalize_terminal_score, normalize_text


FIELD_TO_ATTR = {
    "set_validity": "validity",
    "set_affected_package": "affected_package",
    "set_affected_versions": "affected_versions",
    "set_severity": "severity",
    "set_exploitability": "exploitability",
    "set_next_action": "next_action",
}


class VulnTriageEnvironment(Environment):
    """Deterministic multi-step environment for OSS vulnerability triage."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._case: CaseDefinition = CASE_DEFINITIONS[TASK_ORDER[0]]
        self._rng = random.Random(0)
        self._revealed_evidence_ids: List[str] = []
        self._draft = TriageDraft()
        self._action_history: List[str] = []
        self._submitted = False
        self._score_breakdown: Dict[str, float] = {}
        self._state = VulnTriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._case.task_id,
            difficulty=self._case.difficulty,
            draft=self._draft,
            revealed_evidence_ids=[],
            action_history=[],
            steps_remaining=self._case.max_steps,
            submitted=False,
            final_score=None,
            score_breakdown={},
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> VulnTriageObservation:
        if task_id:
            self._case = get_case_definition(task_id)
        else:
            selected_task_id = choose_balanced_task_id(seed, self._rng)
            self._case = get_case_definition(selected_task_id)

        self._revealed_evidence_ids = []
        self._draft = TriageDraft()
        self._action_history = []
        self._submitted = False
        self._score_breakdown = grade_case(self._case, self._draft)
        self._state = VulnTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._case.task_id,
            difficulty=self._case.difficulty,
            draft=self._draft.model_copy(deep=True),
            revealed_evidence_ids=[],
            action_history=[],
            steps_remaining=self._case.max_steps,
            submitted=False,
            final_score=None,
            score_breakdown=self._score_breakdown,
        )
        return self._observation(reward=0.0)

    def step(
        self,
        action: VulnTriageAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> VulnTriageObservation:
        del timeout_s
        if self._submitted:
            return self._observation(
                reward=-0.05,
                done=True,
                metadata={"error": "episode_already_submitted"},
            )

        self._state.step_count += 1
        reward = -0.005
        note = action.action_type

        if action.action_type == "read_report":
            reward += 0.03 if not any(h.startswith("read_report") for h in self._action_history) else -0.02
            note = "read_report"
        elif action.action_type == "search_nvd_database":
            reward += self._handle_nvd_search(action)
            note = f"search_nvd_database:{action.value or ''}"
        elif action.action_type == "fetch_commit_diff":
            reward += self._handle_commit_fetch(action)
            note = f"fetch_commit_diff:{action.value or ''}"
        elif action.action_type == "message_maintainer":
            reward += self._handle_message_maintainer(action)
            note = f"message_maintainer:{action.value or ''}"
        elif action.action_type == "inspect_evidence":
            reward += self._handle_inspect(action)
            note = f"inspect_evidence:{action.evidence_id or ''}"
        elif action.action_type in FIELD_TO_ATTR:
            reward += self._handle_field_update(action)
            note = f"{action.action_type}:{action.value or ''}"
        elif action.action_type in {"set_missing_information", "request_more_info"}:
            reward += self._handle_missing_info(action)
            note = f"{action.action_type}:{action.value or ''}"
        elif action.action_type == "submit_triage":
            return self._handle_submit(action)
        else:
            reward -= 0.05
            note = f"invalid_action:{action.action_type}"

        self._action_history.append(note)
        self._score_breakdown = grade_case(self._case, self._draft)
        self._sync_state()

        if self._state.steps_remaining == 0:
            timeout_penalty = normalize_terminal_score(
                max(self._score_breakdown["total"] - 0.1, 0.0)
            )
            self._submitted = True
            self._state.submitted = True
            self._score_breakdown = {**self._score_breakdown, "total": timeout_penalty}
            self._state.final_score = timeout_penalty
            return self._observation(
                reward=timeout_penalty,
                done=True,
                final_score=timeout_penalty,
                metadata={"termination_reason": "step_budget_exhausted"},
            )

        return self._observation(reward=round(reward, 4))

    def _handle_nvd_search(self, action: VulnTriageAction) -> float:
        query = (action.value or "").strip().lower()
        if not query:
            return -0.05
        # The query should match one of the aliases in the seed fallback to return the nvd_assessment
        seed = SEEDS[self._case.task_id]
        snapshot_aliases = [normalize_text(a) for a in seed.fallback_snapshot.get("aliases", [])]
        
        # We assume nvd_assessment handles the real data. If they searched a decoy CVE, we should
        # conceptually return the decoy data. For simplicity, we just check if it matches the real CVE.
        if normalize_text(query) in snapshot_aliases or query == normalize_text(seed.osv_id):
            if "nvd_assessment" not in self._revealed_evidence_ids:
                self._revealed_evidence_ids.append("nvd_assessment")
            return 0.08
        return -0.04

    def _handle_commit_fetch(self, action: VulnTriageAction) -> float:
        query = (action.value or "").strip()
        if not query:
            return -0.05
        # If there's a github_commit_diff evidence piece, we check if the query is in the title "GitHub Commit <hash>"
        for item in self._case.evidence:
            if item["evidence_id"] == "github_commit_diff":
                if query.lower() in item["title"].lower():
                    if "github_commit_diff" not in self._revealed_evidence_ids:
                        self._revealed_evidence_ids.append("github_commit_diff")
                    return 0.08
        return -0.04

    def _handle_message_maintainer(self, action: VulnTriageAction) -> float:
        msg = (action.value or "").strip()
        if len(msg) < 5:
            return -0.05 # Need a real message
        
        # When sending a message to maintainer, we return the vendor_status evidence if it exists
        has_vendor_evidence = False
        for item in self._case.evidence:
            if item["evidence_id"] == "vendor_status":
                if "vendor_status" not in self._revealed_evidence_ids:
                    self._revealed_evidence_ids.append("vendor_status")
                has_vendor_evidence = True
                break
        
        return 0.08 if has_vendor_evidence else -0.02

    def _handle_inspect(self, action: VulnTriageAction) -> float:
        evidence_id = action.evidence_id or ""
        all_ids = {item["evidence_id"] for item in self._case.evidence}
        if evidence_id not in all_ids:
            return -0.06
            
        # Trap: Model cannot inspect interactive evidence directly as if it was static JSON
        if evidence_id in {"nvd_assessment", "github_commit_diff", "vendor_status"}:
            return -0.05
            
        if evidence_id in self._revealed_evidence_ids:
            return -0.02

        self._revealed_evidence_ids.append(evidence_id)
        if evidence_id in self._case.truth.supporting_evidence_ids:
            return 0.06
        return 0.02

    def _handle_field_update(self, action: VulnTriageAction) -> float:
        attr = FIELD_TO_ATTR[action.action_type]
        new_value = (action.value or "").strip()
        if not new_value:
            return -0.04

        current_value = getattr(self._draft, attr)
        if normalize_text(current_value) == normalize_text(new_value):
            return -0.01

        setattr(self._draft, attr, new_value)
        expected_value = getattr(self._case.truth, attr)
        if normalize_text(new_value) == normalize_text(expected_value):
            return 0.08
        return -0.03

    def _handle_missing_info(self, action: VulnTriageAction) -> float:
        value = (action.value or "").strip()
        if not value:
            return -0.04

        normalized_existing = {normalize_text(item) for item in self._draft.missing_information}
        if normalize_text(value) not in normalized_existing:
            self._draft.missing_information.append(value)

        required = {normalize_text(item) for item in self._case.truth.missing_information}
        if normalize_text(value) in required:
            return 0.06
        if action.action_type == "request_more_info" and self._case.truth.next_action == "request_info":
            return 0.02
        return -0.02

    def _handle_submit(self, action: VulnTriageAction) -> VulnTriageObservation:
        del action
        self._submitted = True
        breakdown = grade_case(self._case, self._draft)
        final_score = breakdown["total"]
        if len(self._revealed_evidence_ids) < max(2, len(self._case.truth.supporting_evidence_ids) // 2):
            final_score = max(0.0, round(final_score - 0.1, 4))
        final_score = normalize_terminal_score(final_score)

        self._action_history.append("submit_triage")
        self._score_breakdown = {**breakdown, "total": final_score}
        self._state.submitted = True
        self._state.final_score = final_score
        self._sync_state()
        return self._observation(
            reward=final_score,
            done=True,
            final_score=final_score,
            metadata={"termination_reason": "submitted"},
        )

    def _sync_state(self) -> None:
        self._state.task_id = self._case.task_id
        self._state.difficulty = self._case.difficulty
        self._state.draft = self._draft.model_copy(deep=True)
        self._state.revealed_evidence_ids = list(self._revealed_evidence_ids)
        self._state.action_history = list(self._action_history)
        self._state.steps_remaining = max(self._case.max_steps - self._state.step_count, 0)
        self._state.score_breakdown = dict(self._score_breakdown)

    def _observation(
        self,
        reward: float,
        done: bool = False,
        final_score: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> VulnTriageObservation:
        self._sync_state()
        visible_evidence = [
            EvidenceItem.model_validate(item)
            for item in self._case.evidence
            if item["evidence_id"] in self._revealed_evidence_ids
        ]
        return VulnTriageObservation(
            task_id=self._case.task_id,
            difficulty=self._case.difficulty,
            objective=self._case.objective,
            report_summary=self._case.report_summary,
            visible_evidence=visible_evidence,
            available_evidence=[
                item["evidence_id"]
                for item in self._case.evidence
                if item["evidence_id"] not in self._revealed_evidence_ids
            ],
            draft=self._draft.model_copy(deep=True),
            action_history=list(self._action_history),
            steps_remaining=max(self._case.max_steps - self._state.step_count, 0),
            score_breakdown=dict(self._score_breakdown),
            final_score=final_score,
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    @property
    def state(self) -> VulnTriageState:
        self._sync_state()
        return self._state
