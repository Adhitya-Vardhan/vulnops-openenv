"""Typed models for the vulnerability triage environment."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


ActionType = Literal[
    "read_report",
    "inspect_evidence",
    "search_nvd_database",
    "fetch_commit_diff",
    "message_maintainer",
    "set_validity",
    "set_affected_package",
    "set_affected_versions",
    "set_severity",
    "set_exploitability",
    "set_next_action",
    "set_missing_information",
    "request_more_info",
    "submit_triage",
]

ValidityLabel = Literal["unknown", "valid", "invalid", "needs_more_info"]
SeverityLabel = Literal["unknown", "low", "medium", "high", "critical"]
ExploitabilityLabel = Literal["unknown", "low", "medium", "high"]
NextActionLabel = Literal[
    "unknown",
    "request_info",
    "close",
    "escalate",
    "patch",
    "publish_advisory",
]


class EvidenceItem(BaseModel):
    """Evidence the agent can reveal during triage."""

    evidence_id: str = Field(..., description="Unique identifier for this evidence item")
    title: str = Field(..., description="Short evidence title")
    summary: str = Field(..., description="Evidence content shown to the agent")
    kind: str = Field(..., description="Evidence type such as advisory or patch note")


class TriageDraft(BaseModel):
    """Agent-managed triage state."""

    validity: ValidityLabel = "unknown"
    affected_package: str = ""
    affected_versions: str = ""
    severity: SeverityLabel = "unknown"
    exploitability: ExploitabilityLabel = "unknown"
    next_action: NextActionLabel = "unknown"
    missing_information: List[str] = Field(default_factory=list)


class VulnTriageAction(Action):
    """Structured action space for vulnerability triage."""

    action_type: ActionType = Field(..., description="Which environment action to execute")
    evidence_id: Optional[str] = Field(
        default=None,
        description="Evidence identifier used by inspect_evidence",
    )
    value: Optional[str] = Field(
        default=None,
        description="Generic value used for label-setting actions",
    )
    rationale: str = Field(
        default="",
        description="Optional short rationale for debugging and trajectory inspection",
    )


class VulnTriageObservation(Observation):
    """Observation returned after every environment transition."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: str = Field(..., description="Difficulty band for the current task")
    objective: str = Field(..., description="Concrete task objective")
    report_summary: str = Field(..., description="Incoming vulnerability report summary")
    visible_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items currently visible to the agent",
    )
    available_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence identifiers available to inspect next",
    )
    draft: TriageDraft = Field(
        default_factory=TriageDraft,
        description="Current structured triage draft",
    )
    action_history: List[str] = Field(
        default_factory=list,
        description="Compact history of recent agent actions",
    )
    steps_remaining: int = Field(..., ge=0, description="Remaining steps in the episode")
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Current normalized grader breakdown",
    )
    final_score: Optional[float] = Field(
        default=None,
        description="Final submission score when the episode is done",
    )
    available_actions: List[str] = Field(
        default_factory=lambda: [
            "read_report",
            "inspect_evidence",
            "search_nvd_database",
            "fetch_commit_diff",
            "message_maintainer",
            "set_validity",
            "set_affected_package",
            "set_affected_versions",
            "set_severity",
            "set_exploitability",
            "set_next_action",
            "set_missing_information",
            "request_more_info",
            "submit_triage",
        ],
        description="Action names the agent can choose from",
    )


class VulnTriageState(State):
    """Serializable environment state for inspection and debugging."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: str = Field(..., description="Difficulty band")
    draft: TriageDraft = Field(default_factory=TriageDraft)
    revealed_evidence_ids: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    steps_remaining: int = Field(..., ge=0)
    submitted: bool = Field(default=False)
    final_score: Optional[float] = Field(default=None)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
