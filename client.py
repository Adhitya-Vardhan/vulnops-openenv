"""Typed OpenEnv client for the vulnerability triage environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    EvidenceItem,
    TriageDraft,
    VulnTriageAction,
    VulnTriageObservation,
    VulnTriageState,
)


class VulnTriageEnv(
    EnvClient[VulnTriageAction, VulnTriageObservation, VulnTriageState]
):
    """Persistent typed client for the vulnerability triage benchmark."""

    def _step_payload(self, action: VulnTriageAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[VulnTriageObservation]:
        observation = VulnTriageObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> VulnTriageState:
        return VulnTriageState.model_validate(payload)
