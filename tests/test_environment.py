import random

from models import TriageDraft, VulnTriageAction
from server.cases import choose_balanced_task_id, CASE_DEFINITIONS
from server.graders import grade_task, version_range_match
from server.vuln_triage_env_environment import VulnTriageEnvironment


# ---------------------------------------------------------------------------
# Core environment tests
# ---------------------------------------------------------------------------

def test_easy_task_can_be_solved_deterministically():
    """Easy task should be solvable in 10 steps with just 2 evidence reads."""
    env = VulnTriageEnvironment()
    env.reset(task_id="task_easy_guarddog")
    env.step(VulnTriageAction(action_type="read_report"))
    env.step(VulnTriageAction(action_type="inspect_evidence", evidence_id="osv_advisory"))
    env.step(VulnTriageAction(action_type="inspect_evidence", evidence_id="affected_versions"))
    env.step(VulnTriageAction(action_type="set_validity", value="valid"))
    env.step(VulnTriageAction(action_type="set_affected_package", value="guarddog"))
    env.step(VulnTriageAction(action_type="set_affected_versions", value="<0.1.5"))
    env.step(VulnTriageAction(action_type="set_severity", value="medium"))
    env.step(VulnTriageAction(action_type="set_exploitability", value="low"))
    env.step(VulnTriageAction(action_type="set_next_action", value="patch"))
    result = env.step(VulnTriageAction(action_type="submit_triage"))
    assert result.done is True
    assert result.final_score == 1.0


def test_medium_task_uses_real_provider_backed_truth():
    env = VulnTriageEnvironment()
    env.reset(task_id="task_medium_invenio")
    env.step(VulnTriageAction(action_type="set_validity", value="valid"))
    env.step(VulnTriageAction(action_type="set_affected_package", value="invenio-records"))
    breakdown = grade_task("task_medium_invenio", env.state.draft)
    assert breakdown["validity"] == 1.0
    assert breakdown["affected_package"] == 1.0


def test_balanced_sampler_is_seed_reproducible():
    first = choose_balanced_task_id(7, random.Random(0))
    second = choose_balanced_task_id(7, random.Random(999))
    assert first == second


def test_environment_reset_without_task_id_samples_valid_difficulties():
    env = VulnTriageEnvironment()
    seen = {env.reset().difficulty for _ in range(12)}
    assert seen == {"easy", "medium", "hard"}


# ---------------------------------------------------------------------------
# Fix 1: version range normalizer accepts equivalent expressions
# ---------------------------------------------------------------------------

def test_version_range_match_accepts_trivial_lower_bound():
    assert version_range_match(">=0,<0.1.5", "<0.1.5") == 1.0
    assert version_range_match(">=0.0.0,<0.1.5", "<0.1.5") == 1.0


def test_version_range_match_is_order_insensitive_for_segments():
    a = "<1.0.2 ; >=1.1.0,<1.1.1 ; >=1.2.0,<1.2.2"
    b = ">=1.2.0,<1.2.2 ; >=1.1.0,<1.1.1 ; <1.0.2"
    assert version_range_match(a, b) == 1.0


def test_version_range_match_different_ranges_score_zero():
    assert version_range_match("<0.1.4", "<0.1.5") == 0.0


# ---------------------------------------------------------------------------
# Fix 2: multi-branch affected versions captured correctly
# ---------------------------------------------------------------------------

def test_medium_invenio_ground_truth_includes_all_branches():
    truth = CASE_DEFINITIONS["task_medium_invenio"].truth
    assert "<1.0.2" in truth.affected_versions
    assert ">=1.1.0,<1.1.1" in truth.affected_versions
    assert ">=1.2.0,<1.2.2" in truth.affected_versions


def test_medium_invenio_all_branches_score_full_points():
    draft = TriageDraft(
        validity="valid",
        affected_package="invenio-records",
        affected_versions=">=1.2.0,<1.2.2 ; >=1.1.0,<1.1.1 ; <1.0.2",
        severity="medium",
        exploitability="low",
        next_action="publish_advisory",
    )
    breakdown = grade_task("task_medium_invenio", draft)
    assert breakdown["affected_versions"] == 1.0


# ---------------------------------------------------------------------------
# Difficulty redesign — Easy task
# ---------------------------------------------------------------------------

def test_easy_task_only_needs_two_evidence_items():
    """Easy task supporting_evidence_ids should be just 2 items, not 4."""
    truth = CASE_DEFINITIONS["task_easy_guarddog"].truth
    assert truth.supporting_evidence_ids == ["osv_advisory", "affected_versions"]
    assert len(truth.supporting_evidence_ids) == 2


def test_easy_task_max_steps_is_tight():
    assert CASE_DEFINITIONS["task_easy_guarddog"].max_steps == 10


# ---------------------------------------------------------------------------
# Difficulty redesign — Medium task
# ---------------------------------------------------------------------------

def test_medium_task_has_threat_intel_evidence():
    """Medium task should inject a threat_intel_signal evidence item."""
    evidence_ids = [e["evidence_id"] for e in CASE_DEFINITIONS["task_medium_invenio"].evidence]
    assert "threat_intel_signal" in evidence_ids


def test_medium_task_exploitability_is_medium_not_low():
    """EPSS says low but threat intel overrides to medium — key difficulty driver."""
    truth = CASE_DEFINITIONS["task_medium_invenio"].truth
    assert truth.exploitability == "medium", (
        "Medium task exploitability must be 'medium' (overriding EPSS) "
        "so any model that only reads the EPSS evidence gets it wrong."
    )


def test_medium_task_exploitability_costs_points_if_epss_only():
    """A model that reads only EPSS and submits 'low' exploitability loses points."""
    draft = TriageDraft(
        validity="valid",
        affected_package="invenio-records",
        affected_versions="<1.0.2 ; >=1.1.0,<1.1.1 ; >=1.2.0,<1.2.2",
        severity="medium",
        exploitability="low",   # wrong — EPSS-only answer
        next_action="publish_advisory",
    )
    breakdown = grade_task("task_medium_invenio", draft)
    assert breakdown["exploitability"] == 0.0
    assert breakdown["total"] < 1.0


# ---------------------------------------------------------------------------
# Difficulty redesign — Hard task
# ---------------------------------------------------------------------------

def test_hard_task_correct_next_action_is_request_info():
    """Hard task must require request_info, not publish_advisory."""
    truth = CASE_DEFINITIONS["task_hard_gradio"].truth
    assert truth.next_action == "request_info", (
        "Hard task next_action must be 'request_info' — no patch exists yet."
    )


def test_hard_task_has_vendor_status_evidence():
    """Hard task should inject a vendor_status evidence item explaining no patch."""
    evidence_ids = [e["evidence_id"] for e in CASE_DEFINITIONS["task_hard_gradio"].evidence]
    assert "vendor_status" in evidence_ids


def test_hard_task_affected_versions_covers_all():
    """Hard task affected_versions must be >=0 (no fixed version)."""
    truth = CASE_DEFINITIONS["task_hard_gradio"].truth
    assert truth.affected_versions == ">=0"


def test_hard_task_publish_advisory_costs_next_action_points():
    """A model that naively publishes instead of requesting info loses 15%."""
    truth = CASE_DEFINITIONS["task_hard_gradio"].truth
    draft = TriageDraft(
        validity="valid",
        affected_package="gradio",
        affected_versions=">=0",
        severity="medium",
        exploitability="low",
        next_action="publish_advisory",   # wrong — no patch exists
        missing_information=list(truth.missing_information),
    )
    breakdown = grade_task("task_hard_gradio", draft)
    assert breakdown["next_action"] == 0.0
    assert breakdown["total"] < 1.0


def test_hard_task_request_info_scores_full():
    """The correct hard-task answer should score 1.0."""
    truth = CASE_DEFINITIONS["task_hard_gradio"].truth
    draft = TriageDraft(
        validity=truth.validity,
        affected_package=truth.affected_package,
        affected_versions=truth.affected_versions,
        severity=truth.severity,
        exploitability=truth.exploitability,
        next_action="request_info",
        missing_information=list(truth.missing_information),
    )
    breakdown = grade_task("task_hard_gradio", draft)
    assert breakdown["next_action"] == 1.0
    assert breakdown["total"] == 1.0


def test_hard_task_has_non_empty_missing_information():
    truth = CASE_DEFINITIONS["task_hard_gradio"].truth
    assert len(truth.missing_information) >= 3


def test_hard_task_empty_missing_info_costs_points():
    draft = TriageDraft(
        validity="valid",
        affected_package="gradio",
        affected_versions=">=0",
        severity="medium",
        exploitability="low",
        next_action="request_info",
        missing_information=[],
    )
    breakdown = grade_task("task_hard_gradio", draft)
    assert breakdown["missing_information"] == 0.0
    assert breakdown["total"] < 1.0
