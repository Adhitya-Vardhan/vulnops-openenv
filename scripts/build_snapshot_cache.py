"""Build a provider-backed fallback snapshot cache."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.cases import EPSS_URL, NVD_CVE_URL, OSV_VULN_URL, _extract_cve_id

SNAPSHOT_DIR = ROOT / "data" / "snapshots"
INDEX_PATH = ROOT / "data" / "snapshot_index.json"
PYPA_TREE_URL = "https://api.github.com/repos/pypa/advisory-database/git/trees/main?recursive=1"


def get_candidate_ids(limit: int = 200) -> List[str]:
    response = requests.get(PYPA_TREE_URL, timeout=30)
    response.raise_for_status()
    tree = response.json().get("tree", [])
    ids = []
    for item in tree:
        path = item.get("path", "")
        if not path.startswith("vulns/") or not path.endswith(".yaml"):
            continue
        ident = path.rsplit("/", 1)[-1][:-5]
        if ident.startswith(("PYSEC-", "GHSA-")):
            ids.append(ident)
    return ids[: limit * 4]


def fetch_json(url: str, *, params: Dict[str, str] | None = None) -> Dict:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def build_snapshot(osv_id: str) -> Dict | None:
    osv = fetch_json(OSV_VULN_URL.format(osv_id=osv_id))
    if not osv.get("affected"):
        return None

    cve_id = _extract_cve_id(osv)
    snapshot = {
        "id": osv.get("id"),
        "summary": osv.get("summary"),
        "details": osv.get("details"),
        "aliases": osv.get("aliases", []),
        "references": osv.get("references", []),
        "affected": osv.get("affected", []),
        "severity": "MEDIUM",
        "nvd_description": "",
        "epss_score": 0.0,
        "epss_percentile": 0.0,
    }

    if cve_id:
        try:
            nvd = fetch_json(NVD_CVE_URL, params={"cveId": cve_id})
            vulnerability = (nvd.get("vulnerabilities") or [{}])[0].get("cve", {})
            metrics = vulnerability.get("metrics", {})
            severity = None
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
            snapshot["severity"] = severity or snapshot["severity"]
            snapshot["nvd_description"] = next(
                (
                    desc.get("value", "")
                    for desc in descriptions
                    if desc.get("lang") == "en"
                ),
                descriptions[0].get("value", "") if descriptions else "",
            )
        except Exception:
            pass

        try:
            epss = fetch_json(EPSS_URL, params={"cve": cve_id})
            item = (epss.get("data") or [{}])[0]
            snapshot["epss_score"] = float(item.get("epss", 0.0) or 0.0)
            snapshot["epss_percentile"] = float(item.get("percentile", 0.0) or 0.0)
        except Exception:
            pass

    return snapshot


def main(target_count: int = 200) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = get_candidate_ids(target_count)[: max(target_count + 40, 240)]
    saved = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(build_snapshot, osv_id): osv_id for osv_id in candidates}
        for future in as_completed(futures):
            if len(saved) >= target_count:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            osv_id = futures[future]
            try:
                snapshot = future.result()
            except Exception:
                continue
            if not snapshot:
                continue
            out_path = SNAPSHOT_DIR / f"{osv_id}.json"
            out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
            saved.append(
                {
                    "osv_id": osv_id,
                    "file": str(out_path.relative_to(ROOT)),
                    "cve_id": _extract_cve_id(snapshot),
                    "package": (snapshot.get("affected") or [{}])[0].get("package", {}).get("name", ""),
                }
            )

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = sorted(saved, key=lambda item: item["osv_id"])
    INDEX_PATH.write_text(json.dumps({"count": len(saved), "snapshots": saved}, indent=2))
    print(f"Saved {len(saved)} snapshots to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
