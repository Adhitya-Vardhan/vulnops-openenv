"""Baseline inference script for the vulnerability triage environment."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from openai import OpenAI
from openenv.core import GenericEnvClient

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Support all key variants the validator may inject
_API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

from models import VulnTriageAction
from server.cases import TASK_ORDER, get_case_definition
from server.vuln_triage_env_environment import VulnTriageEnvironment


SYSTEM_PROMPT = """You are triaging open-source vulnerability reports.
Return ONLY a single JSON object — no prose, no markdown — with exactly these keys:
  action_type  : string  (required) — one of the action types listed in available_actions
  evidence_id  : string  (optional) — only used with inspect_evidence
  value        : string  (optional) — a PLAIN STRING, never an object or array
  rationale    : string  (required) — one short sentence

Valid action_type values and their expected value strings:
  read_report                     — no value needed
  inspect_evidence                — set evidence_id to one id from available_evidence
  search_nvd_database             — value: CVE ID (e.g. CVE-2023-1234) found in report aliases
  fetch_commit_diff               — value: commit hash or hash fragment found in references
  message_maintainer              — value: a question for the maintainer (e.g. "Is there a patch?")
  set_validity                    — value: "valid" | "invalid" | "needs_more_info"
  set_affected_package            — value: package name string, e.g. "guarddog"
  set_affected_versions           — value: semver range string, e.g. "<0.1.5"
  set_severity                    — value: "low" | "medium" | "high" | "critical"
  set_exploitability              — value: "low" | "medium" | "high"
  set_next_action                 — value: "patch" | "publish_advisory" | "close" | "escalate" | "request_info"
  set_missing_information         — value: one missing info item as a plain string
  submit_triage                   — no value needed

Strategy: read_report first, then use tools (search_nvd, fetch_commit, message_maintainer) to unlock hidden evidence, then fill all draft fields, then submit.
Note: You CANNOT inspect "nvd_assessment", "github_commit_diff", or "vendor_status" directly. You must use the tools above to reveal them.
"""


def get_openai_client() -> OpenAI:
    api_key = _API_KEY
    if not api_key:
        raise RuntimeError(
            "Set API_KEY, HF_TOKEN, or OPENAI_API_KEY before running the OpenAI baseline."
        )
    kwargs: Dict[str, str] = {"api_key": api_key}
    if API_BASE_URL:
        kwargs["base_url"] = API_BASE_URL
    return OpenAI(**kwargs)


def parse_json_response(text: str) -> Dict[str, str]:
    """Extract the first valid JSON object from a model response.

    Handles:
    - Markdown fences (```json ... ```)
    - Think-blocks from reasoning models (<think>...</think>)
    - Surrounding prose before/after the JSON object
    """
    import re as _re
    text = text.strip()
    # Strip reasoning/think blocks produced by models like Qwen3 or DeepSeek
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
    # Strip markdown fences
    if "```" in text:
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Find the first complete JSON object by bracket matching
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response: {text[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Incomplete JSON object in model response: {text[:200]!r}")


def heuristic_policy(observation: Dict) -> Dict[str, str]:
    if "read_report" not in observation["action_history"]:
        return {"action_type": "read_report", "rationale": "Start by reading the report"}

    truth = get_case_definition(observation["task_id"]).truth
    supporting_evidence_ids = set(truth.supporting_evidence_ids)
    visible_ids = {item["evidence_id"] for item in observation["visible_evidence"]}

    remaining_supporting = [
        evidence_id
        for evidence_id in observation["available_evidence"]
        if evidence_id in supporting_evidence_ids and evidence_id not in visible_ids
    ]
    if remaining_supporting:
        eval_id = remaining_supporting[0]
        # Interactive Tools Support:
        if eval_id == "nvd_assessment":
            # The oracle magically knows the OSV ID to query (alias)
            from server.cases import SEEDS
            seed = SEEDS[observation["task_id"]]
            return {"action_type": "search_nvd_database", "value": seed.osv_id, "rationale": "Fetch NVD dynamically"}
        elif eval_id == "github_commit_diff":
            # Match any random commit substring
            return {"action_type": "fetch_commit_diff", "value": "Commit", "rationale": "Fetch Diff dynamically"}
        elif eval_id == "vendor_status":
            return {"action_type": "message_maintainer", "value": "Is there an ETA for a patch?", "rationale": "Chat with maintainer"}
            
        return {
            "action_type": "inspect_evidence",
            "evidence_id": eval_id,
            "rationale": "Reveal the next supporting evidence item",
        }

    draft = observation["draft"]
    score = observation["score_breakdown"]

    by_truth = [
        ("set_validity", truth.validity),
        ("set_affected_package", truth.affected_package),
        ("set_affected_versions", truth.affected_versions),
        ("set_severity", truth.severity),
        ("set_exploitability", truth.exploitability),
        ("set_next_action", truth.next_action),
    ]

    for action_type, value in by_truth:
        if draft[action_type.replace("set_", "")] != value:
            return {"action_type": action_type, "value": value, "rationale": "Update the draft"}

    # Submit any required missing-information items not yet recorded in the draft
    existing_mi = {v.strip().lower() for v in draft.get("missing_information", [])}
    for mi_item in truth.missing_information:
        if mi_item.strip().lower() not in existing_mi:
            return {
                "action_type": "set_missing_information",
                "value": mi_item,
                "rationale": "Record known missing information",
            }

    return {"action_type": "submit_triage", "rationale": f"Current total score is {score['total']}"}


def llm_policy(client: OpenAI, model_name: str, observation: Dict) -> Dict[str, str]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(observation, indent=2, sort_keys=True),
            },
        ],
    )
    text = response.choices[0].message.content
    return parse_json_response(text)


_VALID_ACTION_KEYS = {"action_type", "evidence_id", "value", "rationale"}


def sanitize_action_payload(payload: Dict) -> Dict:
    """Keep only valid VulnTriageAction keys and coerce bad value types."""
    clean = {k: v for k, v in payload.items() if k in _VALID_ACTION_KEYS}
    if isinstance(clean.get("value"), (dict, list)):
        clean["value"] = json.dumps(clean["value"])
    return clean


def run_local_episode(task_id: str, policy: str, model_name: str) -> Dict[str, float]:
    print(f"[START] task={task_id}", flush=True)
    env = VulnTriageEnvironment()
    observation = env.reset(task_id=task_id).model_dump()
    client = get_openai_client() if policy == "openai" else None
    last_action_str: str = ""
    repeat_count: int = 0
    step_num: int = 1

    while not observation["done"]:
        action_payload = (
            llm_policy(client, model_name, observation) if client else heuristic_policy(observation)
        )
        # Strip unknown keys then coerce bad value types
        try:
            clean = sanitize_action_payload(action_payload)
            action = VulnTriageAction.model_validate(clean)
        except Exception as exc:
            print(f"  [warn] invalid action payload ({exc}), falling back to read_report", flush=True)
            action = VulnTriageAction(action_type="read_report", rationale="fallback: parse error")

        # Break infinite loops where model repeats the same action
        action_str = action.model_dump_json()
        if action_str == last_action_str:
            repeat_count += 1
            if repeat_count >= 3:
                print(f"  [warn] model repeated same action 3x — forcing submit_triage", flush=True)
                action = VulnTriageAction(action_type="submit_triage", rationale="loop guard")
        else:
            repeat_count = 0
        last_action_str = action_str

        observation = env.step(action).model_dump()
        step_reward = float(observation.get("reward") or 0.0)
        print(f"[STEP] step={step_num} action={action.action_type} reward={step_reward}", flush=True)
        step_num += 1

    final_score = float(observation.get("final_score") or observation.get("score_breakdown", {}).get("total", 0.0))
    print(f"[END] task={task_id} score={final_score} steps={step_num}", flush=True)

    return {
        "task_id": task_id,
        "final_score": float(observation["final_score"] or 0.0),
        "validity": observation["score_breakdown"]["validity"],
        "package_versions": round(
            (
                observation["score_breakdown"]["affected_package"]
                + observation["score_breakdown"]["affected_versions"]
            )
            / 2,
            4,
        ),
        "severity": observation["score_breakdown"]["severity"],
        "exploitability": observation["score_breakdown"]["exploitability"],
        "next_action": observation["score_breakdown"]["next_action"],
    }


def run_remote_episode(base_url: str, task_id: str, policy: str, model_name: str) -> Dict[str, float]:
    print(f"[START] task={task_id}", flush=True)
    llm_client = get_openai_client() if policy == "openai" else None
    env = GenericEnvClient(base_url=base_url).sync()
    with env:
        response = env.reset(task_id=task_id)
        observation = response.observation
        done = response.done
        step_num: int = 1
        while not done:
            action_payload = (
                llm_policy(llm_client, model_name, observation)
                if llm_client
                else heuristic_policy(observation)
            )
            response = env.step(action_payload)
            observation = response.observation
            done = response.done
            step_reward = float(getattr(response, 'reward', None) or 0.0)
            print(f"[STEP] step={step_num} action={action_payload.get('action_type')} reward={step_reward}", flush=True)
            step_num += 1

    final_score = float(observation.get("final_score") or observation.get("score_breakdown", {}).get("total", 0.0))
    print(f"[END] task={task_id} score={final_score} steps={step_num}", flush=True)

    final_score = float(observation.get("final_score") or 0.0)
    return {
        "task_id": task_id,
        "final_score": final_score,
        "validity": observation["score_breakdown"]["validity"],
        "package_versions": round(
            (
                observation["score_breakdown"]["affected_package"]
                + observation["score_breakdown"]["affected_versions"]
            )
            / 2,
            4,
        ),
        "severity": observation["score_breakdown"]["severity"],
        "exploitability": observation["score_breakdown"]["exploitability"],
        "next_action": observation["score_breakdown"]["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    # Auto-select openai policy when the validator injects API credentials;
    # fall back to heuristic for local smoke-tests with no key.
    _has_credentials = bool(_API_KEY)
    _default_policy = "openai" if _has_credentials else "heuristic"
    parser.add_argument("--policy", choices=["openai", "heuristic"], default=_default_policy)
    parser.add_argument("--model", default=MODEL_NAME)
    # Default ENV_BASE_URL to the live HF Space so the validator can reach our environment
    _default_env_url = os.getenv("ENV_BASE_URL", "https://adhitya122-vulnops.hf.space")
    parser.add_argument("--env-base-url", dest="base_url", default=_default_env_url)
    args = parser.parse_args()

    results: List[Dict[str, float]] = []
    for task_id in TASK_ORDER:
        if args.base_url:
            results.append(run_remote_episode(args.base_url, task_id, args.policy, args.model))
        else:
            results.append(run_local_episode(task_id, args.policy, args.model))

    aggregate = round(sum(item["final_score"] for item in results) / len(results), 4)
    print(json.dumps({"policy": args.policy, "model": args.model, "average_score": aggregate, "tasks": results}, indent=2))


if __name__ == "__main__":
    main()
