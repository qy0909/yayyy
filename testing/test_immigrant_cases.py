"""
Immigrant Guidance Test Runner
==============================
Runs behavior checks for the scenarios in test_cases_immigrant.md.

Usage:
    python test_immigrant_cases.py
    python test_immigrant_cases.py --url http://localhost:8000
"""

import argparse
import json
import re
import sys
import time
from typing import Dict, List

import requests

DEFAULT_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120
TOP_K = 3
MAX_RETRIES = 2
GENERIC_ANSWER_RETRIES = 4

TEST_CASES = [
    {
        "id": 1,
        "query": "I work at a construction site. My boss makes me work 12 hours every day but only pays normal salary. Is this allowed?",
        "step_question": False,
        "min_keyword_groups": 3,
        "must_have_any": [
            ["8 hour", "8-hour", "normal working hours", "eight hours"],
            ["overtime", "extra pay", "paid extra"],
            ["higher rate", "extra pay", "higher than normal", "overtime pay"],
            ["record", "keep proof", "keep records", "working hours"],
        ],
    },
    {
        "id": 2,
        "query": "If I work more than 8 hours, how should overtime be paid?",
        "step_question": False,
        "must_have_any": [
            ["overtime"],
            ["higher rate", "higher than", "extra payment", "extra pay", "1.5", "one and a half"],
        ],
    },
    {
        "id": 3,
        "query": "My employer has not paid my salary for two months. What can I do?",
        "step_question": True,
        "min_keyword_groups": 3,
        "must_have_any": [
            ["pay wages", "salary on time", "must pay", "unpaid salary", "wages on time"],
            ["complaint", "report"],
            ["proof", "records", "evidence"],
            ["labour department", "labor department"],
        ],
    },
    {
        "id": 4,
        "query": "My boss keeps my passport. He says it is company policy.",
        "step_question": False,
        "must_have_any": [
            ["passport", "personal document"],
            ["should not", "cannot", "must not", "not allowed", "not right"],
            ["report", "authorities", "support organization", "labour department"],
        ],
    },
    {
        "id": 5,
        "query": "If I complain, my boss said he will cancel my work permit. I am scared.",
        "step_question": False,
        "must_have_any": [
            ["understand", "scared", "sorry", "worry", "concern"],
            ["rights", "right", "legal rights", "protected"],
            ["authorities", "labour department", "support group", "ngo"],
        ],
    },
    {
        "id": 6,
        "query": "What proof do I need if I want to report my employer?",
        "step_question": True,
        "min_keyword_groups": 3,
        "must_have_any": [
            ["work schedule", "working schedule", "hours", "dates"],
            ["message", "whatsapp", "sms"],
            ["salary", "payslip", "pay slip", "records"],
            ["photo", "notes"],
        ],
    },
    {
        "id": 7,
        "query": "How do I report my employer?",
        "step_question": True,
        "min_keyword_groups": 3,
        "must_have_any": [
            ["collect", "prepare", "gather"],
            ["labour department", "labor department"],
            ["complaint form", "submit complaint", "submit"],
            ["investigation", "wait", "follow up"],
        ],
    },
    {
        "id": 8,
        "query": "I do not know where the labour office is. Where can I go for help?",
        "step_question": False,
        "min_keyword_groups": 1,
        "must_have_any": [
            ["labour department", "labor department", "office"],
            ["ngo", "support organization", "help center", "support center"],
        ],
    },
    {
        "id": 9,
        "query": "What happens after I file a complaint?",
        "step_question": True,
        "must_have_any": [
            ["investigate", "investigation"],
            ["evidence", "provide evidence", "documents"],
            ["pay", "owed wages", "required to pay"],
        ],
    },
    {
        "id": 10,
        "query": "Can you summarize what I should do now?",
        "step_question": True,
        "min_keyword_groups": 3,
        "conversation_history": [
            {"role": "user", "text": "My employer has not paid my salary for two months. What can I do?"},
            {"role": "assistant", "text": "Collect proof, contact the Labour Department, submit a complaint, and keep your records."},
            {"role": "user", "text": "How do I report my employer?"},
            {"role": "assistant", "text": "Use a checklist: collect evidence, file a complaint form, and follow up with the Labour Department."}
        ],
        "must_have_any": [
            ["record", "working hours", "unpaid wages"],
            ["proof", "messages", "documents"],
            ["labour department", "labor department"],
            ["complaint", "submit"],
            ["support organization", "ngo", "help center", "support center", "migrant support", "support group"],
        ],
    },
]


def is_structured_steps(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return False
    numbered = sum(1 for line in lines if re.match(r"^\d+[.)]\s", line))
    bullets = sum(1 for line in lines if re.match(r"^[\-•*]\s", line))
    if (numbered + bullets) >= 2:
        return True

    lower = answer.lower()
    sequencing_cues = ["first", "then", "next", "after that", "finally", "step"]
    cue_hits = sum(1 for cue in sequencing_cues if cue in lower)
    return cue_hits >= 2 and len(answer.strip()) > 80


def has_followup_question(answer: str) -> bool:
    return "?" in answer[-300:]


def has_guidance(answer: str) -> bool:
    cues = ["you can", "you should", "next", "contact", "visit", "report", "apply", "keep"]
    lower = answer.lower()
    return any(cue in lower for cue in cues)


def match_keywords(answer: str, groups: List[List[str]]) -> Dict[str, bool]:
    lower = answer.lower()
    results = {}
    for idx, group in enumerate(groups, start=1):
        results[f"group_{idx}"] = any(term.lower() in lower for term in group)
    return results


def run_case(base_url: str, case: dict, conversation_history: List[Dict[str, str]] = None) -> dict:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {"query": case["query"], "top_k": TOP_K}
    effective_history = case.get("conversation_history") or conversation_history
    if effective_history:
        payload["conversation_history"] = effective_history[-8:]

    last_exc = None
    data = None
    for _attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            answer_text = str(data.get("answer") or "").lower()
            if "sorry, i couldn't generate a response" in answer_text:
                data = None
                time.sleep(1.0)
                continue
            break
        except Exception as exc:
            last_exc = exc

    # Soft-retry for transient model fallback text even when HTTP succeeded.
    if data is not None:
        answer_text = str(data.get("answer") or "").lower()
        for _ in range(GENERIC_ANSWER_RETRIES - 1):
            if "sorry, i couldn't generate a response" not in answer_text:
                break
            try:
                time.sleep(1.0)
                response = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                answer_text = str(data.get("answer") or "").lower()
            except Exception:
                break

    if data is None:
        return {
            "id": case["id"],
            "query": case["query"],
            "error": str(last_exc),
            "passed": False,
            "checks": {},
        }

    answer = (data.get("answer") or "").strip()
    keyword_checks = match_keywords(answer, case["must_have_any"])
    matched_groups = sum(1 for matched in keyword_checks.values() if matched)
    required_groups = case.get("min_keyword_groups", len(case["must_have_any"]))

    checks = {
        "not_empty": len(answer) > 30,
        "has_guidance": has_guidance(answer),
        "has_followup_question": has_followup_question(answer),
        "keyword_coverage": matched_groups >= required_groups,
    }

    if case["step_question"]:
        checks["structured_steps"] = is_structured_steps(answer)

    passed = all(checks.values())

    return {
        "id": case["id"],
        "query": case["query"],
        "answer_preview": answer[:260].replace("\n", " ↵ "),
        "checks": checks,
        "keyword_groups": keyword_checks,
        "passed": passed,
        "intent": data.get("intent"),
        "rag_used": data.get("rag_used"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run immigrant scenario checks against /api/chat")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend URL")
    args = parser.parse_args()

    print("=" * 68)
    print(f"Immigrant Scenario Tests | {args.url}")
    print(f"Total cases: {len(TEST_CASES)}")
    print("=" * 68)

    results = []
    failed = 0
    for case in TEST_CASES:
        result = run_case(args.url, case)
        results.append(result)

        print("-" * 68)
        print(f"Case {result['id']}: {result['query']}")

        if "error" in result:
            failed += 1
            print(f"FAIL: {result['error']}")
            continue

        print(f"Meta: intent={result.get('intent')} rag_used={result.get('rag_used')}")
        print(f"Preview: {result.get('answer_preview', '')}")

        for key, value in result["checks"].items():
            print(f"  {key}: {'PASS' if value else 'FAIL'}")

        if result["passed"]:
            print("Result: PASS")
        else:
            failed += 1
            print("Result: FAIL")
            print(f"Keyword groups: {json.dumps(result.get('keyword_groups', {}), ensure_ascii=False)}")

    print("=" * 68)
    if failed == 0:
        print("All immigrant test cases passed.")
    else:
        print(f"Failed cases: {failed}/{len(TEST_CASES)}")
    print("=" * 68)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
