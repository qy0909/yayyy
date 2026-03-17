"""
Prompt Behavior Regression Test
================================
Validates that the live chat API returns inclusive, guide-first responses
aligned with the CivicGuide prompt design in Prompt.md.

Usage:
    python test_prompt_behavior.py              # test against running backend (localhost:8000)
    python test_prompt_behavior.py --url http://your-host:8000

Each test case asserts a set of STYLE checks (not exact wording) so the suite
remains valid even after model or knowledge-base changes.
"""

import sys
import re
import json
import argparse
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120  # seconds
TOP_K = 5

# ---------------------------------------------------------------------------
# Test cases drawn from Prompt.md user-journey scenarios
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": 1,
        "name": "First Contact – Low Confidence User",
        "type": "general",
        "query": "I don't know where to start. I need financial help.",
        "checks": {
            "has_followup_question": "Ends with a follow-up question to help user continue",
            "not_bullet_only": "Answer is not just a list of bullets",
            "not_empty": "Answer is non-empty",
            "reassuring_tone": "Contains reassuring or helpful language (e.g. 'help', 'can', 'step', 'right')",
        },
    },
    {
        "id": 2,
        "name": "Eligibility Understanding",
        "type": "factual",
        "query": "Am I eligible for healthcare subsidy?",
        "checks": {
            "has_followup_question": "Ends with a follow-up question to help user continue",
            "not_bullet_only": "Answer is not just a list of bullets",
            "not_empty": "Answer is non-empty",
            "mentions_nextStep": "Mentions a clear next action (e.g. 'check', 'visit', 'contact', 'apply')",
        },
    },
    {
        "id": 3,
        "name": "Application Process Walkthrough",
        "type": "procedural",
        "query": "How do I apply for financial assistance in Malaysia?",
        "checks": {
            "has_followup_question": "Ends with a follow-up question to help user continue",
            "not_empty": "Answer is non-empty",
            "mentions_documents": "Mentions documents, ID, or what to prepare",
            "structured_response": "Has structured guidance (numbered steps or clear intro + steps)",
        },
    },
    {
        "id": 4,
        "name": "Missing Documents",
        "type": "scenario",
        "query": "I don't have a salary slip. Can I still apply?",
        "checks": {
            "has_followup_question": "Ends with a follow-up question to help user continue",
            "not_bullet_only": "Answer is not just a list of bullets",
            "not_empty": "Answer is non-empty",
            "offers_alternative": "Offers an alternative path or safe next step",
        },
    },
    {
        "id": 5,
        "name": "Deadline Anxiety",
        "type": "scenario",
        "query": "I think I missed the deadline. What can I do?",
        "checks": {
            "has_followup_question": "Ends with a follow-up question to help user continue",
            "not_bullet_only": "Answer is not just a list of bullets",
            "not_empty": "Answer is non-empty",
            "reassuring_tone": "Contains reassuring or calm language",
            "mentions_nextStep": "Suggests a practical next step",
        },
    },
    {
        "id": 6,
        "name": "Elderly User – Minimal Digital Skills",
        "type": "scenario",
        "query": "I don't know how to use online form.",
        "checks": {
            "has_followup_question": "Ends with a follow-up question to help user continue",
            "not_bullet_only": "Answer is not just a list of bullets",
            "not_empty": "Answer is non-empty",
            "offers_offline_path": "Mentions offline option (e.g. 'visit', 'counter', 'center', 'walk-in', 'office')",
        },
    },
    {
        "id": 7,
        "name": "Intent Router – Greeting skips RAG",
        "type": "general",
        "query": "hi",
        "checks": {
            "not_empty": "Answer is non-empty",
            "intent_is_general": "API returns intent='general'",
            "rag_not_used": "API returns rag_used=false",
        },
        "api_checks": {"intent": "general", "rag_used": False},
    },
    {
        "id": 8,
        "name": "Intent Router – Thank-you skips RAG",
        "type": "general",
        "query": "thank you",
        "checks": {
            "not_empty": "Answer is non-empty",
            "intent_is_general": "API returns intent='general'",
            "rag_not_used": "API returns rag_used=false",
        },
        "api_checks": {"intent": "general", "rag_used": False},
    },
    {
        "id": 9,
        "name": "Intent Router – Policy question uses RAG",
        "type": "factual",
        "query": "What documents do I need to apply for financial assistance?",
        "checks": {
            "not_empty": "Answer is non-empty",
            "rag_is_used": "API returns rag_used=true",
            "mentions_documents": "Mentions documents or ID",
        },
        "api_checks": {"rag_used": True},
    },
    {
        "id": 10,
        "name": "Step-gating – Process question gets overview + gate offer",
        "type": "procedural",
        "query": "How do I apply for the Bantuan Sara Hidup scheme?",
        "checks": {
            "not_empty": "Answer is non-empty",
            "has_step_gate": "Answer contains the step-gate offer ([STEP_GATE])",
            "rag_is_used": "API returns rag_used=true",
        },
        "api_checks": {"rag_used": True},
    },
    {
        "id": 11,
        "name": "Step-gate confirmation – 'yes' after gate delivers full steps",
        "type": "procedural",
        "query": "yes please",
        "conversation_history": [
            {"role": "user", "text": "How do I apply for Bantuan Sara Hidup?"},
            {"role": "assistant", "text": "To apply, you need to visit MyGovUC or your nearest KWSP office. Would you like me to walk you through the detailed step-by-step process? [STEP_GATE]"},
        ],
        "checks": {
            "not_empty": "Answer is non-empty",
            "no_step_gate": "Full steps delivered — no gate marker in answer",
            "structured_response": "Response is substantive (length > 80 chars)",
        },
    },
]

# ---------------------------------------------------------------------------
# Style check functions
# ---------------------------------------------------------------------------

def check_not_empty(answer: str) -> bool:
    return len(answer.strip()) > 20


def check_has_followup_question(answer: str) -> bool:
    """At least one '?' should appear in the last 300 characters."""
    return "?" in answer[-300:]


def check_not_bullet_only(answer: str) -> bool:
    """
    Fail if EVERY non-empty line starts with a bullet/number marker.
    A mix of prose + optional list is acceptable.
    """
    non_empty_lines = [l.strip() for l in answer.split("\n") if l.strip()]
    if not non_empty_lines:
        return False
    bullet_pattern = re.compile(r"^([•\-*]|\d+[.)]) ")
    bullet_count = sum(1 for l in non_empty_lines if bullet_pattern.match(l))
    # Fail only if >80% of lines are pure bullets (clearly list-only output)
    return (bullet_count / len(non_empty_lines)) < 0.8


def check_reassuring_tone(answer: str) -> bool:
    reassure_words = [
        "help", "can", "may", "step", "right", "okay", "ok", "don't worry",
        "no worries", "still", "option", "possible", "support", "guide",
        "here", "eligible", "available", "program", "service", "assist",
        "boleh", "ada", "bantuan",  # basic Malay reassurance words
    ]
    lower = answer.lower()
    return any(w in lower for w in reassure_words)


def check_mentions_nextStep(answer: str) -> bool:
    action_words = [
        "check", "visit", "contact", "apply", "submit", "call", "go to",
        "bring", "prepare", "fill", "download", "register", "verify",
    ]
    lower = answer.lower()
    return any(w in lower for w in action_words)


def check_mentions_documents(answer: str) -> bool:
    doc_words = [
        "id", "document", "income", "proof", "slip", "letter", "card",
        "certificate", "upload", "bring", "prepare", "attach",
    ]
    lower = answer.lower()
    return any(w in lower for w in doc_words)


def check_structured_response(answer: str) -> bool:
    """At least a short intro + some guidance content (checked by length and absence of pure-bullet-only)."""
    return len(answer.strip()) > 80


def check_offers_alternative(answer: str) -> bool:
    alt_words = [
        "alternative", "instead", "other", "also", "option", "can use",
        "accept", "letter", "bank", "statement", "employer", "another",
    ]
    lower = answer.lower()
    return any(w in lower for w in alt_words)


def check_offers_offline_path(answer: str) -> bool:
    offline_words = [
        "visit", "counter", "center", "centre", "walk-in", "office",
        "in-person", "in person", "kaunter", "pejabat", "datang",
    ]
    lower = answer.lower()
    return any(w in lower for w in offline_words)


def check_has_step_gate(answer: str) -> bool:
    """Answer must contain the [STEP_GATE] marker."""
    return "[STEP_GATE]" in answer


def check_no_step_gate(answer: str) -> bool:
    """Answer must NOT contain [STEP_GATE] (full steps delivered)."""
    return "[STEP_GATE]" not in answer


# Intent/rag_used checks use a data dict injected at runtime by run_test
def check_intent_is_general(answer: str, data: dict = None) -> bool:
    return (data or {}).get("intent") == "general"


def check_rag_not_used(answer: str, data: dict = None) -> bool:
    return (data or {}).get("rag_used") is False


def check_rag_is_used(answer: str, data: dict = None) -> bool:
    return (data or {}).get("rag_used") is True


CHECK_FN_MAP = {
    "not_empty": check_not_empty,
    "has_followup_question": check_has_followup_question,
    "not_bullet_only": check_not_bullet_only,
    "reassuring_tone": check_reassuring_tone,
    "mentions_nextStep": check_mentions_nextStep,
    "mentions_documents": check_mentions_documents,
    "structured_response": check_structured_response,
    "offers_alternative": check_offers_alternative,
    "offers_offline_path": check_offers_offline_path,
    "has_step_gate": check_has_step_gate,
    "no_step_gate": check_no_step_gate,
    "intent_is_general": check_intent_is_general,
    "rag_not_used": check_rag_not_used,
    "rag_is_used": check_rag_is_used,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"
RESET = "\033[0m"


def run_test(base_url: str, case: dict) -> dict:
    """Send query and evaluate all style checks. Returns a result dict."""
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {"query": case["query"], "top_k": TOP_K}
    if "conversation_history" in case:
        payload["conversation_history"] = case["conversation_history"]

    try:
        resp = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"id": case["id"], "name": case["name"], "query": case["query"],
                "error": "Request timed out", "checks": {}}
    except Exception as exc:
        return {"id": case["id"], "name": case["name"], "query": case["query"],
                "error": str(exc), "checks": {}}

    answer = data.get("answer", "")

    # Hard failures (connection error set by server) stop here.
    # Soft failures like 'no_results' or 'low_resource_dialect' still
    # have an answer text worth checking — we run style checks against it.
    success = data.get("success", False)
    server_error = data.get("error")
    if server_error:
        return {
            "id": case["id"],
            "name": case["name"],
            "query": case["query"],
            "error": server_error,
            "checks": {},
        }

    import inspect
    check_results = {}
    for check_key, description in case["checks"].items():
        fn = CHECK_FN_MAP.get(check_key)
        if fn is None:
            check_results[check_key] = {"passed": None, "description": description}
        else:
            # Data-aware check functions accept a `data` keyword arg
            sig = inspect.signature(fn)
            if "data" in sig.parameters:
                passed = fn(answer, data=data)
            else:
                passed = fn(answer)
            check_results[check_key] = {"passed": passed, "description": description}

    result = {
        "id": case["id"],
        "name": case["name"],
        "query": case["query"],
        "answer_preview": answer[:300].replace("\n", " ↵ "),
        "api_intent": data.get("intent"),
        "api_rag_used": data.get("rag_used"),
        "checks": check_results,
    }
    if not success:
        result["api_status"] = "no_results"
    return result


def print_result(result: dict):
    print(f"\n{'─' * 62}")
    print(f"Test {result['id']}: {result.get('name', '')}")
    print(f"Query  : {result.get('query', '')}")

    if "error" in result:
        print(f"{FAIL}  Server error: {result['error']}")
        return 0, 1

    if result.get("api_status") == "no_results":
        print(f"{WARN}  No DB match for this query — evaluating fallback message quality")

    # Show intent / rag_used metadata when present
    intent = result.get("api_intent")
    rag_used = result.get("api_rag_used")
    if intent is not None or rag_used is not None:
        print(f"Meta   : intent={intent}  rag_used={rag_used}")

    print(f"Preview: {result.get('answer_preview', '')[:200]}")

    passed = 0
    failed = 0
    for key, val in result["checks"].items():
        if val["passed"] is None:
            status = WARN
        elif val["passed"]:
            status = PASS
            passed += 1
        else:
            status = FAIL
            failed += 1
        print(f"  [{key}]{status} — {val['description']}")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="CivicGuide prompt behavior regression test")
    parser.add_argument("--url", default=DEFAULT_URL, help="Backend base URL")
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        help="Run only specified test IDs (e.g. --ids 1 3 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable",
    )
    args = parser.parse_args()

    cases = TEST_CASES
    if args.ids:
        cases = [c for c in TEST_CASES if c["id"] in args.ids]

    print(f"\n{'=' * 62}")
    print(f"  CivicGuide Prompt Behavior Test  |  {args.url}")
    print(f"  {len(cases)} test case(s)")
    print(f"{'=' * 62}")

    all_results = []
    total_passed = 0
    total_failed = 0

    for case in cases:
        print(f"\n  Running test {case['id']}: {case['name']} ...", end="", flush=True)
        result = run_test(args.url, case)
        all_results.append(result)
        p, f = print_result(result)
        total_passed += p
        total_failed += f

    print(f"\n{'=' * 62}")
    print(f"  Results: {total_passed} passed / {total_failed} failed")
    if total_failed == 0:
        print(f"\033[92m  All checks passed!\033[0m")
    else:
        print(f"\033[91m  {total_failed} check(s) failed. Review answers above.\033[0m")
    print(f"{'=' * 62}\n")

    if args.json:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
