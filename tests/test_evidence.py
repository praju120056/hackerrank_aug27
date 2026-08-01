"""
test_evidence.py
----------------
Tests for the evidence retrieval module. No Gemini API calls required.
Uses the real dataset message_history.csv and message_events.csv.
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

from evidence import retrieve_evidence, evidence_ids_string


def _csv(name: str) -> list[dict]:
    path = REPO_ROOT / "dataset" / name
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


HISTORY = _csv("message_history.csv")
EVENT_ROWS = _csv("message_events.csv")
EVENTS = {(r["user_id"], r["message_id"]): r for r in EVENT_ROWS}
MESSAGES = _csv("messages.csv")
VALID_HIST_IDS = {r["message_id"] for r in HISTORY}


def test_evidence_ids_in_history():
    """All returned evidence IDs must exist in message_history."""
    for msg in MESSAGES:
        for item in retrieve_evidence(msg, HISTORY, EVENTS):
            assert item["message_id"] in VALID_HIST_IDS, (
                f"Evidence ID {item['message_id']} not in history for {msg['message_id']}"
            )
    print("[PASS] All returned evidence IDs exist in message_history")


def test_max_top_k():
    for msg in MESSAGES:
        evidence = retrieve_evidence(msg, HISTORY, EVENTS, top_k=3)
        assert len(evidence) <= 3, f"Evidence exceeds top_k=3 for {msg['message_id']}"
    print("[PASS] Evidence never exceeds top_k=3")


def test_no_evidence_unknown_user():
    fake = {
        "message_id": "fake_001", "user_id": "u_NOBODY",
        "sender_user_id": "u_STRANGER", "group_id": "", "business_id": "",
    }
    assert retrieve_evidence(fake, HISTORY, EVENTS) == []
    print("[PASS] Unknown user/sender returns empty list")


def test_ids_string_none():
    assert evidence_ids_string([]) == "none"
    print("[PASS] evidence_ids_string([]) returns 'none'")


def test_ids_string_join():
    items = [{"message_id": "message_0001"}, {"message_id": "message_0002"}]
    assert evidence_ids_string(items) == "message_0001;message_0002"
    print("[PASS] evidence_ids_string joins with semicolons")


def test_sender_priority_over_group():
    """Priority 1 (sender match) must appear before Priority 2 (group match)."""
    history = [
        {"message_id": "h_group", "user_id": "u_t", "sender_user_id": "u_other",
         "group_id": "g_x", "business_id": "", "message_text": "group msg",
         "created_at": "2026-07-01 10:00"},
        {"message_id": "h_sender", "user_id": "u_t", "sender_user_id": "u_a",
         "group_id": "g_x", "business_id": "", "message_text": "sender msg",
         "created_at": "2026-07-01 09:00"},   # older but higher priority
    ]
    events = {}
    msg = {"message_id": "in_001", "user_id": "u_t", "sender_user_id": "u_a",
           "group_id": "g_x", "business_id": ""}
    evidence = retrieve_evidence(msg, history, events)
    ids = [e["message_id"] for e in evidence]
    assert ids[0] == "h_sender", f"Expected sender match first, got {ids}"
    print("[PASS] Sender match takes priority over group match")


def test_recency_sorting_within_tier():
    """Within the same tier, more recent messages should appear first."""
    history = [
        {"message_id": "old", "user_id": "u_t", "sender_user_id": "u_s",
         "group_id": "", "business_id": "", "message_text": "old",
         "created_at": "2026-01-01 10:00"},
        {"message_id": "new", "user_id": "u_t", "sender_user_id": "u_s",
         "group_id": "", "business_id": "", "message_text": "new",
         "created_at": "2026-07-30 10:00"},
    ]
    msg = {"message_id": "in", "user_id": "u_t", "sender_user_id": "u_s",
           "group_id": "", "business_id": ""}
    evidence = retrieve_evidence(msg, history, {})
    assert evidence[0]["message_id"] == "new", "More recent message should come first"
    print("[PASS] Evidence sorted by recency within tier")


def test_event_flags_present():
    for msg in MESSAGES[:20]:
        for item in retrieve_evidence(msg, HISTORY, EVENTS):
            for flag in ("opened", "replied", "dismissed", "muted_after", "reported"):
                assert flag in item, f"Flag '{flag}' missing from {item.get('message_id')}"
    print("[PASS] All event flags present in evidence items")


def test_text_truncated():
    long_text = "X" * 300
    history = [{"message_id": "h_long", "user_id": "u_t", "sender_user_id": "u_s",
                "group_id": "", "business_id": "", "message_text": long_text,
                "created_at": "2026-07-01 10:00"}]
    msg = {"message_id": "in", "user_id": "u_t", "sender_user_id": "u_s",
           "group_id": "", "business_id": ""}
    evidence = retrieve_evidence(msg, history, {})
    assert len(evidence[0]["text"]) <= 121, "Text should be truncated to <=120 chars + ellipsis"
    print("[PASS] Evidence text truncated to <=120 chars")


if __name__ == "__main__":
    errors = []
    tests = [
        test_evidence_ids_in_history, test_max_top_k,
        test_no_evidence_unknown_user, test_ids_string_none, test_ids_string_join,
        test_sender_priority_over_group, test_recency_sorting_within_tier,
        test_event_flags_present, test_text_truncated,
    ]
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            errors.append(fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] {fn.__name__}: {e}")
            errors.append(fn.__name__)
    print()
    if errors:
        print(f"FAILED: {errors}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
