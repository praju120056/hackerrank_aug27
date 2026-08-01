"""
test_evidence.py
----------------
Tests for the evidence retrieval module:
  - Returns valid message IDs from message_history
  - Respects priority order (sender > group > business)
  - Returns at most top_k results
  - Returns "none" when no evidence exists
  - Handles the no-evidence case gracefully
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

import csv
from evidence import retrieve_evidence, evidence_ids_string


def _load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_events_map(event_rows):
    return {(r["user_id"], r["message_id"]): r for r in event_rows}


# ── Load real dataset for integration-style tests ─────────────────────────────
HISTORY = _load_csv(REPO_ROOT / "dataset" / "message_history.csv")
EVENT_ROWS = _load_csv(REPO_ROOT / "dataset" / "message_events.csv")
EVENTS = _build_events_map(EVENT_ROWS)
MESSAGES = _load_csv(REPO_ROOT / "dataset" / "messages.csv")

VALID_HISTORY_IDS = {row["message_id"] for row in HISTORY}


def test_evidence_ids_are_valid():
    """All returned evidence message_ids must exist in message_history."""
    for msg in MESSAGES:
        evidence = retrieve_evidence(msg, HISTORY, EVENTS)
        for item in evidence:
            mid = item["message_id"]
            assert mid in VALID_HISTORY_IDS, (
                f"Evidence ID '{mid}' not found in message_history for {msg['message_id']}"
            )
    print("[PASS] All returned evidence IDs exist in message_history")


def test_evidence_max_top_k():
    """retrieve_evidence should return at most top_k=3 results."""
    for msg in MESSAGES:
        evidence = retrieve_evidence(msg, HISTORY, EVENTS, top_k=3)
        assert len(evidence) <= 3, (
            f"Evidence count {len(evidence)} exceeds top_k=3 for {msg['message_id']}"
        )
    print("[PASS] Evidence never exceeds top_k=3")


def test_no_evidence_returns_empty_list():
    """A user+sender combination with no history should return []."""
    fake_msg = {
        "message_id": "fake_msg_001",
        "user_id": "u_NONEXISTENT",
        "sender_user_id": "u_STRANGER_999",
        "group_id": "",
        "business_id": "",
        "conversation_type": "personal",
    }
    evidence = retrieve_evidence(fake_msg, HISTORY, EVENTS)
    assert evidence == [], f"Expected empty list, got {evidence}"
    print("[PASS] No evidence for non-existent user returns []")


def test_evidence_ids_string_no_evidence():
    """evidence_ids_string([]) should return 'none'."""
    result = evidence_ids_string([])
    assert result == "none", f"Expected 'none', got '{result}'"
    print("[PASS] evidence_ids_string([]) returns 'none'")


def test_evidence_ids_string_with_items():
    """evidence_ids_string should join IDs with semicolons."""
    evidence = [
        {"message_id": "message_0001"},
        {"message_id": "message_0002"},
    ]
    result = evidence_ids_string(evidence)
    assert result == "message_0001;message_0002", f"Got: {result}"
    print("[PASS] evidence_ids_string joins IDs with semicolons")


def test_evidence_priority_sender_over_group():
    """Priority 1 (sender match) should take precedence over Priority 2 (group match)."""
    # Build a controlled history with both sender and group matches
    user_id = "u_test"
    sender_id = "u_sender_A"
    group_id = "group_X"

    history = [
        # Sender match
        {
            "message_id": "hist_sender_1",
            "user_id": user_id,
            "sender_user_id": sender_id,
            "group_id": group_id,
            "business_id": "",
            "message_text": "Sender message",
        },
        # Group match (different sender)
        {
            "message_id": "hist_group_1",
            "user_id": user_id,
            "sender_user_id": "u_other_sender",
            "group_id": group_id,
            "business_id": "",
            "message_text": "Group message",
        },
    ]
    events = {}

    msg = {
        "message_id": "incoming_001",
        "user_id": user_id,
        "sender_user_id": sender_id,
        "group_id": group_id,
        "business_id": "",
    }

    evidence = retrieve_evidence(msg, history, events, top_k=3)
    ids = [e["message_id"] for e in evidence]
    assert "hist_sender_1" in ids, "Sender match should appear in evidence"
    # Sender match should appear first
    assert ids[0] == "hist_sender_1", f"Expected sender match first, got {ids}"
    print("[PASS] Sender match takes priority over group match in evidence")


def test_evidence_includes_event_flags():
    """Each evidence item should include opened, replied, dismissed, muted_after, reported."""
    for msg in MESSAGES[:20]:  # spot check
        evidence = retrieve_evidence(msg, HISTORY, EVENTS)
        for item in evidence:
            for flag in ("opened", "replied", "dismissed", "muted_after", "reported"):
                assert flag in item, (
                    f"Flag '{flag}' missing from evidence item {item.get('message_id')} "
                    f"for {msg['message_id']}"
                )
    print("[PASS] All evidence items include all event flags")


def test_evidence_text_truncated():
    """Evidence text should be truncated to max 120 chars plus ellipsis."""
    long_text = "A" * 200
    history = [
        {
            "message_id": "hist_long",
            "user_id": "u_trunc",
            "sender_user_id": "u_sender_trunc",
            "group_id": "",
            "business_id": "",
            "message_text": long_text,
        }
    ]
    msg = {
        "message_id": "incoming_trunc",
        "user_id": "u_trunc",
        "sender_user_id": "u_sender_trunc",
        "group_id": "",
        "business_id": "",
    }
    evidence = retrieve_evidence(msg, history, {})
    assert len(evidence) == 1
    text = evidence[0]["text"]
    assert len(text) <= 121, f"Evidence text too long: {len(text)} chars"  # 120 + ellipsis
    print(f"[PASS] Evidence text truncated to ≤120 chars (got {len(text)})")


if __name__ == "__main__":
    errors = []
    tests = [
        test_evidence_ids_are_valid,
        test_evidence_max_top_k,
        test_no_evidence_returns_empty_list,
        test_evidence_ids_string_no_evidence,
        test_evidence_ids_string_with_items,
        test_evidence_priority_sender_over_group,
        test_evidence_includes_event_flags,
        test_evidence_text_truncated,
    ]
    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            errors.append(test_fn.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {test_fn.__name__}: {e}")
            errors.append(test_fn.__name__)

    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s): {errors}")
        sys.exit(1)
    else:
        print(f"ALL TESTS PASSED ({len(tests)} tests)")
