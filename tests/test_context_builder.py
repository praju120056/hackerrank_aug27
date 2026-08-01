"""
test_context_builder.py
-----------------------
Verifies that ContextBuilder produces valid context objects for every message,
including graceful handling of missing group/business/media data.
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

import csv
from context_builder import ContextBuilder

DATASET_DIR = str(REPO_ROOT / "dataset")


def load_messages():
    path = REPO_ROOT / "dataset" / "messages.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_every_message_produces_context():
    cb = ContextBuilder(DATASET_DIR)
    messages = load_messages()
    for msg in messages:
        ctx = cb.build_context(msg)
        assert ctx is not None, f"None context for {msg['message_id']}"
        assert ctx["message_id"] == msg["message_id"]
    print(f"[PASS] All {len(messages)} messages produce a context dict")


def test_context_has_required_keys():
    cb = ContextBuilder(DATASET_DIR)
    messages = load_messages()
    required = [
        "message_id", "message_text", "media_type", "conversation_type",
        "forwarded_count", "user", "group", "business",
        "preference_signals", "evidence", "fatigue",
    ]
    for msg in messages:
        ctx = cb.build_context(msg)
        for key in required:
            assert key in ctx, f"Key '{key}' missing in context for {msg['message_id']}"
    print(f"[PASS] All required context keys present across {len(messages)} messages")


def test_missing_group_handled():
    """A business message has no group → group context should be None."""
    cb = ContextBuilder(DATASET_DIR)
    fake_msg = {
        "message_id": "test_biz",
        "user_id": "u_001",
        "conversation_type": "business",
        "group_id": "",
        "business_id": "business_001",
        "sender_user_id": "",
        "created_at": "2026-07-31 10:00",
        "message_text": "Test",
        "media_type": "",
        "media_id": "",
        "forwarded_count": "0",
    }
    ctx = cb.build_context(fake_msg)
    assert ctx["group"] is None, "group should be None for business message"
    print("[PASS] Missing group handled gracefully (None)")


def test_missing_business_handled():
    """A group message has no business → business context should be None."""
    cb = ContextBuilder(DATASET_DIR)
    fake_msg = {
        "message_id": "test_group",
        "user_id": "u_001",
        "conversation_type": "group",
        "group_id": "group_002",
        "business_id": "",
        "sender_user_id": "u_043",
        "created_at": "2026-07-31 10:00",
        "message_text": "Test group message",
        "media_type": "",
        "media_id": "",
        "forwarded_count": "0",
    }
    ctx = cb.build_context(fake_msg)
    assert ctx["business"] is None, "business should be None for group message"
    print("[PASS] Missing business handled gracefully (None)")


def test_missing_media_path():
    """A message referencing a non-existent media file should return None path."""
    cb = ContextBuilder(DATASET_DIR)
    path = cb.get_image_path("img_nonexistent_999")
    assert path is None, f"Expected None for non-existent media, got {path}"
    print("[PASS] Non-existent media_id returns None path")


def test_user_context_rates_are_floats():
    cb = ContextBuilder(DATASET_DIR)
    messages = load_messages()
    for msg in messages[:10]:  # spot check first 10
        ctx = cb.build_context(msg)
        user = ctx["user"]
        assert isinstance(user["dismiss_rate_30d"], float)
        assert isinstance(user["report_rate_30d"], float)
        assert isinstance(user["engage_rate_30d"], float)
        assert isinstance(user["dnd_active"], bool)
    print("[PASS] User context rates are float, dnd_active is bool")


def test_fatigue_ratio_valid():
    cb = ContextBuilder(DATASET_DIR)
    messages = load_messages()
    for msg in messages:
        ctx = cb.build_context(msg)
        ratio = ctx["fatigue"]["last_7d_dismissed_ratio"]
        assert 0.0 <= ratio <= 1.0, (
            f"Fatigue ratio out of range for {msg['message_id']}: {ratio}"
        )
    print("[PASS] All fatigue ratios in [0.0, 1.0]")


def test_dnd_midnight_spanning():
    """DND window spanning midnight (e.g. 22:00-07:00) should detect late-night messages."""
    cb = ContextBuilder(DATASET_DIR)
    # 23:30 should be in DND window 22:00-07:00
    result = cb._is_dnd_active("22:00-07:00", "2026-07-31 23:30")
    assert result is True, "23:30 should be in DND 22:00-07:00"
    # 10:00 should NOT be in DND window 22:00-07:00
    result2 = cb._is_dnd_active("22:00-07:00", "2026-07-31 10:00")
    assert result2 is False, "10:00 should not be in DND 22:00-07:00"
    print("[PASS] DND midnight-spanning window detection correct")


if __name__ == "__main__":
    errors = []
    tests = [
        test_every_message_produces_context,
        test_context_has_required_keys,
        test_missing_group_handled,
        test_missing_business_handled,
        test_missing_media_path,
        test_user_context_rates_are_floats,
        test_fatigue_ratio_valid,
        test_dnd_midnight_spanning,
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
