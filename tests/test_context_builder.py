"""
test_context_builder.py
-----------------------
Validates ContextBuilder using the real dataset files. No Gemini calls.
"""

import sys
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

from models import MediaSummary
from context_builder import ContextBuilder

DATASET_DIR = str(REPO_ROOT / "dataset")


def _messages():
    path = REPO_ROOT / "dataset" / "messages.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _no_media() -> MediaSummary:
    return MediaSummary("", "unknown", "low", [], False, 1.0)


def test_all_messages_produce_context():
    cb = ContextBuilder(DATASET_DIR)
    msgs = _messages()
    for msg in msgs:
        ctx = cb.build_context(msg, _no_media())
        assert ctx is not None, f"None context for {msg['message_id']}"
        assert ctx["message_id"] == msg["message_id"]
    print(f"[PASS] All {len(msgs)} messages produce a context dict")


def test_required_context_keys():
    cb = ContextBuilder(DATASET_DIR)
    required = [
        "message_id", "text", "media_type", "media", "conversation_type",
        "forwarded_count", "user", "group", "business",
        "preference_signals", "evidence", "fatigue", "created_at",
    ]
    for msg in _messages()[:5]:
        ctx = cb.build_context(msg, _no_media())
        for k in required:
            assert k in ctx, f"Key '{k}' missing for {msg['message_id']}"
    print("[PASS] All required context keys present")


def test_business_message_has_no_group():
    cb = ContextBuilder(DATASET_DIR)
    fake = {
        "message_id": "t_biz", "user_id": "u_001", "conversation_type": "business",
        "group_id": "", "business_id": "business_001", "sender_user_id": "",
        "created_at": "2026-07-31 10:00", "message_text": "Test",
        "media_type": "", "media_id": "", "forwarded_count": "0",
    }
    ctx = cb.build_context(fake, _no_media())
    assert ctx["group"] is None, "group should be None for a business message"
    assert ctx["business"] is not None, "business should be populated"
    print("[PASS] Business message: group=None, business populated")


def test_group_message_has_no_business():
    cb = ContextBuilder(DATASET_DIR)
    fake = {
        "message_id": "t_grp", "user_id": "u_001", "conversation_type": "group",
        "group_id": "group_002", "business_id": "", "sender_user_id": "u_043",
        "created_at": "2026-07-31 10:00", "message_text": "Test",
        "media_type": "", "media_id": "", "forwarded_count": "0",
    }
    ctx = cb.build_context(fake, _no_media())
    assert ctx["business"] is None, "business should be None for a group message"
    assert ctx["group"] is not None, "group should be populated"
    print("[PASS] Group message: business=None, group populated")


def test_missing_media_id_returns_none_path():
    cb = ContextBuilder(DATASET_DIR)
    assert cb.image_path("img_NONEXISTENT") is None
    assert cb.voice_path("vn_NONEXISTENT") is None
    print("[PASS] Non-existent media_id returns None path")


def test_user_rates_are_floats_in_range():
    cb = ContextBuilder(DATASET_DIR)
    for msg in _messages()[:10]:
        ctx = cb.build_context(msg, _no_media())
        u = ctx["user"]
        assert 0.0 <= u["dismiss_rate_30d"] <= 1.0
        assert 0.0 <= u["report_rate_30d"] <= 1.0
        assert isinstance(u["dnd_active"], bool)
    print("[PASS] User rate fields are valid floats in [0,1]")


def test_fatigue_ratio_in_range():
    cb = ContextBuilder(DATASET_DIR)
    for msg in _messages():
        ctx = cb.build_context(msg, _no_media())
        r = ctx["fatigue"]["last_7d_dismissed_ratio"]
        assert 0.0 <= r <= 1.0, f"Fatigue ratio out of range for {msg['message_id']}: {r}"
    print("[PASS] All fatigue ratios in [0.0, 1.0]")


def test_media_summary_embedded_in_context():
    cb = ContextBuilder(DATASET_DIR)
    fake = {
        "message_id": "t_img", "user_id": "u_001", "conversation_type": "group",
        "group_id": "group_001", "business_id": "", "sender_user_id": "u_043",
        "created_at": "2026-07-31 10:00", "message_text": "",
        "media_type": "image", "media_id": "img_008", "forwarded_count": "0",
    }
    ms = MediaSummary(
        summary="A poster showing a Myntra kurta for sale.",
        category="promotional",
        urgency="low",
        entities=["Myntra"],
        action_required=False,
        confidence=0.9,
    )
    ctx = cb.build_context(fake, ms)
    assert ctx["media"] is not None, "media context should be populated"
    assert ctx["media"]["category"] == "promotional"
    assert "Myntra" in ctx["media"]["entities"]
    print("[PASS] MediaSummary correctly embedded in context dict")


def test_dnd_midnight_spanning():
    cb = ContextBuilder(DATASET_DIR)
    assert cb._dnd_active("22:00-07:00", "2026-07-31 23:30") is True
    assert cb._dnd_active("22:00-07:00", "2026-07-31 10:00") is False
    assert cb._dnd_active("22:00-07:00", "2026-07-31 06:00") is True
    print("[PASS] DND midnight-spanning detection correct")


if __name__ == "__main__":
    errors = []
    tests = [
        test_all_messages_produce_context, test_required_context_keys,
        test_business_message_has_no_group, test_group_message_has_no_business,
        test_missing_media_id_returns_none_path, test_user_rates_are_floats_in_range,
        test_fatigue_ratio_in_range, test_media_summary_embedded_in_context,
        test_dnd_midnight_spanning,
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
