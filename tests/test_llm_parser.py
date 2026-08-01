"""
test_llm_parser.py
------------------
Tests for LLM response parsing, output validation, and confidence calibration.
No Gemini API calls required — uses stub clients.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

from models import RoutingPrediction, CONF_MIN, CONF_MAX
from llm_router import (
    LLMRouter, _extract_json_array, _extract_retry_seconds,
    _validate_and_repair, _fallback,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ctx(message_id: str = "msg_t01", bias: float = 0.0) -> dict:
    return {
        "message_id": message_id,
        "text": "Test message",
        "media_type": "",
        "media": None,
        "conversation_type": "personal",
        "forwarded_count": 0,
        "created_at": "2026-07-31 10:00",
        "user": {"dnd_active": False, "dismiss_rate_30d": 0.2,
                 "report_rate_30d": 0.0, "engage_rate_30d": 0.5, "total_messages_30d": 60},
        "group": None, "business": None,
        "preference_signals": {"priority_bias": bias, "reason_hint": ""},
        "evidence": [],
        "fatigue": {"last_7d_dismissed_ratio": 0.1},
    }


class _FakeClient:
    """Stub that immediately returns pre-configured text."""
    def __init__(self, text: str):
        self._text = text
        self.models = self

    def generate_content(self, **kwargs):
        class R:
            text = None
        r = R()
        r.text = self._text
        return r


def _router(text: str) -> LLMRouter:
    return LLMRouter(
        client=_FakeClient(text),
        model_name="fake-model",
        batch_size=5,
        sleep_between_batch=0,
        max_retries=0,
        checkpoint_path=None,
    )


# ── JSON array extraction ─────────────────────────────────────────────────────

def test_extract_plain_array():
    assert _extract_json_array('[{"a":1}]') == '[{"a":1}]'
    print("[PASS] _extract_json_array: plain array")


def test_extract_strips_fences():
    raw = "```json\n[{\"a\":1}]\n```"
    result = json.loads(_extract_json_array(raw))
    assert result == [{"a": 1}]
    print("[PASS] _extract_json_array: strips markdown fences")


def test_extract_from_prose():
    raw = "Here is the result:\n\n[{\"x\": 2}]\n\nDone."
    result = json.loads(_extract_json_array(raw))
    assert result == [{"x": 2}]
    print("[PASS] _extract_json_array: extracts from prose wrapper")


# ── RetryInfo parsing ─────────────────────────────────────────────────────────

def test_retry_seconds_from_error():
    class FakeExc(Exception):
        pass
    exc = FakeExc("'retryDelay': '32s'")
    delay = _extract_retry_seconds(exc)
    assert delay == 37  # 32 + 5 buffer
    print(f"[PASS] RetryInfo parsed correctly: {delay}s")


def test_retry_seconds_default():
    class FakeExc(Exception):
        pass
    exc = FakeExc("some generic error with no retry info")
    delay = _extract_retry_seconds(exc, default=65)
    assert delay == 65
    print("[PASS] RetryInfo default returned when not found")


# ── Fallback ──────────────────────────────────────────────────────────────────

def test_fallback_returns_digest():
    ctx = _ctx("msg_fb01")
    p = _fallback(ctx)
    assert isinstance(p, RoutingPrediction)
    assert p.action == "digest"
    assert p.message_type == "unknown"
    assert p.confidence == 0.5
    assert p.evidence_message_ids == "none"
    print("[PASS] Fallback produces valid digest prediction")


# ── Validate and repair ───────────────────────────────────────────────────────

def test_valid_item_parsed():
    item = {"message_id": "msg_v01", "action": "notify", "message_type": "urgent",
            "reason": "Urgent work message.", "confidence": 0.85, "evidence_message_ids": "message_0001"}
    ctx = _ctx("msg_v01")
    p = _validate_and_repair(item, ctx)
    assert p.action == "notify"
    assert p.message_type == "urgent"
    assert p.confidence == 0.85
    assert "message_0001" in p.evidence_message_ids
    print("[PASS] Valid item parsed correctly")


def test_invalid_action_repaired():
    item = {"message_id": "msg_r01", "action": "INVALID", "message_type": "urgent",
            "reason": "Test", "confidence": 0.8, "evidence_message_ids": "none"}
    p = _validate_and_repair(item, _ctx("msg_r01"))
    assert p.action == "digest", f"Expected 'digest', got {p.action!r}"
    print("[PASS] Invalid action repaired to 'digest'")


def test_invalid_message_type_repaired():
    item = {"message_id": "msg_r02", "action": "mute", "message_type": "GARBAGE",
            "reason": "Test", "confidence": 0.8, "evidence_message_ids": "none"}
    p = _validate_and_repair(item, _ctx("msg_r02"))
    assert p.message_type == "unknown", f"Expected 'unknown', got {p.message_type!r}"
    print("[PASS] Invalid message_type repaired to 'unknown'")


def test_empty_reason_repaired():
    item = {"message_id": "msg_r03", "action": "digest", "message_type": "personal",
            "reason": "   ", "confidence": 0.7, "evidence_message_ids": "none"}
    p = _validate_and_repair(item, _ctx("msg_r03"))
    assert p.reason.strip(), "Reason should not be blank after repair"
    print("[PASS] Empty reason repaired to default")


# ── Confidence calibration ────────────────────────────────────────────────────

def test_confidence_clamped_at_max():
    item = {"message_id": "msg_c01", "action": "notify", "message_type": "urgent",
            "reason": "Test", "confidence": 0.99, "evidence_message_ids": "none"}
    p = _validate_and_repair(item, _ctx("msg_c01", bias=0.0))
    assert p.confidence <= CONF_MAX, f"Expected <={CONF_MAX}, got {p.confidence}"
    print(f"[PASS] Confidence clamped at CONF_MAX={CONF_MAX}: {p.confidence}")


def test_confidence_clamped_at_min():
    item = {"message_id": "msg_c02", "action": "mute", "message_type": "spam",
            "reason": "Test", "confidence": 0.10, "evidence_message_ids": "none"}
    p = _validate_and_repair(item, _ctx("msg_c02", bias=0.0))
    assert p.confidence >= CONF_MIN, f"Expected >={CONF_MIN}, got {p.confidence}"
    print(f"[PASS] Confidence clamped at CONF_MIN={CONF_MIN}: {p.confidence}")


def test_priority_bias_applied():
    bias = -0.20
    llm_conf = 0.80
    item = {"message_id": "msg_c03", "action": "mute", "message_type": "promotion",
            "reason": "Test", "confidence": llm_conf, "evidence_message_ids": "none"}
    p = _validate_and_repair(item, _ctx("msg_c03", bias=bias))
    expected = max(CONF_MIN, min(CONF_MAX, llm_conf + bias))
    assert abs(p.confidence - expected) < 0.001, f"Expected {expected:.3f}, got {p.confidence}"
    print(f"[PASS] priority_bias applied: {llm_conf} + ({bias}) = {p.confidence}")


# ── Malformed response handling ───────────────────────────────────────────────

def test_malformed_json_fallback():
    router = _router("this is not json at all {{{{")
    batch = [_ctx("msg_mf01")]
    results = router._parse_response("not json", batch)
    assert len(results) == 1
    assert results[0].action == "digest"
    assert results[0].confidence == 0.5
    print("[PASS] Malformed JSON triggers per-batch fallback")


def test_missing_message_id_fallback():
    router = _router("")
    batch = [_ctx("msg_miss01")]
    resp = json.dumps([{"message_id": "msg_OTHER", "action": "notify",
                        "message_type": "urgent", "reason": "Test",
                        "confidence": 0.8, "evidence_message_ids": "none"}])
    results = router._parse_response(resp, batch)
    assert results[0].message_id == "msg_miss01"
    assert results[0].action == "digest"  # fallback
    print("[PASS] Missing message_id in response triggers per-message fallback")


def test_valid_batch_of_three():
    router = _router("")
    batch = [_ctx(f"msg_b{i}", bias=0.0) for i in range(1, 4)]
    resp = json.dumps([
        {"message_id": "msg_b1", "action": "notify", "message_type": "urgent",
         "reason": "Urgent", "confidence": 0.85, "evidence_message_ids": "none"},
        {"message_id": "msg_b2", "action": "digest", "message_type": "personal",
         "reason": "Safe", "confidence": 0.75, "evidence_message_ids": "none"},
        {"message_id": "msg_b3", "action": "mute", "message_type": "spam",
         "reason": "Spam", "confidence": 0.80, "evidence_message_ids": "none"},
    ])
    results = router._parse_response(resp, batch)
    assert len(results) == 3
    assert results[0].action == "notify"
    assert results[1].action == "digest"
    assert results[2].action == "mute"
    print("[PASS] Valid batch of 3 messages parsed correctly")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    errors = []
    tests = [
        test_extract_plain_array, test_extract_strips_fences, test_extract_from_prose,
        test_retry_seconds_from_error, test_retry_seconds_default,
        test_fallback_returns_digest,
        test_valid_item_parsed, test_invalid_action_repaired,
        test_invalid_message_type_repaired, test_empty_reason_repaired,
        test_confidence_clamped_at_max, test_confidence_clamped_at_min,
        test_priority_bias_applied,
        test_malformed_json_fallback, test_missing_message_id_fallback,
        test_valid_batch_of_three,
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
