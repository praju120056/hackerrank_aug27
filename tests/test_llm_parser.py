"""
test_llm_parser.py
------------------
Tests for LLMRouter parsing and confidence calibration (no API calls):
  - Malformed JSON falls back gracefully
  - Empty response falls back gracefully
  - Valid JSON is parsed correctly
  - Confidence clamped to [0.55, 0.95] after bias
  - Invalid action/type values default to digest/unknown
  - priority_bias is correctly applied to confidence
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

from llm_router import LLMRouter, CONF_MIN, CONF_MAX, _extract_json_array


# ── Stub client (no real API calls) ──────────────────────────────────────────

class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self, response_text):
        self._response_text = response_text

    def generate_content(self, **kwargs):
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.models = _FakeModel(response_text)


def _make_ctx(message_id="msg_t01", bias=0.0):
    return {
        "message_id": message_id,
        "message_text": "Test message",
        "media_type": "",
        "media_summary": "",
        "conversation_type": "personal",
        "forwarded_count": 0,
        "created_at": "2026-07-31 10:00",
        "user": {"dnd_active": False, "dismiss_rate_30d": 0.2, "report_rate_30d": 0.0, "engage_rate_30d": 0.5},
        "group": None,
        "business": None,
        "preference_signals": {"priority_bias": bias, "reason_hint": ""},
        "evidence": [],
        "fatigue": {"last_7d_dismissed_ratio": 0.1},
    }


# ------------------------------------------------------------------
# JSON extraction helper
# ------------------------------------------------------------------

def test_extract_json_array_plain():
    raw = '[{"a": 1}]'
    result = _extract_json_array(raw)
    assert result == raw
    print("[PASS] _extract_json_array handles plain JSON array")


def test_extract_json_array_with_fences():
    raw = "```json\n[{\"a\": 1}]\n```"
    result = _extract_json_array(raw)
    parsed = json.loads(result)
    assert parsed == [{"a": 1}]
    print("[PASS] _extract_json_array strips markdown fences")


def test_extract_json_array_with_preamble():
    raw = "Sure, here is the result:\n\n[{\"x\": 2}]\n\nHope that helps."
    result = _extract_json_array(raw)
    parsed = json.loads(result)
    assert parsed == [{"x": 2}]
    print("[PASS] _extract_json_array extracts array from prose-wrapped response")


# ------------------------------------------------------------------
# Malformed JSON fallback
# ------------------------------------------------------------------

def test_malformed_json_fallback():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_x01")]

    # Simulate malformed response
    results = router._parse_response("this is not json at all {{{}}", batch)
    assert len(results) == 1
    assert results[0]["action"] == "digest"
    assert results[0]["confidence"] == 0.5
    assert results[0]["evidence_message_ids"] == "none"
    print("[PASS] Malformed JSON falls back to digest/0.5/none")


def test_empty_response_fallback():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_x02")]
    results = router._parse_response("", batch)
    assert len(results) == 1
    assert results[0]["action"] == "digest"
    print("[PASS] Empty response falls back correctly")


def test_missing_message_id_in_response_fallback():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_x03")]
    # Response doesn't include msg_x03
    response = json.dumps([{"message_id": "msg_other", "action": "notify",
                            "message_type": "urgent", "reason": "test", "confidence": 0.8,
                            "evidence_message_ids": "none"}])
    results = router._parse_response(response, batch)
    assert results[0]["message_id"] == "msg_x03"
    assert results[0]["action"] == "digest"  # fallback
    print("[PASS] Missing message_id in response triggers per-message fallback")


# ------------------------------------------------------------------
# Valid JSON parsing
# ------------------------------------------------------------------

def test_valid_response_parsed_correctly():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_v01")]
    response = json.dumps([
        {
            "message_id": "msg_v01",
            "action": "notify",
            "message_type": "urgent",
            "reason": "This is an urgent message requiring immediate attention.",
            "confidence": 0.85,
            "evidence_message_ids": "message_0001;message_0002",
        }
    ])
    results = router._parse_response(response, batch)
    assert len(results) == 1
    r = results[0]
    assert r["action"] == "notify"
    assert r["message_type"] == "urgent"
    assert r["confidence"] == 0.85
    assert "message_0001" in r["evidence_message_ids"]
    print("[PASS] Valid response parsed correctly")


# ------------------------------------------------------------------
# Confidence calibration
# ------------------------------------------------------------------

def test_confidence_clamped_at_max():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_c01", bias=0.0)]
    # LLM returns 0.99 → should be clamped to 0.95
    response = json.dumps([
        {"message_id": "msg_c01", "action": "notify", "message_type": "urgent",
         "reason": "Test", "confidence": 0.99, "evidence_message_ids": "none"}
    ])
    results = router._parse_response(response, batch)
    assert results[0]["confidence"] <= CONF_MAX, (
        f"Expected ≤{CONF_MAX}, got {results[0]['confidence']}"
    )
    print(f"[PASS] Confidence clamped at max {CONF_MAX}: got {results[0]['confidence']}")


def test_confidence_clamped_at_min():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_c02", bias=0.0)]
    # LLM returns 0.10 → should be clamped to 0.55
    response = json.dumps([
        {"message_id": "msg_c02", "action": "mute", "message_type": "spam",
         "reason": "Test", "confidence": 0.10, "evidence_message_ids": "none"}
    ])
    results = router._parse_response(response, batch)
    assert results[0]["confidence"] >= CONF_MIN, (
        f"Expected ≥{CONF_MIN}, got {results[0]['confidence']}"
    )
    print(f"[PASS] Confidence clamped at min {CONF_MIN}: got {results[0]['confidence']}")


def test_priority_bias_applied():
    """Negative priority_bias should reduce confidence."""
    router = LLMRouter(client=None, model_name="fake")
    bias = -0.20
    batch = [_make_ctx("msg_c03", bias=bias)]
    llm_conf = 0.80  # LLM raw confidence
    response = json.dumps([
        {"message_id": "msg_c03", "action": "mute", "message_type": "promotion",
         "reason": "Test", "confidence": llm_conf, "evidence_message_ids": "none"}
    ])
    results = router._parse_response(response, batch)
    expected = max(CONF_MIN, min(CONF_MAX, llm_conf + bias))
    got = results[0]["confidence"]
    assert abs(got - expected) < 0.001, f"Expected {expected:.3f}, got {got}"
    print(f"[PASS] Priority bias applied: {llm_conf} + ({bias}) = {got}")


def test_invalid_action_defaults_to_digest():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_i01")]
    response = json.dumps([
        {"message_id": "msg_i01", "action": "INVALID_ACTION", "message_type": "urgent",
         "reason": "Test", "confidence": 0.8, "evidence_message_ids": "none"}
    ])
    results = router._parse_response(response, batch)
    assert results[0]["action"] == "digest"
    print("[PASS] Invalid action defaults to 'digest'")


def test_invalid_message_type_defaults_to_unknown():
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx("msg_i02")]
    response = json.dumps([
        {"message_id": "msg_i02", "action": "notify", "message_type": "INVALID_TYPE",
         "reason": "Test", "confidence": 0.8, "evidence_message_ids": "none"}
    ])
    results = router._parse_response(response, batch)
    assert results[0]["message_type"] == "unknown"
    print("[PASS] Invalid message_type defaults to 'unknown'")


def test_batch_of_multiple_messages():
    """Parse a response with 3 messages correctly."""
    router = LLMRouter(client=None, model_name="fake")
    batch = [_make_ctx(f"msg_b0{i}") for i in range(1, 4)]
    response = json.dumps([
        {"message_id": "msg_b01", "action": "notify", "message_type": "urgent",
         "reason": "Urgent", "confidence": 0.85, "evidence_message_ids": "none"},
        {"message_id": "msg_b02", "action": "digest", "message_type": "personal",
         "reason": "Not urgent", "confidence": 0.75, "evidence_message_ids": "none"},
        {"message_id": "msg_b03", "action": "mute", "message_type": "spam",
         "reason": "Spam", "confidence": 0.80, "evidence_message_ids": "none"},
    ])
    results = router._parse_response(response, batch)
    assert len(results) == 3
    assert results[0]["action"] == "notify"
    assert results[1]["action"] == "digest"
    assert results[2]["action"] == "mute"
    print("[PASS] Batch of 3 messages parsed correctly")


if __name__ == "__main__":
    errors = []
    tests = [
        test_extract_json_array_plain,
        test_extract_json_array_with_fences,
        test_extract_json_array_with_preamble,
        test_malformed_json_fallback,
        test_empty_response_fallback,
        test_missing_message_id_in_response_fallback,
        test_valid_response_parsed_correctly,
        test_confidence_clamped_at_max,
        test_confidence_clamped_at_min,
        test_priority_bias_applied,
        test_invalid_action_defaults_to_digest,
        test_invalid_message_type_defaults_to_unknown,
        test_batch_of_multiple_messages,
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
