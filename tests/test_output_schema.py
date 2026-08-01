"""
test_output_schema.py
---------------------
Validates the structure and content of output.csv.
"""

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_PATH = REPO_ROOT / "output.csv"
MESSAGES_PATH = REPO_ROOT / "dataset" / "messages.csv"

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}
REQUIRED_COLUMNS = [
    "message_id", "action", "message_type", "reason",
    "confidence", "evidence_message_ids",
]


def load_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_output_exists():
    assert OUTPUT_PATH.exists(), f"output.csv not found at {OUTPUT_PATH}"
    print("[PASS] output.csv exists")


def test_required_columns():
    rows = load_csv(OUTPUT_PATH)
    assert rows, "output.csv is empty"
    cols = list(rows[0].keys())
    for col in REQUIRED_COLUMNS:
        assert col in cols, f"Missing column: {col}"
    print(f"[PASS] All required columns present: {REQUIRED_COLUMNS}")


def test_column_order():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == REQUIRED_COLUMNS, (
        f"Column order mismatch.\n  Expected: {REQUIRED_COLUMNS}\n  Got:      {header}"
    )
    print("[PASS] Column order matches specification")


def test_one_row_per_message_id():
    messages = load_csv(MESSAGES_PATH)
    output = load_csv(OUTPUT_PATH)
    expected_ids = {m["message_id"] for m in messages}
    output_ids = [r["message_id"] for r in output]

    missing = expected_ids - set(output_ids)
    assert not missing, f"Missing message_ids: {missing}"

    duplicates = {mid for mid in output_ids if output_ids.count(mid) > 1}
    assert not duplicates, f"Duplicate message_ids: {duplicates}"

    assert len(output) == len(expected_ids), (
        f"Row count mismatch: expected {len(expected_ids)}, got {len(output)}"
    )
    print(f"[PASS] Exactly one row per message_id ({len(output)} rows)")


def test_confidence_range():
    rows = load_csv(OUTPUT_PATH)
    for row in rows:
        conf_str = row.get("confidence", "")
        assert conf_str != "", f"Empty confidence for {row['message_id']}"
        conf = float(conf_str)
        assert 0.0 <= conf <= 1.0, (
            f"Confidence out of range for {row['message_id']}: {conf}"
        )
    print("[PASS] All confidence values in [0, 1]")


def test_valid_action_values():
    rows = load_csv(OUTPUT_PATH)
    for row in rows:
        action = row.get("action", "")
        assert action in VALID_ACTIONS, (
            f"Invalid action '{action}' for {row['message_id']}"
        )
    print(f"[PASS] All action values valid: {VALID_ACTIONS}")


def test_valid_message_type_values():
    rows = load_csv(OUTPUT_PATH)
    for row in rows:
        mt = row.get("message_type", "")
        assert mt in VALID_TYPES, (
            f"Invalid message_type '{mt}' for {row['message_id']}"
        )
    print(f"[PASS] All message_type values valid")


def test_no_blank_reason():
    rows = load_csv(OUTPUT_PATH)
    for row in rows:
        reason = row.get("reason", "").strip()
        assert reason, f"Blank reason for {row['message_id']}"
    print("[PASS] All reason fields are non-empty")


def test_evidence_ids_format():
    rows = load_csv(OUTPUT_PATH)
    for row in rows:
        eids = row.get("evidence_message_ids", "").strip()
        assert eids != "", f"Empty evidence_message_ids for {row['message_id']}"
        # Must be "none" or semicolon-separated non-empty strings
        if eids != "none":
            parts = [p.strip() for p in eids.split(";")]
            for p in parts:
                assert p, (
                    f"Empty part in evidence_message_ids for {row['message_id']}: {eids}"
                )
    print("[PASS] All evidence_message_ids values are valid")


if __name__ == "__main__":
    errors = []
    tests = [
        test_output_exists,
        test_required_columns,
        test_column_order,
        test_one_row_per_message_id,
        test_confidence_range,
        test_valid_action_values,
        test_valid_message_type_values,
        test_no_blank_reason,
        test_evidence_ids_format,
    ]
    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            errors.append(test_fn.__name__)
        except Exception as e:
            print(f"[ERROR] {test_fn.__name__}: {e}")
            errors.append(test_fn.__name__)

    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s): {errors}")
        sys.exit(1)
    else:
        print(f"ALL TESTS PASSED ({len(tests)} tests)")
