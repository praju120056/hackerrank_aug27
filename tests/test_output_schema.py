"""
test_output_schema.py
---------------------
Validates output.csv structure and content against the submission contract.
Run this AFTER python code/main.py has produced output.csv.
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "code"))

from models import VALID_ACTIONS, VALID_MESSAGE_TYPES, OUTPUT_COLUMNS

OUTPUT_PATH = REPO_ROOT / "output.csv"
MESSAGES_PATH = REPO_ROOT / "dataset" / "messages.csv"


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_output_exists():
    assert OUTPUT_PATH.exists(), f"output.csv not found at {OUTPUT_PATH}"
    print("[PASS] output.csv exists")


def test_header_order():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == OUTPUT_COLUMNS, (
        f"Column order mismatch.\n  Expected: {OUTPUT_COLUMNS}\n  Got:      {header}"
    )
    print("[PASS] Column order matches specification")


def test_one_row_per_message_id():
    messages = _load(MESSAGES_PATH)
    output = _load(OUTPUT_PATH)
    expected = {m["message_id"] for m in messages}
    got_ids = [r["message_id"] for r in output]

    missing = expected - set(got_ids)
    assert not missing, f"Missing message_ids: {missing}"

    dups = {mid for mid in got_ids if got_ids.count(mid) > 1}
    assert not dups, f"Duplicate message_ids: {dups}"

    assert len(output) == len(expected), (
        f"Row count: expected {len(expected)}, got {len(output)}"
    )
    print(f"[PASS] Exactly one row per message_id ({len(output)} rows)")


def test_valid_action_values():
    for row in _load(OUTPUT_PATH):
        assert row["action"] in VALID_ACTIONS, (
            f"Invalid action '{row['action']}' for {row['message_id']}"
        )
    print(f"[PASS] All action values are valid: {VALID_ACTIONS}")


def test_valid_message_type_values():
    for row in _load(OUTPUT_PATH):
        assert row["message_type"] in VALID_MESSAGE_TYPES, (
            f"Invalid message_type '{row['message_type']}' for {row['message_id']}"
        )
    print("[PASS] All message_type values are valid")


def test_confidence_in_range():
    for row in _load(OUTPUT_PATH):
        conf_str = row.get("confidence", "")
        assert conf_str != "", f"Empty confidence for {row['message_id']}"
        conf = float(conf_str)
        assert 0.0 <= conf <= 1.0, f"Confidence out of range for {row['message_id']}: {conf}"
    print("[PASS] All confidence values in [0, 1]")


def test_non_empty_reason():
    for row in _load(OUTPUT_PATH):
        assert row.get("reason", "").strip(), f"Blank reason for {row['message_id']}"
    print("[PASS] All reason fields are non-empty")


def test_evidence_ids_format():
    for row in _load(OUTPUT_PATH):
        eids = row.get("evidence_message_ids", "").strip()
        assert eids, f"Empty evidence_message_ids for {row['message_id']}"
        if eids != "none":
            parts = [p.strip() for p in eids.split(";")]
            for p in parts:
                assert p, f"Empty part in evidence_message_ids for {row['message_id']}: {eids}"
    print("[PASS] All evidence_message_ids values are valid")


if __name__ == "__main__":
    errors = []
    tests = [
        test_output_exists, test_header_order, test_one_row_per_message_id,
        test_valid_action_values, test_valid_message_type_values,
        test_confidence_in_range, test_non_empty_reason, test_evidence_ids_format,
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
