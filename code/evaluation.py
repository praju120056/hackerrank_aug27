"""
evaluation.py
-------------
Runs the full pipeline on dataset/sample_messages.csv and compares predictions
against the ground-truth labels embedded in that file.

The labels (action, message_type, etc.) are ONLY used for evaluation — never
during inference. The pipeline sees only the message content and context fields.

Usage:
    python code/evaluation.py

Output:
    - Action accuracy
    - Message-type accuracy
    - Confusion matrix (action)
    - Mismatched predictions (message_id, expected, got)
    - Summary statistics
"""

from __future__ import annotations
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from models import OUTPUT_COLUMNS, RoutingPrediction, VALID_ACTIONS, VALID_MESSAGE_TYPES, MediaSummary
from context_builder import ContextBuilder
from rule_engine import RuleEngine
from evidence import retrieve_evidence, evidence_ids_string
from media_processor import MediaProcessor
from llm_router import LLMRouter

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ROUTING_MODEL  = os.getenv("ROUTING_MODEL", "gemini-3.5-flash-lite")
MEDIA_MODEL    = os.getenv("MEDIA_MODEL", ROUTING_MODEL)
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "10"))
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "8.0"))
MEDIA_SLEEP    = float(os.getenv("MEDIA_SLEEP", "3.0"))
MAX_RETRIES    = int(os.getenv("MAX_RETRIES", "5"))

DATASET_DIR = str(REPO_ROOT / "dataset")
EVAL_CHECKPOINT = REPO_ROOT / "eval_checkpoint.json"


def _load_sample_messages() -> tuple[list[dict], dict[str, dict]]:
    """
    Load sample_messages.csv.
    Returns:
        (message_rows, labels)
        where message_rows has the same schema as messages.csv (label cols stripped),
        and labels maps message_id -> {action, message_type, reason, confidence, evidence_message_ids}.
    """
    path = os.path.join(DATASET_DIR, "sample_messages.csv")
    if not os.path.exists(path):
        print(f"[ERROR] sample_messages.csv not found at {path}")
        sys.exit(1)

    LABEL_COLS = {"action", "message_type", "reason", "confidence", "evidence_message_ids"}
    MSG_COLS = {
        "message_id", "user_id", "conversation_type", "group_id", "business_id",
        "sender_user_id", "created_at", "message_text", "media_type", "media_id",
        "forwarded_count",
    }

    messages: list[dict] = []
    labels: dict[str, dict] = {}

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mid = row["message_id"]
            labels[mid] = {k: row.get(k, "") for k in LABEL_COLS}
            messages.append({k: row.get(k, "") for k in MSG_COLS})

    return messages, labels


def run_pipeline(messages: list[dict]) -> dict[str, RoutingPrediction]:
    """Run the full inference pipeline on the given messages. Returns predictions keyed by message_id."""
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY not set.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    cb = ContextBuilder(DATASET_DIR)
    rule_engine = RuleEngine(context_builder=cb)
    media_proc = MediaProcessor(client=client, model_name=MEDIA_MODEL, media_sleep=MEDIA_SLEEP)
    llm_router = LLMRouter(
        client=client,
        model_name=ROUTING_MODEL,
        batch_size=BATCH_SIZE,
        sleep_between_batch=SLEEP_BETWEEN_BATCH,
        max_retries=MAX_RETRIES,
        checkpoint_path=EVAL_CHECKPOINT,
    )
    message_events = cb.message_events

    # Deduplicate media
    unique_media: dict[str, dict] = {}
    for msg in messages:
        mt = msg.get("media_type", "") or ""
        mid = msg.get("media_id", "") or ""
        if mt in ("image", "voice") and mid and mid not in unique_media:
            unique_media[mid] = msg

    if unique_media:
        print(f"[MEDIA] Processing {len(unique_media)} unique media assets…")
        for media_id, msg in unique_media.items():
            mt = msg["media_type"]
            path = cb.image_path(media_id) if mt == "image" else cb.voice_path(media_id)
            media_proc.process(mt, media_id, path)

    if os.getenv("DEBUG_MEDIA", "").lower() == "true":
        debug_output = []
        for media_id, msg in unique_media.items():
            mt = msg["media_type"]
            summary_obj = media_proc.process(mt, media_id, None)
            debug_output.append({
                "media_id": media_id,
                "media_type": mt,
                "summary": summary_obj.summary,
                "category": summary_obj.category,
                "urgency": summary_obj.urgency
            })
        debug_path = REPO_ROOT / "media_debug.json"
        with open(debug_path, "w", encoding="utf-8") as df:
            json.dump(debug_output, df, ensure_ascii=False, indent=2)
        print(f"[[OK]] Saved debug media info to {debug_path}\n")

    rule_results: dict[str, RoutingPrediction] = {}
    llm_queue: list[dict] = []

    for msg in messages:
        msg_id = msg["message_id"]
        mt = msg.get("media_type", "") or ""
        media_id = msg.get("media_id", "") or ""

        if mt in ("image", "voice") and media_id:
            media_summary = media_proc.process(mt, media_id, None)
        else:
            media_summary = MediaSummary("", "unknown", "low", [], False, 1.0)

        ctx = cb.build_context(msg, media_summary)
        ctx["user_id"] = msg.get("user_id", "")
        ctx["business_id"] = msg.get("business_id", "") or ""

        rule_pred = rule_engine.apply_absolute_rules(ctx)
        if rule_pred is not None:
            rule_results[msg_id] = rule_pred
            continue

        rule_engine.apply_preference_signals(ctx, cb.message_history, message_events)
        evidence = retrieve_evidence(msg, cb.message_history, message_events)
        ctx["evidence"] = evidence
        llm_queue.append(ctx)

    llm_results: dict[str, RoutingPrediction] = {}
    if llm_queue:
        for p in llm_router.route_all(llm_queue):
            llm_results[p.message_id] = p

    results: dict[str, RoutingPrediction] = {}
    for msg in messages:
        mid = msg["message_id"]
        results[mid] = rule_results.get(mid) or llm_results.get(mid) or RoutingPrediction(
            message_id=mid, action="digest", message_type="unknown",
            reason="Fallback.", confidence=0.5, evidence_message_ids="none",
        )

    # Clean up eval checkpoint
    if EVAL_CHECKPOINT.exists():
        EVAL_CHECKPOINT.unlink()

    return results


# ── Evaluation metrics ────────────────────────────────────────────────────────

def confusion_matrix(
    predictions: dict[str, RoutingPrediction],
    labels: dict[str, dict],
    field: str,
    valid_values: set[str],
) -> dict[str, dict[str, int]]:
    """Return a confusion matrix dict[expected][predicted] = count."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for mid, pred in predictions.items():
        expected = labels[mid].get(field, "unknown")
        got = getattr(pred, field) if field != "message_type" else pred.message_type
        if field == "action":
            got = pred.action
        elif field == "message_type":
            got = pred.message_type
        matrix[expected][got] += 1
    return {k: dict(v) for k, v in matrix.items()}


def print_confusion_matrix(matrix: dict[str, dict[str, int]], title: str) -> None:
    all_labels = sorted({k for k in matrix} | {v for row in matrix.values() for v in row})
    col_w = max(len(l) for l in all_labels) + 2
    label_w = col_w

    header = f"{'':>{label_w}}" + "".join(f"{l:>{col_w}}" for l in all_labels)
    print(f"\n{title}")
    print("-" * len(header))
    print(header)
    for expected in all_labels:
        row_str = f"{expected:>{label_w}}"
        row = matrix.get(expected, {})
        for pred in all_labels:
            row_str += f"{row.get(pred, 0):>{col_w}}"
        print(row_str)


def report(
    predictions: dict[str, RoutingPrediction],
    labels: dict[str, dict],
) -> None:
    """Print a full evaluation report."""
    n = len(predictions)
    if n == 0:
        print("[ERROR] No predictions to evaluate.")
        return

    # ── Action accuracy ───────────────────────────────────────────────
    action_correct = sum(
        1 for mid, p in predictions.items()
        if p.action == labels[mid].get("action", "")
    )
    action_acc = action_correct / n

    # ── Message type accuracy ─────────────────────────────────────────
    type_correct = sum(
        1 for mid, p in predictions.items()
        if p.message_type == labels[mid].get("message_type", "")
    )
    type_acc = type_correct / n

    # ── Both correct ──────────────────────────────────────────────────
    both_correct = sum(
        1 for mid, p in predictions.items()
        if p.action == labels[mid].get("action", "")
        and p.message_type == labels[mid].get("message_type", "")
    )

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total evaluated  : {n}")
    print(f"  Action accuracy  : {action_correct}/{n} = {action_acc:.1%}")
    print(f"  Type accuracy    : {type_correct}/{n} = {type_acc:.1%}")
    print(f"  Both correct     : {both_correct}/{n} = {both_correct/n:.1%}")

    # ── Confusion matrices ────────────────────────────────────────────
    action_cm = confusion_matrix(predictions, labels, "action", VALID_ACTIONS)
    print_confusion_matrix(action_cm, "ACTION CONFUSION MATRIX (rows=expected, cols=predicted)")

    # ── Mismatches ────────────────────────────────────────────────────
    mismatches: list[dict] = []
    for mid, pred in predictions.items():
        label = labels[mid]
        if pred.action != label.get("action", "") or pred.message_type != label.get("message_type", ""):
            mismatches.append({
                "message_id": mid,
                "expected_action": label.get("action", ""),
                "predicted_action": pred.action,
                "expected_type": label.get("message_type", ""),
                "predicted_type": pred.message_type,
                "reason": pred.reason,
                "rule_fired": pred.rule_fired,
            })

    if mismatches:
        print(f"\nMISMATCHED PREDICTIONS ({len(mismatches)}):")
        print("-" * 60)
        for m in mismatches:
            flag = " [RULE]" if m["rule_fired"] else ""
            print(
                f"  {m['message_id']:20s} "
                f"action: {m['expected_action']!r} -> {m['predicted_action']!r}  "
                f"type: {m['expected_type']!r} -> {m['predicted_type']!r}"
                f"{flag}"
            )
            print(f"    reason: {m['reason'][:80]}")
    else:
        print("\nAll predictions match labels!")

    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HackerRank Orchestrate — Evaluation on sample_messages.csv")
    print("=" * 60)

    messages, labels = _load_sample_messages()
    print(f"[[OK]] Loaded {len(messages)} labelled messages for evaluation\n")
    print("  NOTE: Labels are only used AFTER inference for comparison.\n")

    predictions = run_pipeline(messages)
    report(predictions, labels)
