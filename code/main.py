"""
main.py
-------
Entry point for the HackerRank Orchestrate: Message Notification Router.

Pipeline stages (in order for each message):
  1. Media Understanding  — Gemini multimodal for images/voice; cached by media_id
  2. Context Building     — Structured dict from all dataset CSVs
  3. Absolute Rules       — Deterministic checks; bypass LLM on match
  4. Preference Signals   — Behavioural bias injected into context
  5. Evidence Retrieval   — Top-3 historical messages by key-match + ranking
  6. LLM Decision         — Gemini Flash; batched 10 msgs/call; checkpoint/resume
  7. Output Validation    — Schema check; repair malformed values
  8. Write output.csv     — Exact contract: one row per message_id in messages.csv

Usage:
    python code/main.py

Config via environment variables (all optional):
    ROUTING_MODEL       Gemini model for decision engine (default: gemini-2.0-flash-lite)
    MEDIA_MODEL         Gemini model for media understanding (default: same as ROUTING_MODEL)
    BATCH_SIZE          Messages per LLM call (default: 10)
    SLEEP_BETWEEN_BATCH Seconds between LLM batches (default: 8.0)
    MEDIA_SLEEP         Seconds between media Gemini calls (default: 3.0)
    MAX_RETRIES         Per-batch retry attempts on 429 errors (default: 5)
"""

from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from models import OUTPUT_COLUMNS, RoutingPrediction, VALID_ACTIONS, VALID_MESSAGE_TYPES
from context_builder import ContextBuilder
from rule_engine import RuleEngine
from evidence import retrieve_evidence, evidence_ids_string
from media_processor import MediaProcessor
from llm_router import LLMRouter

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR    = str(REPO_ROOT / "dataset")
OUTPUT_PATH    = REPO_ROOT / "output.csv"
CHECKPOINT_PATH = REPO_ROOT / "output_checkpoint.json"

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
ROUTING_MODEL       = os.getenv("ROUTING_MODEL", "gemini-3.5-flash-lite")
MEDIA_MODEL         = os.getenv("MEDIA_MODEL", ROUTING_MODEL)
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "10"))
SLEEP_BETWEEN_BATCH = float(os.getenv("SLEEP_BETWEEN_BATCH", "8.0"))
MEDIA_SLEEP         = float(os.getenv("MEDIA_SLEEP", "3.0"))
MAX_RETRIES         = int(os.getenv("MAX_RETRIES", "5"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_messages(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_output(predictions: list[RoutingPrediction]) -> None:
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for p in predictions:
            writer.writerow(p.to_csv_row())


def _validate_final(p: RoutingPrediction) -> RoutingPrediction:
    """Last-resort validation and repair before writing output.csv."""
    action = p.action if p.action in VALID_ACTIONS else "digest"
    mtype = p.message_type if p.message_type in VALID_MESSAGE_TYPES else "unknown"
    reason = p.reason.strip() or "No reason provided."
    conf = max(0.0, min(1.0, float(p.confidence)))
    eids = p.evidence_message_ids.strip() or "none"
    return RoutingPrediction(
        message_id=p.message_id,
        action=action,
        message_type=mtype,
        reason=reason,
        confidence=round(conf, 3),
        evidence_message_ids=eids,
        rule_fired=p.rule_fired,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  HackerRank Orchestrate — Message Notification Router")
    print("=" * 60)

    # ── Validate API key ──────────────────────────────────────────────
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set. Add it to .env and retry.")
        sys.exit(1)

    print(f"[[OK]] Routing model : {ROUTING_MODEL}")
    print(f"[[OK]] Media model   : {MEDIA_MODEL}")
    print(f"[[OK]] Batch size    : {BATCH_SIZE}  |  inter-batch sleep: {SLEEP_BETWEEN_BATCH}s")

    # ── Initialise Gemini client ──────────────────────────────────────
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Initialise pipeline modules ───────────────────────────────────
    cb = ContextBuilder(DATASET_DIR)
    rule_engine = RuleEngine(context_builder=cb)
    media_proc = MediaProcessor(client=client, model_name=MEDIA_MODEL, media_sleep=MEDIA_SLEEP)
    llm_router = LLMRouter(
        client=client,
        model_name=ROUTING_MODEL,
        batch_size=BATCH_SIZE,
        sleep_between_batch=SLEEP_BETWEEN_BATCH,
        max_retries=MAX_RETRIES,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # ── Load messages ─────────────────────────────────────────────────
    messages = _load_messages(os.path.join(DATASET_DIR, "messages.csv"))
    message_events = cb.message_events
    print(f"[[OK]] Loaded {len(messages)} messages | {len(cb.message_history)} history rows\n")

    # ── Collect unique media assets to process ────────────────────────
    # Pre-pass: deduplicate media by media_id before processing.
    unique_media: dict[str, dict] = {}
    for msg in messages:
        mt = msg.get("media_type", "") or ""
        mid = msg.get("media_id", "") or ""
        if mt in ("image", "voice") and mid and mid not in unique_media:
            unique_media[mid] = msg

    if unique_media:
        print(f"[MEDIA] {len(unique_media)} unique media assets to process…")
        for media_id, msg in unique_media.items():
            mt = msg["media_type"]
            path = cb.image_path(media_id) if mt == "image" else cb.voice_path(media_id)
            media_proc.process(mt, media_id, path)
        print(f"[MEDIA] Done. Cache size: {media_proc.cache_size}\n")

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

    # ── Process each message ──────────────────────────────────────────
    rule_results: dict[str, RoutingPrediction] = {}
    llm_queue: list[dict] = []

    for i, msg in enumerate(messages):
        msg_id = msg["message_id"]
        mt = msg.get("media_type", "") or ""
        media_id = msg.get("media_id", "") or ""

        # Retrieve media summary from cache (free — already processed above)
        if mt in ("image", "voice") and media_id:
            media_summary = media_proc.process(mt, media_id, None)  # cache hit
        else:
            from models import MediaSummary
            media_summary = MediaSummary("", "unknown", "low", [], False, 1.0)

        # Build structured context
        ctx = cb.build_context(msg, media_summary)
        ctx["user_id"] = msg["user_id"]           # needed by rule engine helpers
        ctx["business_id"] = msg.get("business_id", "") or ""

        # Phase 1: Absolute rules
        rule_pred = rule_engine.apply_absolute_rules(ctx)
        if rule_pred is not None:
            print(f"  [{i+1:3d}/{len(messages)}] {msg_id}  -> RULE [{rule_pred.action.upper()}]")
            rule_results[msg_id] = rule_pred
            continue

        # Phase 2: Preference signals
        rule_engine.apply_preference_signals(ctx, cb.message_history, message_events)

        # Phase 3: Evidence retrieval
        evidence = retrieve_evidence(msg, cb.message_history, message_events)
        ctx["evidence"] = evidence

        bias = ctx["preference_signals"]["priority_bias"]
        print(
            f"  [{i+1:3d}/{len(messages)}] {msg_id}  -> LLM "
            f"(bias={bias:+.2f}, evidence={len(evidence)})"
        )
        llm_queue.append(ctx)

    print(
        f"\n[[OK]] Absolute rules: {len(rule_results)}  |  LLM queue: {len(llm_queue)}\n"
    )

    # ── LLM batch routing ─────────────────────────────────────────────
    llm_results: dict[str, RoutingPrediction] = {}
    if llm_queue:
        preds = llm_router.route_all(llm_queue)
        for p in preds:
            llm_results[p.message_id] = p

    # ── Merge, validate, and write output ─────────────────────────────
    final_predictions: list[RoutingPrediction] = []
    missing: list[str] = []

    for msg in messages:
        mid = msg["message_id"]
        if mid in rule_results:
            pred = rule_results[mid]
        elif mid in llm_results:
            pred = llm_results[mid]
        else:
            print(f"  [WARN] No result for {mid} — using fallback")
            missing.append(mid)
            pred = RoutingPrediction(
                message_id=mid,
                action="digest",
                message_type="unknown",
                reason="Routing fallback: no prediction produced.",
                confidence=0.5,
                evidence_message_ids="none",
            )

        final_predictions.append(_validate_final(pred))

    _write_output(final_predictions)

    # ── Sanity check ──────────────────────────────────────────────────
    expected = {m["message_id"] for m in messages}
    written = {p.message_id for p in final_predictions}
    if expected != written:
        print(f"[ERROR] ID mismatch — expected {len(expected)}, wrote {len(written)}")
    else:
        print(f"[[OK]] {len(final_predictions)} rows written to {OUTPUT_PATH}")
    if missing:
        print(f"[WARN] Fallback used for: {missing}")

    # Clean up checkpoint on success
    if not missing and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("[[OK]] Checkpoint removed (run complete)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
