"""
main.py
-------
Entry point for the HackerRank Orchestrate: Message Notification Router.

Pipeline:
  1. Load all dataset CSVs (ContextBuilder)
  2. For each message in messages.csv:
     a. Process media (image/voice) → media_summary
     b. Build structured context dict
     c. Apply absolute rules (rule_engine) → if fired, use result directly
     d. If no absolute rule, apply preference signals
     e. Retrieve top-3 evidence from message_history
     f. Queue for LLM batch routing
  3. Batch route remaining messages via Gemini Flash (LLMRouter)
  4. Merge rule-fired results + LLM results
  5. Write output.csv

Usage:
    python code/main.py

Reads GEMINI_API_KEY from environment (set in .env file).
"""

import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives in code/ ; repo root is one level up.
REPO_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR = str(REPO_ROOT / "dataset")
OUTPUT_PATH = str(REPO_ROOT / "output.csv")

# Add code/ to sys.path so sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from context_builder import ContextBuilder
from rule_engine import RuleEngine
from evidence import retrieve_evidence, evidence_ids_string
from media_processor import process_media
from llm_router import LLMRouter, CONF_MIN, CONF_MAX

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(REPO_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY is not set. Add it to .env and retry.")
    sys.exit(1)

# Gemini model names
ROUTING_MODEL = "gemini-2.0-flash-001"   # Decision Engine (batched 5 msgs/call)
MEDIA_MODEL = "gemini-2.0-flash-001"     # Media description (inline image/audio)

OUTPUT_COLUMNS = [
    "message_id", "action", "message_type", "reason",
    "confidence", "evidence_message_ids",
]


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def main():
    print("=" * 60)
    print("  HackerRank Orchestrate — Message Notification Router")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Initialise Gemini client
    # ------------------------------------------------------------------
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[✓] Gemini client initialised (models: {ROUTING_MODEL})")

    # ------------------------------------------------------------------
    # 2. Load all CSVs
    # ------------------------------------------------------------------
    cb = ContextBuilder(DATASET_DIR)
    rule_engine = RuleEngine(context_builder=cb)
    llm_router = LLMRouter(client=client, model_name=ROUTING_MODEL)

    messages_path = os.path.join(DATASET_DIR, "messages.csv")
    with open(messages_path, encoding="utf-8", newline="") as f:
        messages = list(csv.DictReader(f))

    print(f"[✓] Loaded {len(messages)} messages to route")
    print(f"[✓] Loaded {len(cb.message_history)} historical messages")

    # Build message_events lookup once (used by rule_engine + evidence)
    message_events = cb.message_events  # dict keyed by (user_id, message_id)

    # ------------------------------------------------------------------
    # 3. Process each message
    # ------------------------------------------------------------------
    rule_results: dict[str, dict] = {}   # message_id → final output
    llm_queue: list[dict] = []            # contexts to send to LLM

    for i, msg in enumerate(messages):
        mid = msg["message_id"]
        media_type = msg.get("media_type", "") or ""
        media_id = msg.get("media_id", "") or ""

        print(f"  [{i+1}/{len(messages)}] Processing {mid} …", end=" ")

        # ── a. Media processing ──────────────────────────────────────
        media_summary = ""
        if media_type == "image":
            path = cb.get_image_path(media_id)
            media_summary = process_media("image", path, client, MEDIA_MODEL)
        elif media_type == "voice":
            path = cb.get_voice_path(media_id)
            media_summary = process_media("voice", path, client, MEDIA_MODEL)

        # ── b. Build context ─────────────────────────────────────────
        ctx = cb.build_context(msg, media_summary=media_summary)
        # Add user_id directly to ctx for rule_engine helpers
        ctx["user_id"] = msg["user_id"]

        # ── c. Absolute rules ────────────────────────────────────────
        abs_result = rule_engine.apply_absolute_rules(ctx)
        if abs_result is not None:
            print(f"→ RULE [{abs_result['action'].upper()}]")
            rule_results[mid] = abs_result
            continue

        # ── d. Preference signals ────────────────────────────────────
        rule_engine.apply_preference_signals(ctx, cb.message_history, message_events)

        # ── e. Evidence retrieval ────────────────────────────────────
        evidence = retrieve_evidence(msg, cb.message_history, message_events)
        ctx["evidence"] = evidence

        print(
            f"→ LLM queue "
            f"(bias={ctx['preference_signals']['priority_bias']:+.2f}, "
            f"evidence={len(evidence)})"
        )
        llm_queue.append(ctx)

    print(
        f"\n[✓] Absolute rules fired for {len(rule_results)} messages; "
        f"{len(llm_queue)} sent to LLM\n"
    )

    # ------------------------------------------------------------------
    # 4. LLM batch routing
    # ------------------------------------------------------------------
    llm_results_list: list[dict] = []
    if llm_queue:
        print(f"[LLM] Routing {len(llm_queue)} messages in batches of 5…")
        llm_results_list = llm_router.route_all(llm_queue)

    llm_results: dict[str, dict] = {r["message_id"]: r for r in llm_results_list}

    # ------------------------------------------------------------------
    # 5. Merge and write output.csv
    # ------------------------------------------------------------------
    output_rows: list[dict] = []
    missing: list[str] = []

    for msg in messages:
        mid = msg["message_id"]
        if mid in rule_results:
            row = rule_results[mid]
        elif mid in llm_results:
            row = llm_results[mid]
        else:
            print(f"  [WARN] No result for {mid}, using fallback")
            missing.append(mid)
            row = {
                "message_id": mid,
                "action": "digest",
                "message_type": "unknown",
                "reason": "Routing fallback: no result produced.",
                "confidence": 0.5,
                "evidence_message_ids": "none",
            }

        # Ensure confidence is a valid float, clamped to [0,1]
        conf = _safe_float(row.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))

        output_rows.append(
            {
                "message_id": row["message_id"],
                "action": row["action"],
                "message_type": row["message_type"],
                "reason": row["reason"],
                "confidence": round(conf, 3),
                "evidence_message_ids": row.get("evidence_message_ids", "none"),
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n[✓] Wrote {len(output_rows)} rows to {OUTPUT_PATH}")
    if missing:
        print(f"[WARN] Fallback used for: {missing}")

    # ------------------------------------------------------------------
    # 6. Quick sanity check
    # ------------------------------------------------------------------
    expected_ids = {msg["message_id"] for msg in messages}
    written_ids = {row["message_id"] for row in output_rows}
    if expected_ids != written_ids:
        print(f"[ERROR] ID mismatch: expected {len(expected_ids)}, got {len(written_ids)}")
    else:
        print(f"[✓] All {len(expected_ids)} message IDs present in output.csv")
    print("Done.\n")


if __name__ == "__main__":
    main()
