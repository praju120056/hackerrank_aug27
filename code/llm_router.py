"""
llm_router.py
-------------
Decision Engine: batched Gemini calls for message routing.

Key improvements over v1:
- Uses gemini-2.0-flash-lite (30 RPM free tier) by default
- Parses RetryInfo from 429 errors to use the API-suggested retry delay
- Configurable batch size (default 10) and inter-batch sleep (default 8 s)
- Checkpoint/resume: partial results are saved to disk after every batch
- Compact system prompt (no per-call few-shot block) reduces token usage
- Validates and repairs every LLM response before accepting it
- Raises KeyboardInterrupt gracefully by saving checkpoint before exit
"""

from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path

from models import (
    RoutingPrediction,
    VALID_ACTIONS,
    VALID_MESSAGE_TYPES,
    CONF_MIN,
    CONF_MAX,
)


# ── System prompt ─────────────────────────────────────────────────────────────
# Kept short to minimise tokens consumed per call.
# Few-shot examples are loaded from disk once at import time.

_SYSTEM_PROMPT = """\
You are a WhatsApp notification routing engine.

For each incoming message you receive, decide the routing action.

ACTIONS:
- notify  -> message is urgent, time-sensitive, personally relevant, or requires immediate action
- digest  -> message is useful or safe but can wait; no urgent action needed
- mute    -> low-value, repetitive, unwanted, promotional (opted-out), suspicious, or unsafe

MESSAGE TYPES (pick the single best fit):
personal | urgent | event | payment | business_update | promotion | greeting | forward | spam | scam | unknown

PERSONALIZATION:
Use all provided context fields. The same message text may warrant different actions for different users based on their history, relationship, and preferences.

PREFERENCE SIGNALS:
priority_bias is a pre-computed float (negative = lower priority). reason_hint explains why.
These inform your decision but do NOT override it — urgent content can still be notify even with a negative bias.

CONFIDENCE:
Assign your raw confidence (0.0–1.0). The pipeline will apply priority_bias and clamp.

EVIDENCE:
If historical messages informed your decision, list their IDs (semicolon-separated). Otherwise write "none".

OUTPUT FORMAT:
Return a JSON array only. No markdown, no prose. Exactly one object per message_id provided.

[
  {
    "message_id": "...",
    "action": "notify|digest|mute",
    "message_type": "...",
    "reason": "One concise sentence explaining the decision.",
    "confidence": 0.00,
    "evidence_message_ids": "message_001;message_002 or none"
  }
]"""


# ── Few-shot examples ─────────────────────────────────────────────────────────

def _load_few_shot() -> str:
    this_dir = Path(__file__).parent
    path = this_dir / "prompts" / "few_shot_examples.txt"
    if not path.exists():
        return ""
    return (
        "\n\nLABELLED EXAMPLES — learn style and calibration, do NOT route these:\n\n"
        + path.read_text(encoding="utf-8").strip()
    )


_FEW_SHOT = _load_few_shot()
_FULL_SYSTEM = _SYSTEM_PROMPT + _FEW_SHOT


# ── RetryInfo parser ──────────────────────────────────────────────────────────

def _extract_retry_seconds(exc: Exception, default: int = 65) -> int:
    """
    Parse the retryDelay field from a Gemini 429 RetryInfo response.
    Returns delay_seconds + 5s buffer, or `default` if not found.

    The error details contain something like:
        {'@type': '.../RetryInfo', 'retryDelay': '32s'}
    """
    try:
        text = str(exc)
        # Try both quote styles: 'retryDelay': '32s' and "retryDelay": "32s"
        for pattern in (
            r"['\"]retryDelay['\"]\s*:\s*['\"](\d+)s['\"]",
            r"retry in\s+(\d+(?:\.\d+)?)s",
        ):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return int(float(m.group(1))) + 5
    except Exception:
        pass
    return default


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "quota" in s or "resource_exhausted" in s or "rate" in s


# ── Output validation & repair ────────────────────────────────────────────────

def _validate_and_repair(item: dict, ctx: dict) -> RoutingPrediction:
    """
    Validate one parsed LLM response object against the output contract.
    Repairs invalid enum values (action -> digest, message_type -> unknown).
    Applies priority_bias and clamps confidence.
    """
    mid = str(item.get("message_id", ctx["message_id"]))

    action = str(item.get("action", "digest")).strip().lower()
    if action not in VALID_ACTIONS:
        action = "digest"

    mtype = str(item.get("message_type", "unknown")).strip().lower()
    if mtype not in VALID_MESSAGE_TYPES:
        mtype = "unknown"

    reason = str(item.get("reason", "Routing decision."))[:300].strip()
    if not reason:
        reason = "Routing decision."

    # Confidence: parse, apply bias, clamp
    raw_conf = float(item.get("confidence", 0.7))
    bias = float((ctx.get("preference_signals") or {}).get("priority_bias", 0.0))
    conf = max(CONF_MIN, min(CONF_MAX, raw_conf + bias))

    eids = str(item.get("evidence_message_ids", "none")).strip()
    if not eids:
        eids = "none"

    return RoutingPrediction(
        message_id=mid,
        action=action,
        message_type=mtype,
        reason=reason,
        confidence=round(conf, 3),
        evidence_message_ids=eids,
        rule_fired=False,
    )


def _fallback(ctx: dict) -> RoutingPrediction:
    return RoutingPrediction(
        message_id=ctx["message_id"],
        action="digest",
        message_type="unknown",
        reason="Routing fallback: unable to obtain a valid LLM decision.",
        confidence=0.5,
        evidence_message_ids="none",
        rule_fired=False,
    )


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json_array(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


# ── Main router class ─────────────────────────────────────────────────────────

class LLMRouter:
    """
    Routes all messages through Gemini in batches with automatic rate-limit handling.

    Configuration (all overridable via env vars, but constructor params take priority):
        model_name          Gemini model ID
        batch_size          Messages per API call (default 10)
        sleep_between_batch Seconds to sleep between batches (default 8)
        max_retries         Per-batch retry attempts on rate-limit errors (default 5)
        checkpoint_path     Where to save/resume partial results (or None to disable)
    """

    def __init__(
        self,
        client,
        model_name: str,
        batch_size: int = 10,
        sleep_between_batch: float = 8.0,
        max_retries: int = 5,
        checkpoint_path: Path | None = None,
    ):
        self._client = client
        self._model = model_name
        self._batch_size = batch_size
        self._sleep = sleep_between_batch
        self._max_retries = max_retries
        self._checkpoint_path = checkpoint_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_all(self, contexts: list[dict]) -> list[RoutingPrediction]:
        """
        Route all context dicts in batches.
        Returns a list of RoutingPrediction objects in the same order.
        Falls back per-batch on unrecoverable errors.
        Saves a checkpoint after every batch.
        """
        # Load checkpoint to resume partial runs
        completed: dict[str, RoutingPrediction] = self._load_checkpoint()
        if completed:
            print(f"[LLM] Resuming from checkpoint ({len(completed)} already done)")

        # Filter out already-completed messages
        pending = [c for c in contexts if c["message_id"] not in completed]
        batches = [
            pending[i : i + self._batch_size]
            for i in range(0, len(pending), self._batch_size)
        ]

        total = len(batches)
        try:
            for i, batch in enumerate(batches):
                if i > 0:
                    print(f"  [LLM] Sleeping {self._sleep}s …")
                    time.sleep(self._sleep)

                print(f"  [LLM] Batch {i + 1}/{total} ({len(batch)} messages) …")
                results = self._route_batch(batch)
                for pred in results:
                    completed[pred.message_id] = pred
                self._save_checkpoint(completed)
        except KeyboardInterrupt:
            print("\n[LLM] Interrupted — checkpoint saved. Re-run to resume.")
            raise

        # Reconstruct in original order
        ordered: list[RoutingPrediction] = []
        for ctx in contexts:
            mid = ctx["message_id"]
            ordered.append(completed.get(mid) or _fallback(ctx))
        return ordered

    # ------------------------------------------------------------------
    # Batch routing with retry
    # ------------------------------------------------------------------

    def _route_batch(self, batch: list[dict]) -> list[RoutingPrediction]:
        """Send one batch to Gemini, retry on rate-limit errors using RetryInfo delay."""
        from google.genai import types

        user_prompt = self._build_prompt(batch)

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=_FULL_SYSTEM,
                        temperature=0.0,
                        max_output_tokens=2048,
                    ),
                )
                raw = response.text or ""
                return self._parse_response(raw, batch)

            except KeyboardInterrupt:
                raise

            except Exception as exc:
                if _is_rate_limit(exc) and attempt < self._max_retries:
                    delay = _extract_retry_seconds(exc)
                    print(
                        f"  [LLM] Rate-limit (attempt {attempt + 1}/{self._max_retries}), "
                        f"retrying in {delay}s …"
                    )
                    time.sleep(delay)
                    continue

                # Non-rate-limit error or retries exhausted
                print(f"  [LLM] Failed after {attempt + 1} attempt(s): {exc}")
                break

        return [_fallback(ctx) for ctx in batch]

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, batch: list[dict]) -> str:
        """Serialise the batch as a compact JSON array for the user message."""
        items = []
        for ctx in batch:
            items.append({
                "message_id": ctx["message_id"],
                "text": ctx.get("text", "")[:600],
                "media_type": ctx.get("media_type", ""),
                "media": ctx.get("media"),
                "conversation_type": ctx.get("conversation_type"),
                "forwarded_count": ctx.get("forwarded_count", 0),
                "created_at": ctx.get("created_at", ""),
                "user": ctx.get("user"),
                "group": ctx.get("group"),
                "business": ctx.get("business"),
                "preference_signals": ctx.get("preference_signals"),
                "evidence": ctx.get("evidence", []),
                "fatigue": ctx.get("fatigue"),
            })
        return (
            "Route each message. Return ONLY a JSON array:\n\n"
            + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, batch: list[dict]) -> list[RoutingPrediction]:
        """Parse and validate the JSON array response from Gemini."""
        json_str = _extract_json_array(raw)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            print(f"  [LLM] JSON parse error: {exc}")
            return [_fallback(ctx) for ctx in batch]

        if not isinstance(parsed, list):
            print("  [LLM] Response is not a list; falling back.")
            return [_fallback(ctx) for ctx in batch]

        # Build lookup by message_id
        result_map: dict[str, dict] = {}
        for item in parsed:
            mid = item.get("message_id", "")
            if mid:
                result_map[mid] = item

        results: list[RoutingPrediction] = []
        for ctx in batch:
            mid = ctx["message_id"]
            item = result_map.get(mid)
            if item is None:
                print(f"  [LLM] No result for {mid}; using fallback.")
                results.append(_fallback(ctx))
            else:
                try:
                    results.append(_validate_and_repair(item, ctx))
                except Exception as exc:
                    print(f"  [LLM] Validation error for {mid}: {exc}; using fallback.")
                    results.append(_fallback(ctx))
        return results

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, completed: dict[str, RoutingPrediction]) -> None:
        if self._checkpoint_path is None:
            return
        try:
            data = {
                mid: {
                    "message_id": p.message_id,
                    "action": p.action,
                    "message_type": p.message_type,
                    "reason": p.reason,
                    "confidence": p.confidence,
                    "evidence_message_ids": p.evidence_message_ids,
                    "rule_fired": p.rule_fired,
                }
                for mid, p in completed.items()
            }
            self._checkpoint_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"  [LLM] Checkpoint save failed: {exc}")

    def _load_checkpoint(self) -> dict[str, RoutingPrediction]:
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return {}
        try:
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            return {
                mid: RoutingPrediction(**row)
                for mid, row in data.items()
            }
        except Exception as exc:
            print(f"  [LLM] Checkpoint load failed: {exc}")
            return {}
