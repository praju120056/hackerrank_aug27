"""
llm_router.py
-------------
Decision Engine: batch Gemini Flash calls for message routing.

- Batches 5 messages per API call
- Sleeps 4 seconds between batches (free-tier: 15 RPM, 1500 RPD)
- Exponential backoff on rate-limit errors (max 3 retries)
- Falls back to digest/0.5/none on JSON parse failure
- Clamps final confidence to [0.55, 0.95]
"""

from __future__ import annotations
import json
import time
import os
import re

BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 4  # seconds
MAX_RETRIES = 3

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}

CONF_MIN = 0.55
CONF_MAX = 0.95


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a WhatsApp notification routing system. Your job is to decide, for each incoming message, whether to notify the user immediately, digest it for later, or mute it.

ROUTING RULES:
- notify: message is urgent, time-sensitive, personally relevant, or requires immediate action
- digest: message is useful or safe but can wait; no urgent action needed
- mute: low-value, repetitive, unwanted, promotional (opted-out), suspicious, or unsafe

ALLOWED action values: notify, digest, mute
ALLOWED message_type values: personal, urgent, event, payment, business_update, promotion, greeting, forward, spam, scam, unknown

PERSONALIZATION: Use the user context, group context, business context, preference_signals, and evidence to make a decision personalized to this specific user. The same message may be notify for one user and mute for another.

PREFERENCE SIGNALS: The priority_bias and reason_hint are pre-computed behavioral signals. Use them to inform your decision but do not be bound by them — a biased message can still be notify if the content is genuinely urgent.

CONFIDENCE CALIBRATION:
- Start with your raw confidence (0.0–1.0)
- Add priority_bias from the context (may be negative)
- Clamp final value between 0.55 and 0.95
- Higher confidence = clearer signal; use 0.75–0.85 for most decisions

EVIDENCE: If you used historical messages in your reasoning, list their message_ids semicolon-separated. Otherwise write "none".

OUTPUT FORMAT: Return a JSON array. One object per message. Exactly these fields:
[
  {
    "message_id": "msg_xxx",
    "action": "notify|digest|mute",
    "message_type": "...",
    "reason": "One sentence. Match the style of the examples below.",
    "confidence": 0.00,
    "evidence_message_ids": "message_0001;message_0002 or none"
  }
]

Do not include any text outside the JSON array."""


def _load_few_shot_examples() -> str:
    """Load few-shot examples from prompts/few_shot_examples.txt."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, "prompts", "few_shot_examples.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    return (
        "\n\nLABELLED EXAMPLES (do not route these, use them to learn the style and calibration):\n\n"
        + raw
    )


FEW_SHOT_BLOCK = _load_few_shot_examples()
FULL_SYSTEM_PROMPT = SYSTEM_PROMPT + FEW_SHOT_BLOCK


# ── Public class ──────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Usage:
        router = LLMRouter(client, model_name="gemini-2.0-flash-001")
        results = router.route_batch(contexts)  # list of ctx dicts
    """

    def __init__(self, client, model_name: str = "gemini-2.0-flash-001"):
        self.client = client
        self.model_name = model_name

    def route_all(self, contexts: list[dict]) -> list[dict]:
        """
        Route all contexts in batches. Returns a list of result dicts in the
        same order as contexts.
        """
        results: list[dict] = []
        batches = [
            contexts[i: i + BATCH_SIZE]
            for i in range(0, len(contexts), BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            if batch_idx > 0:
                print(f"  [LLM] Sleeping {SLEEP_BETWEEN_BATCHES}s between batches…")
                time.sleep(SLEEP_BETWEEN_BATCHES)

            print(
                f"  [LLM] Batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} messages)…"
            )
            batch_results = self._route_batch_with_retry(batch)
            results.extend(batch_results)

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _route_batch_with_retry(self, batch: list[dict]) -> list[dict]:
        """Attempt batch routing with exponential backoff on rate-limit errors."""
        from google.genai import types

        user_prompt = self._build_user_prompt(batch)
        msg_ids = [ctx["message_id"] for ctx in batch]

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=FULL_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                raw_text = response.text or ""
                parsed = self._parse_response(raw_text, batch)
                return parsed

            except Exception as exc:
                err_str = str(exc).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "quota" in err_str
                    or "rate" in err_str
                    or "resource" in err_str
                )
                if is_rate_limit and attempt < MAX_RETRIES:
                    wait = (2 ** attempt) * 10  # 10s, 20s, 40s
                    print(
                        f"  [LLM] Rate limit hit (attempt {attempt + 1}), "
                        f"waiting {wait}s…"
                    )
                    time.sleep(wait)
                    continue
                else:
                    print(f"  [LLM] Error (attempt {attempt + 1}): {exc}")
                    break

        # Fallback for all messages in this batch
        return [self._fallback(ctx) for ctx in batch]

    def _build_user_prompt(self, batch: list[dict]) -> str:
        """Serialise the batch of context dicts into a compact JSON prompt."""
        items = []
        for ctx in batch:
            # Strip heavy fields that bloat the prompt but are redundant
            compact = {
                "message_id": ctx.get("message_id"),
                "message_text": (ctx.get("message_text") or "")[:800],
                "media_type": ctx.get("media_type", ""),
                "media_summary": ctx.get("media_summary", ""),
                "conversation_type": ctx.get("conversation_type"),
                "forwarded_count": ctx.get("forwarded_count", 0),
                "created_at": ctx.get("created_at", ""),
                "user": ctx.get("user"),
                "group": ctx.get("group"),
                "business": ctx.get("business"),
                "preference_signals": ctx.get("preference_signals"),
                "evidence": ctx.get("evidence", []),
                "fatigue": ctx.get("fatigue"),
            }
            items.append(compact)

        return (
            "Route each of these messages. Return ONLY a JSON array:\n\n"
            + json.dumps(items, ensure_ascii=False, indent=2)
        )

    def _parse_response(self, raw: str, batch: list[dict]) -> list[dict]:
        """Parse the LLM JSON array response. Falls back per-message on errors."""
        # Extract JSON array from response (strip markdown fences if present)
        json_str = _extract_json_array(raw)

        try:
            parsed_list = json.loads(json_str)
        except Exception as exc:
            print(f"  [LLM] JSON parse failed: {exc}. Falling back to all digest.")
            return [self._fallback(ctx) for ctx in batch]

        if not isinstance(parsed_list, list):
            return [self._fallback(ctx) for ctx in batch]

        # Build a lookup by message_id
        result_map: dict[str, dict] = {}
        for item in parsed_list:
            mid = item.get("message_id", "")
            if mid:
                result_map[mid] = item

        results = []
        for ctx in batch:
            mid = ctx["message_id"]
            item = result_map.get(mid)
            if item is None:
                results.append(self._fallback(ctx))
                continue

            # Validate and sanitise
            action = item.get("action", "digest")
            if action not in VALID_ACTIONS:
                action = "digest"

            msg_type = item.get("message_type", "unknown")
            if msg_type not in VALID_TYPES:
                msg_type = "unknown"

            reason = str(item.get("reason", "Unable to determine routing."))[:300]

            # Confidence: apply priority_bias then clamp
            raw_conf = _safe_float(item.get("confidence", 0.7))
            bias = _safe_float(
                (ctx.get("preference_signals") or {}).get("priority_bias", 0.0)
            )
            final_conf = max(CONF_MIN, min(CONF_MAX, raw_conf + bias))

            evidence_ids = str(item.get("evidence_message_ids", "none"))

            results.append(
                {
                    "message_id": mid,
                    "action": action,
                    "message_type": msg_type,
                    "reason": reason,
                    "confidence": round(final_conf, 3),
                    "evidence_message_ids": evidence_ids,
                }
            )

        return results

    def _fallback(self, ctx: dict) -> dict:
        return {
            "message_id": ctx["message_id"],
            "action": "digest",
            "message_type": "unknown",
            "reason": "Routing fallback: unable to determine action from available context.",
            "confidence": 0.5,
            "evidence_message_ids": "none",
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json_array(text: str) -> str:
    """Strip markdown fences and extract the JSON array substring."""
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    text = text.strip()

    # Find the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start: end + 1]
    return text


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default
