"""
rule_engine.py
--------------
Two-phase rule engine:

1. Absolute Rules: If any rule fires, returns a final output dict immediately
   (skip LLM entirely). First match wins, checked in order.

2. Preference Signals: If no absolute rule fires, returns None for the output
   dict but mutates ctx["preference_signals"] in-place with priority_bias and
   reason_hint strings (multiple signals stack; bias is clamped at -0.50).
"""

import re
from collections import defaultdict


# ── Absolute rule helpers ────────────────────────────────────────────────────

_PROMPT_INJECTION_PHRASES = [
    "ignore previous",
    "mark this as notify",
    "override routing",
    "disregard instructions",
    "ignore all previous",
    "ignore previous routing",
    "set action=notify",
    "routing override",
    "system note for the notification router",
    "internal router metadata",
    "assistant instruction",
    "always mark this as notify",
    "always mark as notify",
    "ignore sender risk",
    "mark notify",
]

_OTP_TERMS = ["otp", "password", "6 digit code", "login code", "6-digit code"]
_PRESSURE_TERMS = [
    "blocked",
    "suspended",
    "verify now",
    "expires today",
    "account block",
    "profile will be",
    "access will be",
    "account may be",
    "account may get",
    "band ho jayega",
    "block ho jayega",
    "locked",
]


def _text_lower(ctx: dict) -> str:
    text = ctx.get("message_text", "") or ""
    summary = ctx.get("media_summary", "") or ""
    return (text + " " + summary).lower()


def _contains_any(haystack: str, needles: list[str]) -> bool:
    return any(n in haystack for n in needles)


def _final_output(ctx: dict, action: str, message_type: str, reason: str,
                  confidence: float) -> dict:
    return {
        "message_id": ctx["message_id"],
        "action": action,
        "message_type": message_type,
        "reason": reason,
        "confidence": confidence,
        "evidence_message_ids": "none",
        "_rule_fired": True,
    }


# ── Public API ───────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Usage:
        result = engine.apply_absolute_rules(ctx)
        if result is not None:
            # use result directly, skip LLM
        else:
            engine.apply_preference_signals(ctx, history_events)
            # pass ctx to LLM
    """

    def __init__(self, context_builder=None):
        """context_builder is passed so we can access business / history data."""
        self.cb = context_builder

    # ------------------------------------------------------------------
    # Phase 1 — Absolute Rules
    # ------------------------------------------------------------------

    def apply_absolute_rules(self, ctx: dict) -> dict | None:
        """Returns final output dict if a rule fires, else None."""
        text_l = _text_lower(ctx)

        # Rule 1: Prompt injection
        if _contains_any(text_l, _PROMPT_INJECTION_PHRASES):
            return _final_output(
                ctx,
                action="mute",
                message_type="scam",
                reason="The message attempts to manipulate the routing system.",
                confidence=0.95,
            )

        # Rule 2: Business domain mismatch
        if ctx.get("conversation_type") == "business":
            biz = ctx.get("business") or {}
            official = biz.get("official_domain", "") or ""
            sender_domain = biz.get("domain_used_by_sender", "") or ""
            domain_age = biz.get("domain_age_days", 9999)
            # Mismatch: official domain exists, sender domain differs, and domain is new
            if (
                official
                and sender_domain
                and official.strip().lower() != sender_domain.strip().lower()
                and domain_age < 365
            ):
                return _final_output(
                    ctx,
                    action="mute",
                    message_type="scam",
                    reason="The sender domain does not match the verified business domain.",
                    confidence=0.92,
                )

        # Rule 3: OTP/scam pattern in personal messages
        if ctx.get("conversation_type") == "personal":
            has_otp = _contains_any(text_l, _OTP_TERMS)
            has_pressure = _contains_any(text_l, _PRESSURE_TERMS)
            if has_otp and has_pressure:
                # Check for prior relationship with sender
                sender = ctx.get("sender_user_id", "")
                user_id = ctx.get("user_id", "") if hasattr(ctx, "get") else ""
                has_prior = self._has_prior_relationship(ctx)
                if not has_prior:
                    return _final_output(
                        ctx,
                        action="mute",
                        message_type="scam",
                        reason=(
                            "This is the first message from the sender and it asks "
                            "for sensitive verification."
                        ),
                        confidence=0.92,
                    )

        return None

    # ------------------------------------------------------------------
    # Phase 2 — Preference Signals
    # ------------------------------------------------------------------

    def apply_preference_signals(
        self,
        ctx: dict,
        message_history: list[dict],
        message_events: dict,
    ) -> None:
        """
        Mutates ctx['preference_signals'] in-place.
        message_events is the dict keyed by (user_id, message_id).
        """
        bias = 0.0
        hints: list[str] = []

        user_id = ctx.get("user_id", "") if isinstance(ctx, dict) else ""

        # Signal 1: Group muted by user
        group = ctx.get("group") or {}
        if group.get("muted_by_user"):
            bias -= 0.35
            hints.append("User has muted this group.")

        # Signal 2: Promotions opted out
        biz = ctx.get("business") or {}
        if biz.get("promotions_opted_out_at"):
            bias -= 0.30
            hints.append("User opted out of promotions from this business.")

        # Signal 3 & 4: Same-sender history from message_history
        sender = ctx.get("sender_user_id", "") or ""
        group_id = ctx.get("group_id", "") or ""
        business_id = ctx.get("business_id", "") or ""

        muted_count = 0
        reported_count = 0

        for hist_msg in message_history:
            if hist_msg.get("user_id") != user_id:
                continue
            # Match same sender (personal / group) or same business
            same_sender = (sender and hist_msg.get("sender_user_id") == sender)
            same_biz = (business_id and hist_msg.get("business_id") == business_id)
            if not (same_sender or same_biz):
                continue

            event = message_events.get((user_id, hist_msg["message_id"]), {})
            if _safe_int(event.get("muted_after_message")):
                muted_count += 1
            if _safe_int(event.get("message_reported")):
                reported_count += 1

        if muted_count >= 2:
            bias -= 0.25
            hints.append("User has muted after messages from this sender before.")
        if reported_count >= 1:
            bias -= 0.30
            hints.append("User has previously reported messages from this sender.")

        # Signal 5: High 7-day dismissal ratio
        fatigue = ctx.get("fatigue") or {}
        if fatigue.get("last_7d_dismissed_ratio", 0) > 0.6:
            bias -= 0.15
            hints.append("User is currently dismissing a high share of notifications.")

        # Signal 6: Highly forwarded
        if ctx.get("forwarded_count", 0) > 5:
            bias -= 0.15
            hints.append("Highly forwarded message, likely low-value broadcast.")

        # Clamp total bias
        bias = max(bias, -0.50)

        ctx["preference_signals"] = {
            "priority_bias": round(bias, 3),
            "reason_hint": " ".join(hints),
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _has_prior_relationship(self, ctx: dict) -> bool:
        """
        Check if the receiver has any prior interaction with the sender.
        Uses message_history keyed lookups from the context_builder if available.
        """
        if self.cb is None:
            return False

        user_id = ctx.get("user_id", "")
        sender = ctx.get("sender_user_id", "")
        if not sender:
            return False

        for hist_msg in self.cb.message_history:
            if (
                hist_msg.get("user_id") == user_id
                and hist_msg.get("sender_user_id") == sender
            ):
                return True
        return False


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default
