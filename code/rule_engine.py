"""
rule_engine.py
--------------
Two-phase rule engine.

Phase 1 — Absolute Rules
    Deterministic checks that produce a final prediction without calling Gemini.
    First match wins. Checks are ordered from highest to lowest confidence.

Phase 2 — Preference Signals
    Behavioral signals that enrich the context dict (mutating preference_signals)
    to bias the LLM decision. They do NOT force an outcome; the LLM remains free
    to override a negative bias if the message content is genuinely urgent.
"""

from __future__ import annotations
import re
from models import RoutingPrediction, CONF_MIN, CONF_MAX


# ── Phrase lists ──────────────────────────────────────────────────────────────

_INJECTION_PHRASES = [
    "ignore previous",
    "ignore all previous",
    "override routing",
    "disregard instructions",
    "mark this as notify",
    "mark as notify",
    "set action=notify",
    "action=notify",
    "routing override",
    "system note for the notification router",
    "internal router metadata",
    "assistant instruction",
    "always mark this as notify",
    "ignore sender risk",
    "mark notify",
    "notification router:",
    "router instruction",
]

_OTP_TERMS = [
    "otp", "6 digit code", "6-digit code", "login code",
    "verification code", "one time password",
]

_PRESSURE_TERMS = [
    "blocked", "suspended", "verify now", "expires today",
    "account block", "account will be blocked", "account may be blocked",
    "profile will be restricted", "profile will be",
    "access will be suspended", "access will expire",
    "band ho jayega", "block ho jayega", "locked",
    "permanently block", "permanent block",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _text(ctx: dict) -> str:
    """Combined lowercased text + media summary for pattern matching."""
    t = (ctx.get("text") or "").lower()
    media = ctx.get("media") or {}
    s = (media.get("summary") or "").lower()
    return t + " " + s


def _contains(haystack: str, needles: list[str]) -> bool:
    return any(n in haystack for n in needles)


def _make_prediction(
    ctx: dict,
    action: str,
    message_type: str,
    reason: str,
    confidence: float,
) -> RoutingPrediction:
    return RoutingPrediction(
        message_id=ctx["message_id"],
        action=action,
        message_type=message_type,
        reason=reason,
        confidence=confidence,
        evidence_message_ids="none",
        rule_fired=True,
    )


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


# ── Public class ──────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Stateless rule engine. Pass a ContextBuilder instance so that
    prior-relationship checks can query message_history.
    """

    def __init__(self, context_builder=None):
        self.cb = context_builder

    # ------------------------------------------------------------------
    # Phase 1 — Absolute Rules
    # ------------------------------------------------------------------

    def apply_absolute_rules(self, ctx: dict) -> RoutingPrediction | None:
        """
        Check absolute rules in order. Returns a RoutingPrediction on first match,
        or None if no rule fires (message should proceed to LLM).
        """
        text_l = _text(ctx)

        # ── Rule 1: Prompt injection ─────────────────────────────────
        if _contains(text_l, _INJECTION_PHRASES):
            return _make_prediction(
                ctx, "mute", "scam",
                "The message attempts to manipulate the notification routing system.",
                0.95,
            )

        # ── Rule 2: Business domain mismatch ─────────────────────────
        if ctx.get("conversation_type") == "business":
            biz = ctx.get("business") or {}
            official = (biz.get("official_domain") or "").strip().lower()
            sender = (biz.get("sender_domain") or "").strip().lower()
            age = biz.get("sender_domain_age_days", 9999)
            # Only fire when: official domain exists, sender differs, and domain is <1 year old
            if official and sender and official != sender and age < 365:
                return _make_prediction(
                    ctx, "mute", "scam",
                    "The sender domain does not match the verified business domain.",
                    0.92,
                )

        # ── Rule 3: OTP/account-lock scam in personal messages ───────
        if ctx.get("conversation_type") == "personal":
            if _contains(text_l, _OTP_TERMS) and _contains(text_l, _PRESSURE_TERMS):
                if not self._has_prior_relationship(ctx):
                    return _make_prediction(
                        ctx, "mute", "scam",
                        "This is the first message from the sender and it requests "
                        "sensitive verification under pressure.",
                        0.92,
                    )

        return None  # No absolute rule matched

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
        Compute stacked priority_bias and reason_hint, clamped at −0.50.
        Mutates ctx['preference_signals'] in-place.

        Args:
            ctx:             Message context dict (will be mutated).
            message_history: Full historical message list from ContextBuilder.
            message_events:  Dict keyed by (user_id, message_id).
        """
        bias = 0.0
        hints: list[str] = []

        user_id = ctx.get("user_id", "")
        sender = ctx.get("sender_user_id", "") or ""
        group = ctx.get("group") or {}
        biz = ctx.get("business") or {}

        # Signal 1: Group muted by user
        if group.get("muted_by_user"):
            bias -= 0.35
            hints.append("User has muted this group.")

        # Signal 2: Promotion opt-out from this business
        if biz.get("promotions_opted_out"):
            bias -= 0.30
            hints.append("User opted out of promotions from this business.")

        # Signals 3 & 4: Look at same-sender / same-business history
        business_id = ctx.get("business_id", "") or (biz and "") or ""
        # Infer business_id from context
        raw_bid = ""
        for k in ("business_id",):
            raw_bid = ctx.get(k, "") or ""
        biz_name = biz.get("name", "") if biz else ""

        muted_count = 0
        reported_count = 0
        for hist in message_history:
            if hist.get("user_id") != user_id:
                continue
            same_sender = sender and hist.get("sender_user_id") == sender
            same_biz = raw_bid and hist.get("business_id") == raw_bid
            if not (same_sender or same_biz):
                continue
            ev = message_events.get((user_id, hist["message_id"]), {})
            if _safe_int(ev.get("muted_after_message")):
                muted_count += 1
            if _safe_int(ev.get("message_reported")):
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

        # Clamp
        bias = max(bias, -0.50)

        ctx["preference_signals"] = {
            "priority_bias": round(bias, 3),
            "reason_hint": " ".join(hints),
        }

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _has_prior_relationship(self, ctx: dict) -> bool:
        """Check message_history for any prior exchange with this sender."""
        if self.cb is None:
            return False
        uid = ctx.get("user_id", "")
        sender = ctx.get("sender_user_id", "") or ""
        if not sender:
            return False
        return any(
            h.get("user_id") == uid and h.get("sender_user_id") == sender
            for h in self.cb.message_history
        )
