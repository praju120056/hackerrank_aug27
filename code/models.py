"""
models.py
---------
Shared dataclasses and constants used across all pipeline modules.
Keeping types in one place avoids circular imports and duplicate definitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ── Output contract ───────────────────────────────────────────────────────────

VALID_ACTIONS = {"notify", "digest", "mute"}

VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}

OUTPUT_COLUMNS = [
    "message_id", "action", "message_type",
    "reason", "confidence", "evidence_message_ids",
]

# Confidence range enforced by the output validator
CONF_MIN = 0.55
CONF_MAX = 0.95


# ── Media understanding ───────────────────────────────────────────────────────

@dataclass
class MediaSummary:
    """
    Structured semantic representation of an image or voice note produced by
    Gemini's multimodal understanding. Downstream modules should consume this
    object rather than raw Gemini text.
    """
    summary: str           # One-sentence description of content
    category: str          # promotional | informational | urgent | scam | personal | unknown
    urgency: str           # low | medium | high
    entities: list[str]    # People, orgs, URLs, phone numbers, brands visible
    action_required: bool  # Does the content ask the recipient to do something?
    confidence: float      # Model's confidence in its understanding (0–1)

    def to_prompt_str(self) -> str:
        """Compact representation to embed in the decision-engine prompt."""
        entities_str = ", ".join(self.entities) if self.entities else "none"
        return (
            f"[{self.category.upper()}/{self.urgency.upper()}] "
            f"{self.summary} "
            f"(entities: {entities_str}; "
            f"action_required: {self.action_required})"
        )

    @staticmethod
    def unavailable() -> "MediaSummary":
        return MediaSummary(
            summary="Media unavailable or could not be processed.",
            category="unknown",
            urgency="low",
            entities=[],
            action_required=False,
            confidence=0.0,
        )


# ── Routing prediction ────────────────────────────────────────────────────────

@dataclass
class RoutingPrediction:
    """One validated prediction row ready for output.csv."""
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str   # semicolon-separated or "none"
    rule_fired: bool = False    # True if produced by an absolute rule (no LLM)

    def to_csv_row(self) -> dict:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "evidence_message_ids": self.evidence_message_ids,
        }
