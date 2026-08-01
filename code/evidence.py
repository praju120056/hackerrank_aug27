"""
evidence.py
-----------
Key-match retrieval from message_history + message_events.

Evidence is ranked by:
  Priority tier (sender > group > business)
  Within each tier: recency, opens, replies, then suppression signals

Returns at most top_k items (default 3).
Evidence IDs string: semicolon-separated or "none".
"""

from __future__ import annotations
from datetime import datetime


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _truncate(text: str, limit: int = 120) -> str:
    """Truncate and normalise whitespace."""
    text = " ".join((text or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _recency_score(created_at: str) -> float:
    """Higher is more recent. Returns days-since-epoch as a float."""
    try:
        dt = datetime.strptime(created_at.strip(), "%Y-%m-%d %H:%M")
        return dt.timestamp()
    except Exception:
        return 0.0


def _engagement_score(event: dict) -> float:
    """
    Positive engagement (open/reply) raises score; dismissal/mute/report lowers it.
    Used as a secondary ranking key within the same priority tier.
    """
    score = 0.0
    score += 2.0 * _safe_int(event.get("message_replied"))
    score += 1.0 * _safe_int(event.get("message_opened"))
    score -= 1.5 * _safe_int(event.get("notification_dismissed"))
    score -= 2.0 * _safe_int(event.get("muted_after_message"))
    score -= 3.0 * _safe_int(event.get("message_reported"))
    return score


def retrieve_evidence(
    msg: dict,
    message_history: list[dict],
    message_events: dict,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve up to top_k evidence messages for the incoming message.

    Ranking:
      1. Tier priority:  same sender_user_id  >  same group_id  >  same business_id
      2. Within tier:    (recency, engagement_score) — both descending

    Each returned item includes message_id, text (truncated), and all event flags.

    Args:
        msg:             Incoming message row from messages.csv.
        message_history: List of historical message rows.
        message_events:  Dict keyed by (user_id, message_id).
        top_k:           Maximum evidence items to return.
    """
    uid = msg.get("user_id", "")
    sender = msg.get("sender_user_id", "") or ""
    gid = msg.get("group_id", "") or ""
    bid = msg.get("business_id", "") or ""

    tier1: list[tuple] = []   # (recency, engagement, item)
    tier2: list[tuple] = []
    tier3: list[tuple] = []

    for hist in message_history:
        if hist.get("user_id") != uid:
            continue

        mid = hist.get("message_id", "")
        event = message_events.get((uid, mid), {})
        rec = _recency_score(hist.get("created_at", ""))
        eng = _engagement_score(event)

        item = {
            "message_id": mid,
            "text": _truncate(hist.get("message_text", "")),
            "media_type": hist.get("media_type", "") or "",
            "opened": bool(_safe_int(event.get("message_opened"))),
            "replied": bool(_safe_int(event.get("message_replied"))),
            "dismissed": bool(_safe_int(event.get("notification_dismissed"))),
            "muted_after": bool(_safe_int(event.get("muted_after_message"))),
            "reported": bool(_safe_int(event.get("message_reported"))),
        }

        if sender and hist.get("sender_user_id") == sender:
            tier1.append((rec, eng, item))
        elif gid and hist.get("group_id") == gid:
            tier2.append((rec, eng, item))
        elif bid and hist.get("business_id") == bid:
            tier3.append((rec, eng, item))

    # Sort each tier: most recent first, then by engagement
    for tier in (tier1, tier2, tier3):
        tier.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Merge tiers, deduplicate, take top_k
    seen: set[str] = set()
    result: list[dict] = []
    for _, _, item in tier1 + tier2 + tier3:
        mid = item["message_id"]
        if mid not in seen:
            seen.add(mid)
            result.append(item)
            if len(result) >= top_k:
                break

    return result


def evidence_ids_string(evidence: list[dict]) -> str:
    """Return semicolon-separated evidence IDs, or 'none'."""
    ids = [e["message_id"] for e in evidence if e.get("message_id")]
    return ";".join(ids) if ids else "none"
