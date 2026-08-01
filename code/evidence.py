"""
evidence.py
-----------
Key-match retrieval from message_history + message_events.

Priority order for same user_id:
  1. Same sender_user_id
  2. Same group_id
  3. Same business_id

Returns up to 3 evidence dicts, each with message text (truncated to 120 chars)
plus all event flags. If no matches, evidence_message_ids = "none".
"""

from __future__ import annotations


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _truncate(text: str, limit: int = 120) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def retrieve_evidence(
    msg: dict,
    message_history: list[dict],
    message_events: dict,  # keyed by (user_id, message_id)
    top_k: int = 3,
) -> list[dict]:
    """
    Returns a list of up to top_k evidence dicts sorted by priority.
    Each dict contains:
      message_id, text, opened, replied, dismissed, muted_after, reported
    """
    user_id = msg.get("user_id", "")
    sender = msg.get("sender_user_id", "") or ""
    group_id = msg.get("group_id", "") or ""
    business_id = msg.get("business_id", "") or ""

    priority_1: list[dict] = []
    priority_2: list[dict] = []
    priority_3: list[dict] = []

    for hist in message_history:
        if hist.get("user_id") != user_id:
            continue

        mid = hist.get("message_id", "")
        event = message_events.get((user_id, mid), {})

        evidence_item = {
            "message_id": mid,
            "text": _truncate(hist.get("message_text", "")),
            "opened": bool(_safe_int(event.get("message_opened"))),
            "replied": bool(_safe_int(event.get("message_replied"))),
            "dismissed": bool(_safe_int(event.get("notification_dismissed"))),
            "muted_after": bool(_safe_int(event.get("muted_after_message"))),
            "reported": bool(_safe_int(event.get("message_reported"))),
        }

        # Priority 1: same sender
        if sender and hist.get("sender_user_id") == sender:
            priority_1.append(evidence_item)
        # Priority 2: same group
        elif group_id and hist.get("group_id") == group_id:
            priority_2.append(evidence_item)
        # Priority 3: same business
        elif business_id and hist.get("business_id") == business_id:
            priority_3.append(evidence_item)

    # Merge in priority order, deduplicate by message_id
    seen: set[str] = set()
    result: list[dict] = []
    for bucket in (priority_1, priority_2, priority_3):
        for item in bucket:
            if item["message_id"] not in seen:
                seen.add(item["message_id"])
                result.append(item)
                if len(result) >= top_k:
                    return result

    return result


def evidence_ids_string(evidence: list[dict]) -> str:
    """Returns semicolon-separated evidence message IDs, or 'none'."""
    ids = [e["message_id"] for e in evidence if e.get("message_id")]
    return ";".join(ids) if ids else "none"
