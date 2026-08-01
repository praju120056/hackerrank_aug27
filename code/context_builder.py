"""
context_builder.py
------------------
Loads all auxiliary dataset CSV files at startup and builds a compact,
structured context dict for each incoming message.

The LLM should only receive this structured context — never raw CSV rows.
"""

from __future__ import annotations
import csv
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional

from models import MediaSummary


# ── CSV utilities ─────────────────────────────────────────────────────────────

def _load_csv(filepath: str) -> list[dict]:
    """Load a CSV file; return [] if the file is missing."""
    if not os.path.exists(filepath):
        print(f"[WARN] Missing dataset file: {filepath}")
        return []
    with open(filepath, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _int(val, default: int = 0) -> int:
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


# ── Public class ──────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Loads all dataset CSVs once at construction time.
    Use build_context(msg_row, media_summary) to produce a structured dict
    for a single incoming message.
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._load_all()

    # ------------------------------------------------------------------
    # Load phase
    # ------------------------------------------------------------------

    def _p(self, name: str) -> str:
        return os.path.join(self.dataset_dir, name)

    def _load_all(self) -> None:
        # Users indexed by user_id
        self.users: dict[str, dict] = {
            r["user_id"]: r for r in _load_csv(self._p("users.csv"))
        }

        # Groups indexed by group_id
        self.groups: dict[str, dict] = {
            r["group_id"]: r for r in _load_csv(self._p("groups.csv"))
        }

        # Group membership indexed by (group_id, user_id)
        self.group_members: dict[tuple, dict] = {
            (r["group_id"], r["user_id"]): r
            for r in _load_csv(self._p("group_members.csv"))
        }

        # Businesses indexed by business_id
        self.businesses: dict[str, dict] = {
            r["business_id"]: r for r in _load_csv(self._p("business_accounts.csv"))
        }

        # User-business history indexed by (user_id, business_id)
        self.user_biz: dict[tuple, dict] = {
            (r["user_id"], r["business_id"]): r
            for r in _load_csv(self._p("user_business_history.csv"))
        }

        # Message events indexed by (user_id, message_id)
        self.message_events: dict[tuple, dict] = {
            (r["user_id"], r["message_id"]): r
            for r in _load_csv(self._p("message_events.csv"))
        }

        # Daily notification summary grouped by user_id
        self.daily_notif: dict[str, list[dict]] = defaultdict(list)
        for r in _load_csv(self._p("daily_notification_summary.csv")):
            self.daily_notif[r["user_id"]].append(r)

        # Image/voice paths indexed by ID
        self.images: dict[str, str] = {
            r["image_id"]: r["file_path"]
            for r in _load_csv(self._p("images.csv"))
        }
        self.voice_notes: dict[str, str] = {
            r["voice_note_id"]: r["file_path"]
            for r in _load_csv(self._p("voice_notes.csv"))
        }

        # Full message history kept as a list (used by evidence + rule engine)
        self.message_history: list[dict] = _load_csv(self._p("message_history.csv"))

    # ------------------------------------------------------------------
    # Media path helpers
    # ------------------------------------------------------------------

    def image_path(self, media_id: str) -> str | None:
        """Absolute path for an image, or None if missing."""
        rel = self.images.get(media_id)
        if not rel:
            return None
        full = os.path.join(self.dataset_dir, rel)
        return full if os.path.exists(full) else None

    def voice_path(self, media_id: str) -> str | None:
        """Absolute path for a voice note, or None if missing."""
        rel = self.voice_notes.get(media_id)
        if not rel:
            return None
        full = os.path.join(self.dataset_dir, rel)
        return full if os.path.exists(full) else None

    # ------------------------------------------------------------------
    # Build context
    # ------------------------------------------------------------------

    def build_context(self, msg: dict, media_summary: MediaSummary) -> dict:
        """
        Return a compact structured context dict for one incoming message.

        Args:
            msg:           Raw row from messages.csv.
            media_summary: Structured media understanding (MediaSummary).

        Returns:
            A plain dict suitable for JSON serialisation and LLM consumption.
        """
        uid = msg.get("user_id", "")
        gid = msg.get("group_id", "") or ""
        bid = msg.get("business_id", "") or ""
        ctype = msg.get("conversation_type", "")
        ts = msg.get("created_at", "")

        return {
            # ── Message identity ──────────────────────────────────────
            "message_id": msg["message_id"],
            "conversation_type": ctype,
            "created_at": ts,
            "forwarded_count": _int(msg.get("forwarded_count")),

            # ── Message content ───────────────────────────────────────
            "text": (msg.get("message_text") or "")[:800],
            "media_type": msg.get("media_type", "") or "",
            "media": {
                "summary": media_summary.summary,
                "category": media_summary.category,
                "urgency": media_summary.urgency,
                "entities": media_summary.entities,
                "action_required": media_summary.action_required,
            } if media_summary.summary else None,

            # ── User behaviour ────────────────────────────────────────
            "user": self._user_ctx(uid, ts),

            # ── Group context ─────────────────────────────────────────
            "group": self._group_ctx(gid, uid) if (ctype == "group" and gid) else None,

            # ── Business context ──────────────────────────────────────
            "business": self._biz_ctx(bid, uid) if (ctype == "business" and bid) else None,

            # ── Sender identity (for personal / group messages) ───────
            "sender_user_id": msg.get("sender_user_id", "") or "",

            # ── Notification fatigue (last 7 days) ────────────────────
            "fatigue": self._fatigue(uid, ts),

            # ── Preference signals — filled in by rule_engine ─────────
            "preference_signals": {"priority_bias": 0.0, "reason_hint": ""},

            # ── Evidence — filled in by evidence module ───────────────
            "evidence": [],
        }

    # ------------------------------------------------------------------
    # Sub-context builders
    # ------------------------------------------------------------------

    def _user_ctx(self, uid: str, ts: str) -> dict:
        row = self.users.get(uid, {})
        opened = _int(row.get("messages_opened_30d"))
        replied = _int(row.get("messages_replied_30d"))
        dismissed = _int(row.get("notifications_dismissed_30d"))
        reported = _int(row.get("messages_reported_30d"))
        total = max(opened + dismissed, 1)
        return {
            "dnd_active": self._dnd_active(row.get("do_not_disturb_window", ""), ts),
            "dismiss_rate_30d": round(dismissed / total, 3),
            "report_rate_30d": round(reported / total, 3),
            "engage_rate_30d": round((opened + replied) / total, 3),
            "total_messages_30d": total,
        }

    def _group_ctx(self, gid: str, uid: str) -> dict:
        g = self.groups.get(gid, {})
        m = self.group_members.get((gid, uid), {})
        return {
            "name": g.get("group_name", ""),
            "type": g.get("group_type", ""),
            "member_count": _int(g.get("member_count")),
            "messages_30d": _int(g.get("messages_30d")),
            "user_role": m.get("role", "member"),
            "muted_by_user": bool(_int(m.get("group_muted_by_user"))),
            "user_dismissals_30d": _int(m.get("notifications_dismissed_30d")),
            "user_messages_read_30d": _int(m.get("messages_read_30d")),
        }

    def _biz_ctx(self, bid: str, uid: str) -> dict:
        b = self.businesses.get(bid, {})
        h = self.user_biz.get((uid, bid), {})
        return {
            "name": b.get("display_name", ""),
            "category": b.get("category", ""),
            "verified": bool(_int(b.get("verified"))),
            "official_domain": b.get("official_domain", "") or "",
            "sender_domain": b.get("domain_used_by_sender", "") or "",
            "sender_domain_age_days": _int(b.get("domain_used_by_sender_age_days")),
            "account_age_days": _int(b.get("account_age_days")),
            "user_reports_30d": _int(b.get("user_reports_30d")),
            "relationship": h.get("why_user_knows_account", "none"),
            "promotions_opted_out": bool(h.get("promotions_opted_out_at", "")),
            "promotions_opted_out_at": h.get("promotions_opted_out_at", ""),
            "activity_count_180d": _int(h.get("activity_count_180d")),
            "messages_opened_30d": _int(h.get("messages_opened_30d")),
            "messages_dismissed_30d": _int(h.get("messages_dismissed_30d")),
        }

    def _fatigue(self, uid: str, ts: str) -> dict:
        """Compute notification dismissal ratio for the last 7 days."""
        try:
            ref = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M").date()
        except Exception:
            ref = date.today()

        cutoff = ref - timedelta(days=7)
        sent = dismissed = 0
        for row in self.daily_notif.get(uid, []):
            try:
                d = date.fromisoformat(row["date"])
            except Exception:
                continue
            if cutoff <= d < ref:
                sent += _int(row.get("notifications_sent"))
                dismissed += _int(row.get("notifications_dismissed"))

        ratio = dismissed / sent if sent > 0 else 0.0
        return {
            "last_7d_dismissed_ratio": round(ratio, 3),
            "last_7d_sent": sent,
        }

    # ------------------------------------------------------------------
    # DND helper
    # ------------------------------------------------------------------

    @staticmethod
    def _dnd_active(window: str, ts: str) -> bool:
        """Return True if the message timestamp falls within the DND window."""
        if not window or not ts:
            return False
        try:
            start_str, end_str = window.split("-")
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M")
            t = dt.hour * 60 + dt.minute
            s = sh * 60 + sm
            e = eh * 60 + em
            # Spans midnight when start > end
            return (t >= s or t < e) if s > e else (s <= t < e)
        except Exception:
            return False
