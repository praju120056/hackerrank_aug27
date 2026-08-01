"""
context_builder.py
------------------
Loads all auxiliary CSV files at startup and builds a compact, structured
JSON context dict for each incoming message in messages.csv.
"""

import csv
import os
from datetime import datetime, date, timedelta
from collections import defaultdict


def _load_csv(filepath: str) -> list[dict]:
    """Load a CSV file and return a list of row dicts. Returns [] if missing."""
    if not os.path.exists(filepath):
        print(f"[WARN] Missing CSV: {filepath}")
        return []
    with open(filepath, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


class ContextBuilder:
    """Loads all dataset CSVs and exposes build_context(message_row) -> dict."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._load_all()

    # ------------------------------------------------------------------
    # Load phase
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> str:
        return os.path.join(self.dataset_dir, filename)

    def _load_all(self):
        # Keyed lookups for O(1) access
        self.users: dict[str, dict] = {}
        for row in _load_csv(self._path("users.csv")):
            self.users[row["user_id"]] = row

        self.groups: dict[str, dict] = {}
        for row in _load_csv(self._path("groups.csv")):
            self.groups[row["group_id"]] = row

        # group_members keyed by (group_id, user_id)
        self.group_members: dict[tuple, dict] = {}
        for row in _load_csv(self._path("group_members.csv")):
            self.group_members[(row["group_id"], row["user_id"])] = row

        self.businesses: dict[str, dict] = {}
        for row in _load_csv(self._path("business_accounts.csv")):
            self.businesses[row["business_id"]] = row

        # user_business_history keyed by (user_id, business_id)
        self.user_biz_history: dict[tuple, dict] = {}
        for row in _load_csv(self._path("user_business_history.csv")):
            self.user_biz_history[(row["user_id"], row["business_id"])] = row

        # message_events keyed by (user_id, message_id)
        self.message_events: dict[tuple, dict] = {}
        for row in _load_csv(self._path("message_events.csv")):
            self.message_events[(row["user_id"], row["message_id"])] = row

        # daily_notification_summary grouped by user_id
        self.daily_notif: dict[str, list[dict]] = defaultdict(list)
        for row in _load_csv(self._path("daily_notification_summary.csv")):
            self.daily_notif[row["user_id"]].append(row)

        # images and voice_notes keyed by ID
        self.images: dict[str, str] = {}
        for row in _load_csv(self._path("images.csv")):
            self.images[row["image_id"]] = row["file_path"]

        self.voice_notes: dict[str, str] = {}
        for row in _load_csv(self._path("voice_notes.csv")):
            self.voice_notes[row["voice_note_id"]] = row["file_path"]

        # message_history raw list (used by evidence.py) - also store here
        self.message_history: list[dict] = _load_csv(
            self._path("message_history.csv")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_context(self, msg: dict, media_summary: str = "") -> dict:
        """Return a compact context dict for one incoming message row."""
        user_id = msg["user_id"]
        group_id = msg.get("group_id") or ""
        business_id = msg.get("business_id") or ""
        conversation_type = msg.get("conversation_type", "")

        # ---- User context ----
        user_row = self.users.get(user_id, {})
        opened_30d = _safe_int(user_row.get("messages_opened_30d"))
        replied_30d = _safe_int(user_row.get("messages_replied_30d"))
        dismissed_30d = _safe_int(user_row.get("notifications_dismissed_30d"))
        reported_30d = _safe_int(user_row.get("messages_reported_30d"))
        total_received_approx = max(opened_30d + dismissed_30d, 1)
        dnd_window = user_row.get("do_not_disturb_window", "")
        dnd_active = self._is_dnd_active(dnd_window, msg.get("created_at", ""))

        user_ctx = {
            "dnd_active": dnd_active,
            "dismiss_rate_30d": round(dismissed_30d / total_received_approx, 3),
            "report_rate_30d": round(reported_30d / total_received_approx, 3),
            "engage_rate_30d": round(
                (opened_30d + replied_30d) / total_received_approx, 3
            ),
        }

        # ---- Group context ----
        group_ctx = None
        if conversation_type == "group" and group_id:
            group_row = self.groups.get(group_id, {})
            member_row = self.group_members.get((group_id, user_id), {})
            group_ctx = {
                "name": group_row.get("group_name", ""),
                "type": group_row.get("group_type", ""),
                "member_count": _safe_int(group_row.get("member_count")),
                "role": member_row.get("role", "member"),
                "muted_by_user": bool(_safe_int(member_row.get("group_muted_by_user"))),
                "user_dismissals_in_group_30d": _safe_int(
                    member_row.get("notifications_dismissed_30d")
                ),
            }

        # ---- Business context ----
        business_ctx = None
        if conversation_type == "business" and business_id:
            biz_row = self.businesses.get(business_id, {})
            biz_hist = self.user_biz_history.get((user_id, business_id), {})
            business_ctx = {
                "display_name": biz_row.get("display_name", ""),
                "category": biz_row.get("category", ""),
                "verified": bool(_safe_int(biz_row.get("verified"))),
                "official_domain": biz_row.get("official_domain", ""),
                "domain_used_by_sender": biz_row.get("domain_used_by_sender", ""),
                "domain_age_days": _safe_int(
                    biz_row.get("domain_used_by_sender_age_days")
                ),
                "account_age_days": _safe_int(biz_row.get("account_age_days")),
                "user_reports_30d": _safe_int(biz_row.get("user_reports_30d")),
                "relationship": biz_hist.get("why_user_knows_account", "none"),
                "allows_promotions": bool(_safe_int(biz_hist.get("allows_promotions"))),
                "promotions_opted_out_at": biz_hist.get("promotions_opted_out_at", ""),
                "activity_count_180d": _safe_int(biz_hist.get("activity_count_180d")),
                "messages_opened_30d": _safe_int(biz_hist.get("messages_opened_30d")),
                "messages_dismissed_30d": _safe_int(
                    biz_hist.get("messages_dismissed_30d")
                ),
            }

        # ---- Fatigue: last-7d dismissed ratio ----
        fatigue = self._compute_fatigue(user_id, msg.get("created_at", ""))

        ctx = {
            "message_id": msg["message_id"],
            "message_text": msg.get("message_text", ""),
            "media_type": msg.get("media_type", ""),
            "media_id": msg.get("media_id", ""),
            "conversation_type": conversation_type,
            "group_id": group_id,
            "business_id": business_id,
            "sender_user_id": msg.get("sender_user_id", ""),
            "created_at": msg.get("created_at", ""),
            "forwarded_count": _safe_int(msg.get("forwarded_count")),
            "media_summary": media_summary,
            "user": user_ctx,
            "group": group_ctx,
            "business": business_ctx,
            "preference_signals": {"priority_bias": 0.0, "reason_hint": ""},
            "evidence": [],
            "fatigue": fatigue,
        }
        return ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_dnd_active(self, dnd_window: str, created_at: str) -> bool:
        """Check if message timestamp falls within the DND window."""
        if not dnd_window or not created_at:
            return False
        try:
            parts = dnd_window.split("-")
            if len(parts) != 2:
                return False
            start_h, start_m = map(int, parts[0].split(":"))
            end_h, end_m = map(int, parts[1].split(":"))
            msg_dt = datetime.strptime(created_at.strip(), "%Y-%m-%d %H:%M")
            msg_t = msg_dt.hour * 60 + msg_dt.minute
            start_t = start_h * 60 + start_m
            end_t = end_h * 60 + end_m
            # DND spans midnight if start > end
            if start_t > end_t:
                return msg_t >= start_t or msg_t < end_t
            else:
                return start_t <= msg_t < end_t
        except Exception:
            return False

    def _compute_fatigue(self, user_id: str, created_at: str) -> dict:
        """Compute last-7d dismissed/sent ratio for fatigue signal."""
        try:
            ref_date = datetime.strptime(created_at.strip(), "%Y-%m-%d %H:%M").date()
        except Exception:
            ref_date = date.today()

        seven_days_ago = ref_date - timedelta(days=7)
        rows = self.daily_notif.get(user_id, [])

        total_sent = 0
        total_dismissed = 0
        for row in rows:
            try:
                row_date = date.fromisoformat(row["date"])
            except Exception:
                continue
            if seven_days_ago <= row_date < ref_date:
                total_sent += _safe_int(row.get("notifications_sent"))
                total_dismissed += _safe_int(row.get("notifications_dismissed"))

        ratio = total_dismissed / total_sent if total_sent > 0 else 0.0
        return {
            "last_7d_dismissed_ratio": round(ratio, 3),
            "last_7d_sent": total_sent,
        }

    def get_image_path(self, media_id: str) -> str | None:
        """Return filesystem path for an image media_id, or None."""
        rel = self.images.get(media_id)
        if not rel:
            return None
        full = os.path.join(self.dataset_dir, rel)
        return full if os.path.exists(full) else None

    def get_voice_path(self, media_id: str) -> str | None:
        """Return filesystem path for a voice_note media_id, or None."""
        rel = self.voice_notes.get(media_id)
        if not rel:
            return None
        full = os.path.join(self.dataset_dir, rel)
        return full if os.path.exists(full) else None
