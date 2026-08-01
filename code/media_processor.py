"""
media_processor.py
------------------
Multimodal understanding using Gemini's native capabilities.

Key design decisions:
- Every unique media asset (keyed by media_id) is processed EXACTLY ONCE.
- Results are cached in-memory for the lifetime of the pipeline run.
- Returns a structured MediaSummary (not a raw string) so downstream code
  can use category/urgency/entities without re-parsing free text.
- Missing or unreadable media files produce MediaSummary.unavailable().
"""

from __future__ import annotations
import json
import os
import re
import time

from models import MediaSummary


# ── Prompts ───────────────────────────────────────────────────────────────────

_IMAGE_SYSTEM = (
    "You are a content analysis tool for a notification routing system. "
    "Analyse the provided image and return ONLY a JSON object — no other text."
)

_IMAGE_USER = """Analyse this image and return this JSON:
{
  "summary": "one sentence describing the image content",
  "category": "promotional|informational|urgent|scam|personal|unknown",
  "urgency": "low|medium|high",
  "entities": ["brand names, URLs, phone numbers, people, orgs visible — empty list if none"],
  "action_required": true_or_false,
  "confidence": 0.0_to_1.0
}

Focus on: content type, any visible text, urgency signals, promotional vs informational vs scam.
Return ONLY valid JSON."""

_VOICE_SYSTEM = (
    "You are an audio analysis tool for a notification routing system. "
    "Analyse the provided voice note and return ONLY a JSON object — no other text."
)

_VOICE_USER = """Transcribe and analyse this voice note, then return this JSON:
{
  "summary": "one sentence: main topic, request or action if any",
  "category": "promotional|informational|urgent|scam|personal|unknown",
  "urgency": "low|medium|high",
  "entities": ["names, brands, numbers, URLs mentioned — empty list if none"],
  "action_required": true_or_false,
  "confidence": 0.0_to_1.0
}

Return ONLY valid JSON."""


# ── Mime types ────────────────────────────────────────────────────────────────

def _mime(path: str) -> str:
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    if lower.endswith((".ogg", ".oga")):
        return "audio/ogg"
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".m4a"):
        return "audio/mp4"
    return "application/octet-stream"


# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_media_json(raw: str) -> dict:
    """Strip markdown fences and parse the JSON object from a Gemini response."""
    text = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _build_summary(data: dict) -> MediaSummary:
    """Convert parsed dict to a validated MediaSummary."""
    valid_categories = {"promotional", "informational", "urgent", "scam", "personal", "unknown"}
    valid_urgency = {"low", "medium", "high"}

    category = str(data.get("category", "unknown")).lower()
    if category not in valid_categories:
        category = "unknown"

    urgency = str(data.get("urgency", "low")).lower()
    if urgency not in valid_urgency:
        urgency = "low"

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    entities = [str(e) for e in entities if e]

    return MediaSummary(
        summary=str(data.get("summary", ""))[:200],
        category=category,
        urgency=urgency,
        entities=entities[:10],  # cap at 10 entities
        action_required=bool(data.get("action_required", False)),
        confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
    )


# ── Public interface ──────────────────────────────────────────────────────────

class MediaProcessor:
    """
    Stateful processor that caches results by media_id.
    Instantiate once per pipeline run and pass the same instance to all callers.
    """

    def __init__(self, client, model_name: str, media_sleep: float = 2.0):
        """
        Args:
            client:       Initialised google.genai.Client.
            model_name:   Gemini model for multimodal understanding.
            media_sleep:  Seconds to sleep after each Gemini call (rate-limit guard).
        """
        self._client = client
        self._model = model_name
        self._sleep = media_sleep
        self._cache: dict[str, MediaSummary] = {}   # key: media_id

    def process(self, media_type: str, media_id: str, media_path: str | None) -> MediaSummary:
        """
        Return a MediaSummary for the given media asset.
        Results are cached — calling this twice with the same media_id is free.

        Args:
            media_type:  "image" | "voice" | "" (text messages -> returns empty summary)
            media_id:    Unique identifier (e.g. "img_008", "vn_012").
            media_path:  Absolute filesystem path, or None if not found.
        """
        if not media_type or media_type not in ("image", "voice"):
            return MediaSummary("", "unknown", "low", [], False, 1.0)

        cache_key = media_id if media_id else (media_path or "")
        if cache_key in self._cache:
            print(f"    [MEDIA] Cache hit: {cache_key}")
            return self._cache[cache_key]

        summary = self._fetch(media_type, media_path, cache_key)
        self._cache[cache_key] = summary
        return summary

    def _fetch(self, media_type: str, media_path: str | None, cache_key: str) -> MediaSummary:
        """Call Gemini to analyse media. Returns unavailable on any failure."""
        if not media_path or not os.path.exists(media_path):
            print(f"    [MEDIA] Missing file for {cache_key}: {media_path}")
            return MediaSummary.unavailable()

        try:
            file_bytes = open(media_path, "rb").read()
            if not file_bytes:
                return MediaSummary.unavailable()

            from google.genai import types

            mime = _mime(media_path)
            if media_type == "image":
                sys_prompt, user_prompt = _IMAGE_SYSTEM, _IMAGE_USER
            else:
                sys_prompt, user_prompt = _VOICE_SYSTEM, _VOICE_USER

            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime),
                    user_prompt,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )

            raw = response.text or ""
            data = _parse_media_json(raw)
            summary = _build_summary(data)
            print(f"    [MEDIA] Processed {cache_key}: {summary.category}/{summary.urgency}")
            return summary

        except json.JSONDecodeError as exc:
            print(f"    [MEDIA] JSON parse error for {cache_key}: {exc}")
            return MediaSummary.unavailable()
        except Exception as exc:
            print(f"    [MEDIA] Error for {cache_key}: {exc}")
            return MediaSummary.unavailable()
        finally:
            # Always sleep to respect rate limits, even on errors
            if self._sleep > 0:
                time.sleep(self._sleep)

    @property
    def cache_size(self) -> int:
        return len(self._cache)
