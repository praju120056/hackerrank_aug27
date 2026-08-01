"""
media_processor.py
------------------
Handles inline image/audio multimodal description via the Gemini API.

- Text messages: pass through (no processing)
- Image messages: describe with Gemini (inline bytes)
- Voice messages: transcribe with Gemini (inline bytes)
- Missing/unreadable media: return "Media unavailable."
"""

from __future__ import annotations
import os
import base64


# ── Prompts ──────────────────────────────────────────────────────────────────

IMAGE_PROMPT = (
    "Describe this image in one sentence for a notification routing system. "
    "Focus on: content type, any text visible, urgency signals, promotional vs "
    "informational vs scam signals. Be brief."
)

VOICE_PROMPT = (
    "Transcribe this voice message in one sentence. "
    "Focus on: main request or topic, any urgency, tone. Be brief."
)

UNAVAILABLE = "Media unavailable."


# ── Public API ────────────────────────────────────────────────────────────────

def process_media(
    media_type: str,
    media_path: str | None,
    gemini_client,
    model_name: str = "gemini-2.0-flash-001",
) -> str:
    """
    Returns a one-sentence summary string.

    Args:
        media_type:    "image", "voice", or "" (text)
        media_path:    absolute filesystem path, or None if not found
        gemini_client: initialised google.genai.Client instance
        model_name:    Gemini model to use for multimodal calls
    """
    if not media_type or media_type not in ("image", "voice"):
        return ""

    if not media_path or not os.path.exists(media_path):
        return UNAVAILABLE

    try:
        file_bytes = _read_bytes(media_path)
        if not file_bytes:
            return UNAVAILABLE

        if media_type == "image":
            return _describe_image(file_bytes, media_path, gemini_client, model_name)
        else:
            return _transcribe_voice(file_bytes, media_path, gemini_client, model_name)

    except Exception as exc:
        print(f"[WARN] Media processing failed for {media_path}: {exc}")
        return UNAVAILABLE


# ── Internals ─────────────────────────────────────────────────────────────────

def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _mime_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    if lower.endswith(".ogg"):
        return "audio/ogg"
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".m4a"):
        return "audio/mp4"
    return "application/octet-stream"


def _describe_image(
    file_bytes: bytes,
    path: str,
    client,
    model_name: str,
) -> str:
    from google import genai
    from google.genai import types

    mime = _mime_type(path)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime),
            IMAGE_PROMPT,
        ],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=120),
    )
    return response.text.strip() if response.text else UNAVAILABLE


def _transcribe_voice(
    file_bytes: bytes,
    path: str,
    client,
    model_name: str,
) -> str:
    from google import genai
    from google.genai import types

    mime = _mime_type(path)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime),
            VOICE_PROMPT,
        ],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=120),
    )
    return response.text.strip() if response.text else UNAVAILABLE
