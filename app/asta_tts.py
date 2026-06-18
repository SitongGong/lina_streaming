"""HTTP client for the remote *asta* TTS engine (Qwen3-TTS, discrete emotions).

Text-chat replies are voiced sentence-by-sentence with a fixed emotion (default
"joy") by POSTing to the asta TTS service's ``/tts`` endpoint. Unlike
``app.voice`` (local-GPU qwen-tts CustomVoice), this needs **no GPU** on the host
running lina — it just calls the already-deployed TTS over HTTP. That means it
works both in local dev (via an SSH tunnel to the aliyun container) and on the
aliyun box itself (same container, so plain ``localhost``).

Config (env):
  LINA_ASTA_TTS_URL       base URL of the asta TTS engine. Default
                          ``http://127.0.0.1:5002`` — the port the engine listens
                          on inside the aliyun container; reachable locally via
                          ``ssh -L 5002:localhost:5002 <aliyun>``.
  LINA_ASTA_TTS_EMOTION   fixed emotion label. Default ``joy``. One of:
                          joy/sad/angry/disgust/embarrased/surprised/wonder/neutral.
  LINA_ASTA_TTS_LANGUAGE  TTS language. Default ``Chinese``.
  LINA_ASTA_TTS_TIMEOUT   per-request timeout seconds. Default ``30``.

The engine returns ``audio/wav`` (24 kHz, mono, int16 PCM).
"""

from __future__ import annotations

import json
import os
import urllib.request

ASTA_TTS_URL = os.environ.get("LINA_ASTA_TTS_URL", "http://127.0.0.1:5002").rstrip("/")
ASTA_TTS_EMOTION = os.environ.get("LINA_ASTA_TTS_EMOTION", "joy")
ASTA_TTS_LANGUAGE = os.environ.get("LINA_ASTA_TTS_LANGUAGE", "Chinese")
_TIMEOUT = float(os.environ.get("LINA_ASTA_TTS_TIMEOUT", "30"))


def synthesize(text: str, emotion: str | None = None) -> bytes:
    """Synthesize one chunk of text → WAV bytes (24 kHz mono int16).

    Returns b"" for empty text. Raises on network/HTTP error — callers treat
    TTS as best-effort and stay silent on failure (the text reply is already
    shown to the user).
    """
    text = (text or "").strip()
    if not text:
        return b""
    payload = json.dumps(
        {
            "text": text,
            "emotion": emotion or ASTA_TTS_EMOTION,
            "language": ASTA_TTS_LANGUAGE,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        ASTA_TTS_URL + "/tts",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def available() -> bool:
    """Best-effort liveness probe (GET /emotions). Never raises."""
    try:
        req = urllib.request.Request(ASTA_TTS_URL + "/emotions", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return getattr(resp, "status", 200) == 200
    except Exception:
        return False
