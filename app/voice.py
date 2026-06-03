"""Local voice pipeline: ASR (speech→text) + TTS (text→speech), all on-GPU.

Wraps the locally-installed `qwen_asr` / `qwen_tts` inference packages:
  - ASR: Qwen3-ASR-1.7B               (default device cuda:0)
  - TTS: Qwen3-TTS-12Hz-1.7B-CustomVoice (default device cuda:1)

The two models live on separate GPUs so transcription and synthesis never
contend for the same device. Both are big, so loading is lazy and happens in
a background thread — the web layer polls `status()` and only enables the mic
once `state == "ready"`.

Design notes:
- Everything is a process-wide singleton (`get_voice_engine()`); the models
  are loaded once and reused for the life of the server.
- `transcribe()` takes a float32 mono waveform + sample rate (the browser
  records and WAV-encodes client-side, the web layer decodes the WAV with
  soundfile, so we never need ffmpeg here).
- `synthesize()` returns one sentence's audio as WAV bytes. The streaming
  win lives in web.py: it feeds Claude's reply to `iter_sentences()` and
  calls `synthesize()` per sentence so playback starts on sentence #1.
- GPU calls are serialized per-model with a lock; this is a single-user
  testing tool, not a throughput service.
"""

from __future__ import annotations

import io
import os
import re
import threading
import traceback

import numpy as np
import soundfile as sf


# ---- Configuration (env-overridable) ----
ASR_MODEL = os.environ.get("LINA_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
TTS_MODEL = os.environ.get("LINA_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
ASR_DEVICE = os.environ.get("LINA_ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("LINA_TTS_DEVICE", "cuda:1")
# Optional pinned speaker / language; if unset we pick sensible defaults from
# the model's own supported lists at load time.
TTS_SPEAKER = os.environ.get("LINA_TTS_SPEAKER") or None
TTS_LANGUAGE = os.environ.get("LINA_TTS_LANGUAGE", "Chinese")
# A short biasing context handed to the ASR model — proper nouns and domain
# words from Lina's world improve recognition of otherwise-rare terms.
ASR_CONTEXT = os.environ.get(
    "LINA_ASR_CONTEXT",
    "莉娜 西比莉娜 炼金术 魔法石 古代语 遗物 香草茶 戏剧",
)

# ASR sample rate the model expects internally; the package resamples for us,
# but we standardize the upload path on 16 kHz mono to keep clips small.
ASR_TARGET_SR = 16000


def iter_sentences(buffer: str, *, flush: bool = False) -> tuple[list[str], str]:
    """Split off complete sentences from a growing text `buffer`.

    Returns (complete_sentences, remainder). A sentence is considered complete
    at Chinese/Latin terminal punctuation or a newline. When `flush=True`
    (stream finished), whatever is left is returned as a final sentence.

    This is what lets TTS start on the first sentence while Claude is still
    generating the rest.
    """
    # Terminal punctuation we treat as a speakable boundary.
    boundary = re.compile(r"[。！？!?…\n]+")
    sentences: list[str] = []
    last = 0
    for m in boundary.finditer(buffer):
        end = m.end()
        chunk = buffer[last:end].strip()
        if chunk:
            sentences.append(chunk)
        last = end
    remainder = buffer[last:]
    if flush:
        tail = remainder.strip()
        if tail:
            sentences.append(tail)
        remainder = ""
    return sentences, remainder


def mood_to_instruct(mood: dict | None) -> str:
    """Turn Lina's parsed [mood: word | intensity | trust=N] into a natural
    language delivery instruction for the CustomVoice model.

    Kept gentle: we describe tone, not content. Empty string = neutral."""
    if not mood:
        return ""
    word = (mood.get("mood") or "").strip()
    if not word:
        return ""
    intensity = mood.get("intensity") or 5
    try:
        intensity = int(intensity)
    except (TypeError, ValueError):
        intensity = 5
    if intensity >= 8:
        degree = "非常"
    elif intensity >= 5:
        degree = "比较"
    else:
        degree = "略微"
    return f"用{degree}{word}的语气说话，像在和人面对面聊天，自然、口语化。"


class VoiceEngine:
    """Lazy-loaded singleton holding the ASR + TTS models."""

    def __init__(self) -> None:
        self._asr = None
        self._tts = None
        self._asr_lock = threading.Lock()
        self._tts_lock = threading.Lock()

        self._state = "idle"  # idle | loading | ready | error
        self._error: str | None = None
        self._state_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None

        self._speaker: str | None = TTS_SPEAKER
        self._tts_language: str = TTS_LANGUAGE
        self._tts_sample_rate: int = 24000  # corrected from model at load

    # ---- status ----
    def status(self) -> dict:
        with self._state_lock:
            return {
                "state": self._state,
                "error": self._error,
                "asr_model": ASR_MODEL,
                "tts_model": TTS_MODEL,
                "speaker": self._speaker,
                "language": self._tts_language,
            }

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._error = error

    # ---- loading ----
    def ensure_loading(self) -> dict:
        """Kick off a background load if we're idle. Idempotent."""
        with self._state_lock:
            if self._state in ("loading", "ready"):
                return {"state": self._state, "error": self._error}
            self._state = "loading"
            self._error = None
        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()
        return {"state": "loading", "error": None}

    def _load(self) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
            from qwen_tts import Qwen3TTSModel

            asr = Qwen3ASRModel.from_pretrained(
                ASR_MODEL,
                dtype=torch.bfloat16,
                device_map=ASR_DEVICE,
                max_new_tokens=512,
            )
            tts = Qwen3TTSModel.from_pretrained(
                TTS_MODEL,
                dtype=torch.bfloat16,
                device_map=TTS_DEVICE,
            )

            # Resolve a default speaker / language from the model's own lists.
            self._resolve_voice_defaults(tts)

            self._asr = asr
            self._tts = tts
            self._set_state("ready")

            # Warm up both models so the user's first real interaction doesn't
            # pay the cold-start kernel-compilation cost (~6s TTS / ~12s ASR).
            try:
                wav_bytes, _sr = self.synthesize("嗯。")
                import soundfile as _sf
                _wav, _rsr = _sf.read(io.BytesIO(wav_bytes), dtype="float32")
                self.transcribe(_wav, _rsr)
            except Exception:
                pass  # warmup is best-effort; never block readiness on it
        except Exception as e:  # noqa: BLE001 — surface any load failure to UI
            self._set_state("error", f"{e}\n{traceback.format_exc()}")

    def _resolve_voice_defaults(self, tts) -> None:
        # Speaker
        try:
            speakers = list(tts.model.get_supported_speakers() or [])
        except Exception:
            speakers = []
        if self._speaker and speakers:
            match = next((s for s in speakers if s.lower() == self._speaker.lower()), None)
            self._speaker = match or speakers[0]
        elif speakers:
            # Prefer a name hinting at a young female voice if present, else first.
            preferred = next(
                (s for s in speakers if re.search(r"female|woman|girl|女|少女", s, re.I)),
                None,
            )
            self._speaker = preferred or speakers[0]
        # Language
        try:
            langs = list(tts.model.get_supported_languages() or [])
        except Exception:
            langs = []
        if langs:
            match = next(
                (l for l in langs if l.lower() == (self._tts_language or "").lower()),
                None,
            )
            if match is None:
                match = next((l for l in langs if "chin" in l.lower() or "中文" in l), None)
            self._tts_language = match or langs[0]

    def _require_ready(self) -> None:
        with self._state_lock:
            if self._state != "ready":
                raise RuntimeError(f"语音模型尚未就绪（当前：{self._state}）。")

    # ---- ASR ----
    def transcribe(self, wav: np.ndarray, sample_rate: int) -> str:
        """Transcribe a float32 mono waveform. Returns recognized text."""
        self._require_ready()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1).astype(np.float32)
        with self._asr_lock:
            results = self._asr.transcribe(
                audio=(wav, int(sample_rate)),
                context=ASR_CONTEXT,
            )
        if not results:
            return ""
        return (getattr(results[0], "text", "") or "").strip()

    # ---- TTS ----
    def synthesize(self, text: str, instruct: str = "") -> tuple[bytes, int]:
        """Synthesize one chunk of text → (WAV bytes, sample_rate)."""
        self._require_ready()
        text = (text or "").strip()
        if not text:
            return b"", self._tts_sample_rate
        with self._tts_lock:
            wavs, sr = self._tts.generate_custom_voice(
                text=text,
                speaker=self._speaker,
                language=self._tts_language,
                instruct=instruct or None,
            )
        self._tts_sample_rate = int(sr)
        wav = wavs[0] if wavs else np.zeros(1, dtype=np.float32)
        return _wav_bytes(np.asarray(wav, dtype=np.float32), int(sr)), int(sr)


def _wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 [-1, 1] mono waveform to 16-bit PCM WAV bytes."""
    if wav.ndim > 1:
        wav = wav.reshape(-1)
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


_engine: VoiceEngine | None = None
_engine_lock = threading.Lock()


def get_voice_engine() -> VoiceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = VoiceEngine()
        return _engine
