"""Verbatim speech-to-text for Taglish audio using faster-whisper.

The Whisper model is loaded once and cached per (size, device, compute_type)
combination so repeated runs in the same container are fast.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from faster_whisper import WhisperModel


# Whisper's language codes. "tl" is Tagalog; Taglish is code-switched, so
# leaving it on auto-detect is usually the safest default.
LANGUAGE_CHOICES = {
    "Auto-detect (recommended for Taglish)": None,
    "Tagalog (tl)": "tl",
    "English (en)": "en",
}

# A short prompt nudges Whisper toward Taglish spelling/output instead of
# translating or "normalizing" into pure English or pure Tagalog.
DEFAULT_INITIAL_PROMPT = (
    "Taglish conversation sa Pilipinas. Mixed Tagalog and English. "
    "Transcribe verbatim, huwag i-translate. Example: 'So guys, kailangan "
    "natin i-check yung deployment bukas ng umaga.'"
)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str  # one segment per line (verbatim)
    segments: List[Segment] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None

    @property
    def timestamped(self) -> str:
        return "\n".join(
            f"[{_fmt(s.start)} -> {_fmt(s.end)}] {s.text}" for s in self.segments
        )


def _fmt(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - total) * 1000)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


_MODEL_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def load_model(
    model_size: str,
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    key = (model_size, device, compute_type)
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            download_root = os.getenv("WHISPER_DOWNLOAD_ROOT", "/models")
            _MODEL_CACHE[key] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )
        return _MODEL_CACHE[key]


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    beam_size: int = 5,
    progress_cb=None,
) -> Transcript:
    """Transcribe an audio file and return verbatim text + segments.

    progress_cb, if given, is called with a status string.
    """
    model = load_model(model_size, device=device, compute_type=compute_type)
    if progress_cb:
        progress_cb(f"Transcribing with {model_size} ({device}/{compute_type})...")

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",  # never translate
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=False,  # avoids repetition/hallucination loops
        initial_prompt=initial_prompt or None,
        word_timestamps=False,
    )

    segments: List[Segment] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append(Segment(start=seg.start, end=seg.end, text=text))
        if progress_cb and len(segments) % 25 == 0 and len(segments) > 0:
            progress_cb(f"Transcribed {len(segments)} segments...")

    transcript = Transcript(
        text="\n".join(s.text for s in segments),
        segments=segments,
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration=getattr(info, "duration", None),
    )
    if progress_cb:
        progress_cb(
            f"Transcribed {len(segments)} segments "
            f"(detected: {transcript.language}, "
            f"{transcript.duration or 0:.1f}s audio)."
        )
    return transcript
