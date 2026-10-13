"""Clean up verbatim Taglish ASR output with the OpenCode Go API.

Uses the OpenAI-compatible endpoint at https://opencode.ai/zen/go/v1 with
DeepSeek V4.1 Flash. Long transcripts are chunked on line boundaries, cleaned
chunk by chunk, and stitched back together.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Callable, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = """\
You are a professional Taglish (Filipino-English code-switching) transcript \
editor. You receive raw, verbatim speech-to-text output of a Taglish audio \
recording and return a cleaned-up, readable version of the SAME content.

Rules:
1. Preserve Taglish code-switching exactly as spoken: keep English words in \
English and Tagalog words in Tagalog. DO NOT translate anything to a single \
language.
2. Change the meaning as little as possible. Do not add, remove, summarize, or \
invent facts, names, numbers, or opinions.
3. Remove speech disfluencies and fillers (e.g. "uh", "um", "ah", "ano", \
"kasi po", "yung parang", repeated words, stutters) unless removing them \
changes the meaning.
4. Fix punctuation, capitalization, spacing, and sentence boundaries so it \
reads like clean written Taglish.
5. Fix obvious ASR mistakes using surrounding context (e.g. wrong homophones, \
garbled words), but keep proper nouns as spoken. If a word is unintelligible, \
keep the closest reasonable spelling rather than guessing wildly.
6. Keep the original order of ideas. Break the text into natural paragraphs \
where the topic shifts.
7. Output ONLY the cleaned Taglish text. No preamble, no notes, no \
explanations, no markdown fences, no translation.
"""

USER_TEMPLATE = """\
Clean up the following verbatim Taglish transcript. Return only the cleaned \
Taglish text.

--- BEGIN TRANSCRIPT ---
{chunk}
--- END TRANSCRIPT ---
"""


class CleanerError(RuntimeError):
    pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def api_key_present() -> bool:
    return bool(_env("OPENCODE_API_KEY"))


def _make_client(session_id: str) -> OpenAI:
    api_key = _env("OPENCODE_API_KEY")
    if not api_key:
        raise CleanerError(
            "OPENCODE_API_KEY is not set. Add it to .env and restart the app."
        )
    base_url = _env("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            # OpenCode Go asks clients to identify themselves and send a stable
            # session id per conversation (helps routing / prompt caching).
            "User-Agent": "taglish-cleaner/1.0",
            "x-opencode-session": session_id,
        },
    )


def _chunk_lines(text: str, max_chars: int) -> List[str]:
    """Group lines into chunks at or below max_chars, never splitting a line
    unless a single line exceeds the limit."""
    chunks: List[str] = []
    current: List[str] = []
    size = 0

    def flush():
        nonlocal current, size
        if current:
            chunks.append("\n".join(current))
            current = []
            size = 0

    for line in text.splitlines():
        line = line.rstrip()
        if len(line) > max_chars:
            flush()
            for i in range(0, len(line), max_chars):
                chunks.append(line[i : i + max_chars])
            continue
        if size + len(line) + 1 > max_chars:
            flush()
        current.append(line)
        size += len(line) + 1
    flush()
    return chunks


def _clean_chunk(
    client: OpenAI,
    model: str,
    chunk: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: Optional[str] = None,
    retries: int = 3,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(chunk=chunk)},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            last_err = exc
            # Some endpoints reject reasoning_effort; drop it and try again.
            if "reasoning_effort" in kwargs and "reasoning_effort" in str(exc).lower():
                kwargs.pop("reasoning_effort", None)
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise CleanerError(
                f"Cleanup request failed after {retries} attempts: {last_err}"
            )

        content = (resp.choices[0].message.content or "").strip()
        if content:
            return content

        # Empty content: the reasoning model most likely spent the whole token
        # budget on reasoning. Disable reasoning, then fall back to more tokens.
        if kwargs.get("reasoning_effort") != "none":
            kwargs["reasoning_effort"] = "none"
        else:
            kwargs["max_tokens"] = min(int(kwargs["max_tokens"]) * 2, 32768)
        if attempt < retries:
            time.sleep(1)
            continue

    raise CleanerError(
        "Model returned empty content (token budget likely consumed by "
        "reasoning). Increase OPENCODE_MAX_TOKENS."
    )


def clean_transcript(
    verbatim: str,
    model: Optional[str] = None,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.2,
    reasoning_effort: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Return the cleaned-up Taglish text for a verbatim transcript."""
    if not verbatim or not verbatim.strip():
        raise CleanerError("No transcript text to clean.")

    model = model or _env("OPENCODE_MODEL", "deepseek-v4.1-flash")
    max_chars = max_chars or int(_env("OPENCODE_CHUNK_CHARS", "6000"))
    max_tokens = max_tokens or int(_env("OPENCODE_MAX_TOKENS", "8192"))
    if reasoning_effort is None:
        reasoning_effort = _env("OPENCODE_REASONING_EFFORT", "none")
    session_id = f"taglish-{uuid.uuid4()}"

    client = _make_client(session_id)
    chunks = _chunk_lines(verbatim, max_chars)

    cleaned_parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        if progress_cb:
            progress_cb(f"Cleaning chunk {i}/{len(chunks)} with {model}...")
        cleaned_parts.append(
            _clean_chunk(
                client, model, chunk, max_tokens, temperature, reasoning_effort
            )
        )

    # Separate chunks with a blank line so paragraphs don't run together.
    return "\n\n".join(p for p in cleaned_parts if p)
