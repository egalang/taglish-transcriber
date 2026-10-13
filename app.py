"""Taglish Transcriber - Gradio web UI.

Upload or record Taglish audio, get:
  1. a verbatim transcript (faster-whisper), and
  2. a cleaned-up Taglish version (OpenCode Go / DeepSeek V4.1 Flash).
"""

from __future__ import annotations

import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

from cleaner import CleanerError, api_key_present, clean_transcript
from transcriber import LANGUAGE_CHOICES, DEFAULT_INITIAL_PROMPT, transcribe

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE_CHOICES = [
    "large-v3",
    "distil-large-v3",
    "medium",
    "small",
    "base",
    "tiny",
]

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
DEFAULT_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEFAULT_OPENCODE_MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4.1-flash")

# Outputs yielded by the job generators: (status, verbatim, timestamps, cleaned, files)
Outputs = Tuple[str, str, str, str, Optional[List[str]]]


def _save_outputs(
    source_name: str,
    verbatim: str,
    timestamps: str,
    cleaned: str,
) -> List[str]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = Path(source_name).stem if source_name else "audio"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40]
    folder = OUTPUT_DIR / f"{stamp}_{stem}"
    folder.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for name, content in (
        ("verbatim.txt", verbatim),
        ("timestamps.txt", timestamps),
        ("cleaned.txt", cleaned),
    ):
        if content:
            path = folder / name
            path.write_text(content, encoding="utf-8")
            written.append(str(path))
    return written


def _run_job(
    audio_path: Optional[str],
    language_label: str,
    model_size: str,
    vad: bool,
    initial_prompt: str,
    opencode_model: str,
    do_clean: bool,
):
    """Generator that streams status updates while transcribing/cleaning."""
    status = ""
    verbatim = ""
    timestamps = ""
    cleaned = ""

    def emit():
        return (status, verbatim, timestamps, cleaned, None)

    if not audio_path:
        status = "Please upload or record an audio file first."
        yield emit()
        return

    if do_clean and not api_key_present():
        status = (
            "Warning: OPENCODE_API_KEY is not set, so cleanup is skipped. "
            "Add it to .env and restart to enable the cleanup pass."
        )
        do_clean = False

    q: "queue.Queue[tuple]" = queue.Queue()
    source_name = os.path.basename(audio_path)

    def worker():
        try:
            def cb(msg: str):
                q.put(("status", msg))

            language = LANGUAGE_CHOICES.get(language_label)
            cb(f"Loading model '{model_size}' ({DEFAULT_DEVICE}/{DEFAULT_COMPUTE})...")
            transcript = transcribe(
                audio_path,
                model_size=model_size,
                device=DEFAULT_DEVICE,
                compute_type=DEFAULT_COMPUTE,
                language=language,
                initial_prompt=initial_prompt,
                vad_filter=vad,
                progress_cb=cb,
            )
            q.put(("verbatim", transcript.text))
            q.put(("timestamps", transcript.timestamped))

            if do_clean:
                cb("Sending transcript to OpenCode Go for cleanup...")
                cleaned_text = clean_transcript(
                    transcript.text,
                    model=opencode_model,
                    progress_cb=cb,
                )
                q.put(("cleaned", cleaned_text))
            q.put(("done", None))
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            q.put(("error", f"{type(exc).__name__}: {exc}"))
            q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = q.get()
        if kind == "done":
            break
        if kind == "status":
            status = str(payload)
        elif kind == "verbatim":
            verbatim = str(payload)
        elif kind == "timestamps":
            timestamps = str(payload)
        elif kind == "cleaned":
            cleaned = str(payload)
        elif kind == "error":
            status = f"Error: {payload}"
        yield emit()

    files = _save_outputs(source_name, verbatim, timestamps, cleaned)
    if files:
        status = (status + f"\n\nSaved to {Path(files[0]).parent}").strip()
    yield (status, verbatim, timestamps, cleaned, files or None)


def on_run(audio, language_label, model_size, vad, initial_prompt, opencode_model):
    yield from _run_job(
        audio, language_label, model_size, vad, initial_prompt, opencode_model, True
    )


def on_transcribe_only(
    audio, language_label, model_size, vad, initial_prompt, opencode_model
):
    yield from _run_job(
        audio, language_label, model_size, vad, initial_prompt, opencode_model, False
    )


def on_reclean(verbatim, opencode_model):
    if not verbatim or not verbatim.strip():
        yield ("Nothing to clean - transcribe first.", "", "", "", None)
        return
    q: "queue.Queue[tuple]" = queue.Queue()

    def worker():
        try:
            cleaned = clean_transcript(
                verbatim,
                model=opencode_model,
                progress_cb=lambda m: q.put(("status", m)),
            )
            q.put(("cleaned", cleaned))
        except CleanerError as exc:
            q.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001
            q.put(("error", f"{type(exc).__name__}: {exc}"))
        q.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    cleaned = ""
    status = "Cleaning..."
    while True:
        kind, payload = q.get()
        if kind == "done":
            break
        if kind == "status":
            status = str(payload)
        elif kind == "cleaned":
            cleaned = str(payload)
            status = "Done."
        elif kind == "error":
            status = f"Error: {payload}"
        yield (status, cleaned)


with gr.Blocks(title="Taglish Transcriber") as demo:
    gr.Markdown(
        "# 🎙️ Taglish Transcriber\n"
        "Verbatim transcription with **faster-whisper**, then a cleaned-up "
        "Taglish version with **OpenCode Go (DeepSeek V4.1 Flash)**."
    )

    with gr.Row():
        with gr.Column(scale=2):
            audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio (upload a file or record)",
            )
            language = gr.Dropdown(
                choices=list(LANGUAGE_CHOICES.keys()),
                value=list(LANGUAGE_CHOICES.keys())[0],
                label="Spoken language",
            )
            opencode_model = gr.Textbox(
                value=DEFAULT_OPENCODE_MODEL,
                label="Cleanup model (OpenCode Go)",
            )

            with gr.Accordion("Advanced", open=False):
                model_size = gr.Dropdown(
                    choices=MODEL_SIZE_CHOICES,
                    value=DEFAULT_MODEL if DEFAULT_MODEL in MODEL_SIZE_CHOICES else "large-v3",
                    label="Whisper model",
                )
                vad = gr.Checkbox(
                    value=True,
                    label="Voice activity detection (reduces silence hallucination)",
                )
                initial_prompt = gr.Textbox(
                    value=DEFAULT_INITIAL_PROMPT,
                    lines=3,
                    label="Initial prompt (biases Whisper toward Taglish)",
                )

        with gr.Column(scale=3):
            status = gr.Textbox(label="Status", lines=3, interactive=False)
            verbatim = gr.Textbox(label="Verbatim transcript", lines=12)
            timestamps = gr.Textbox(label="Verbatim with timestamps", lines=6)
            cleaned = gr.Textbox(label="Cleaned-up Taglish", lines=12)
            files = gr.File(label="Downloads", file_count="multiple")

    with gr.Row():
        run_btn = gr.Button("Transcribe & Clean", variant="primary")
        transcribe_btn = gr.Button("Transcribe only")
        reclean_btn = gr.Button("Re-clean verbatim")

    inputs = [audio, language, model_size, vad, initial_prompt, opencode_model]
    outputs = [status, verbatim, timestamps, cleaned, files]

    run_btn.click(on_run, inputs=inputs, outputs=outputs)
    transcribe_btn.click(on_transcribe_only, inputs=inputs, outputs=outputs)
    reclean_btn.click(
        on_reclean,
        inputs=[verbatim, opencode_model],
        outputs=[status, cleaned],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        theme=gr.themes.Soft(),
    )
