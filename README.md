# Taglish Transcriber

Upload (or record) Taglish audio and get:

1. **Verbatim transcript** — speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
2. **Cleaned-up Taglish** — the verbatim text is edited for readability by
   **DeepSeek V4.1 Flash** through the **OpenCode Go** API.

Everything runs in Docker. The only data that leaves your machine is the
transcript text sent to OpenCode Go for the cleanup pass.

## Requirements

- Docker Desktop (with Docker Compose).
- An OpenCode Go API key (starts with `sk-`).

## Setup

1. Put your key in `.env` (already created for you):

   ```dotenv
   OPENCODE_API_KEY=sk-...
   ```

2. Build and start:

   ```powershell
   cd "C:\Users\TOP C\taglish-transcriber"
   docker compose up --build
   ```

3. Open <http://localhost:7860> in your browser.

The first run downloads the Whisper `large-v3` weights (~3 GB) into a Docker
volume, so it takes a while. Later runs reuse the cached weights.

## Using the app

- **Audio** — upload a file (mp3, wav, m4a, ogg, video, …) or use the mic.
- **Spoken language** — leave on *Auto-detect* for Taglish.
- **Transcribe & Clean** — runs both steps.
- **Transcribe only** — verbatim output only (no API call).
- **Re-clean verbatim** — re-run the cleanup on the text currently in the
  verbatim box, e.g. after tweaking the text by hand.

Results are also written to `outputs/<timestamp>_<name>/` on your host:
`verbatim.txt`, `timestamps.txt`, and `cleaned.txt`.

## Configuration

All settings live in `.env`:

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENCODE_API_KEY` | – | Required for the cleanup pass. |
| `OPENCODE_BASE_URL` | `https://opencode.ai/zen/go/v1` | OpenCode Go OpenAI-compatible endpoint. |
| `OPENCODE_MODEL` | `deepseek-v4.1-flash` | Any Go model id served on `/chat/completions`. |
| `OPENCODE_REASONING_EFFORT` | `none` | Disables the model's reasoning pass (`none`, `minimal`, `low`, `medium`, `high`). |
| `OPENCODE_CHUNK_CHARS` | `6000` | Transcript chunk size for long recordings. |
| `OPENCODE_MAX_TOKENS` | `8192` | Max output tokens per chunk. |
| `HF_TOKEN` | – | Optional; authenticates Whisper weight downloads. |
| `WHISPER_MODEL` | `large-v3` | Default model (overridable in the UI). |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda`. |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` for CPU, `float16` for GPU. |

## Performance notes (CPU)

`large-v3` on CPU is accurate but slow — roughly a few times the audio length
on a modern laptop. If you need faster iteration, pick `medium` or
`distil-large-v3` in the Advanced panel.

## Troubleshooting

- **`Model returned empty content`** — raise `OPENCODE_MAX_TOKENS`; the model
  is a reasoning model and can spend the budget on reasoning.
- **Slow first request** — the model download is still in progress. Watch the
  status box and `docker compose logs -f`.
- **Cleanup skipped warning** — `OPENCODE_API_KEY` is missing in `.env`;
  restart the container after adding it.
