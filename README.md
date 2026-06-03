# 西比莉娜 · Lina

An AI character — 18-year-old alchemist's apprentice in a pre-industrial world with magic — running on the Claude API, with both a CLI and a web GUI for testing.

The character's personality, world, hobbies, and sample conversations are stored as plain markdown in `static/`. Edit those files and restart to reshape the character.

## Highlights

- **Strict in-character**: behavior rules in the system prompt enforce no AI self-reference, no post-1760 knowledge, no breaking the fourth wall.
- **Prompt caching**: the large character corpus is sent once and cached at the API level for cheap subsequent turns.
- **RAG**: a small BM25 index (character-bigram tokenization, no extra deps) retrieves relevant slices of `personality.md` / `hobbies.md` / `others.md` / `sample_conversations.md` based on each user turn.
- **Persistent history**: each session is stored as a JSON file in `conversations/`.
- **Voice (web GUI)**: optional push-to-talk loop with fully-local GPU ASR + streaming TTS — Lina starts speaking after her first sentence while the rest is still generating.
- **Multi-user ready**: per-visitor API keys and per-browser session isolation, served behind gunicorn for public deployment.

## Files

```
static/                    # character data — edit these to reshape her
  person_setup.md          ← always in system prompt (core identity)
  world.md                 ← always in system prompt (world setting)
  personality.md           ← RAG-indexed
  hobbies.md               ← RAG-indexed
  others.md                ← RAG-indexed
  sample_conversations.md  ← RAG-indexed

app/
  rag.py                   # BM25 retrieval over static files
  character.py             # system prompt + Claude API call
  conversation.py          # session persistence
  config.py                # API key resolution
  cli.py                   # CLI REPL
  web.py                   # Flask web server
  voice.py                 # local ASR + streaming TTS pipeline
  templates/chat.html      # web GUI

conversations/             # auto-created, holds per-session JSON
gunicorn_conf.py           # production server config (public / multi-user)
run_cli.py
run_web.py
```

## Install

```bash
pip install -r requirements.txt
```

## API key

**CLI** resolves the key in this order:
1. `--api-key` on the command line
2. `ANTHROPIC_API_KEY` environment variable
3. `~/.lina_key` file (single line)
4. Interactive prompt

**Web GUI** does *not* auto-load a key at startup — it boots unauthenticated so
it's safe to expose to multiple users. Each visitor pastes their own Anthropic
key into the top-right card and clicks 连接; the key is validated against the API
before it's accepted, and is used only for that visitor (see *Deployment*).

## Run — CLI

```bash
python run_cli.py
# or with explicit key / model / session id
python run_cli.py --api-key sk-... --model claude-opus-4-7 --session my-test
```

In-session commands: `/help`, `/new [id]`, `/load <id>`, `/list`, `/reset`, `/history`, `/context`, `/model <name>`, `/quit`.

## Run — Web GUI

```bash
python run_web.py             # http://127.0.0.1:8000
python run_web.py --port 8080
```

The page has a sidebar of past sessions, a chat area, and a right-hand inspector that shows the RAG chunks retrieved for each turn plus token-usage stats (so you can see prompt caching working).

The server starts unauthenticated — paste your Anthropic key into the top-right card and click 连接 (it's validated before it's accepted). `run_web.py` uses the Flask dev server; for a public or multi-user setup use gunicorn instead (see *Deployment*).

## Voice (web GUI)

The web GUI has a push-to-talk **mic button** in the composer for a fully-local
voice loop — no cloud speech APIs, everything runs on your GPU(s):

- **ASR**: `Qwen/Qwen3-ASR-1.7B` (default `cuda:0`)
- **TTS**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (default `cuda:1`)

**How it flows.** Click 🎤 to start talking, click again to stop. The browser
records and encodes a 16 kHz mono WAV client-side (so no `ffmpeg` is needed
server-side), uploads it to `/api/voice/transcribe`, and the recognized text
is sent to `/api/voice/chat` — a **Server-Sent Events** stream. Claude's reply
streams out token-by-token; the server splits it into sentences and
**synthesizes + streams each sentence's audio the moment it completes**, so Lina
starts speaking after her first line instead of after the whole reply. Her
parsed `[mood: …]` tag drives the TTS delivery style.

**Barge-in.** While she's responding, clicking 🎤 again **interrupts**: the
stream is aborted, playback stops, and the in-flight turn is discarded on both
sides (persisted to neither) — on the assumption you misspoke — and recording
restarts immediately.

The models **pre-load in a background thread when the server starts** (~30 s);
the status pill next to the composer shows progress and flips to 就绪 when
ready. Set `LINA_VOICE_PRELOAD=0` to load lazily on first mic click instead.
Override defaults via env:

```bash
LINA_ASR_DEVICE=cuda:0  LINA_TTS_DEVICE=cuda:1 \
LINA_TTS_SPEAKER=serena LINA_TTS_LANGUAGE=Chinese \
python run_web.py
```

Requires a CUDA GPU plus `torch`, `qwen-asr`, `qwen-tts` (see `requirements.txt`).
If those aren't installed the rest of the app runs normally; only the mic is
disabled.

> Note: the mic needs a **secure context** — it works on `localhost`, or over
> HTTPS (see *Deployment*). Over plain `http://<ip>` the browser blocks it.

## Deployment (public / multi-user)

For anything beyond a single local user, run behind **gunicorn** rather than the
Flask dev server (which spawns an unbounded thread per connection and gets
exhausted by internet scanners within hours, eventually becoming unreachable):

```bash
gunicorn -c gunicorn_conf.py        # binds 0.0.0.0:7000 over HTTPS using certs/
```

- **HTTPS is required for the mic.** Browsers only expose the microphone in a
  secure context. `gunicorn_conf.py` serves a cert/key — point `LINA_SSL_CERT` /
  `LINA_SSL_KEY` at real files, or generate a self-signed pair into `certs/`:
  ```bash
  mkdir -p certs && openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout certs/key.pem -out certs/cert.pem -subj "/CN=lina" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
  ```
  A bare IP + self-signed cert means a one-time browser warning; a domain with a
  real cert (e.g. via Caddy/nginx in front) removes it.
- **Single worker, many threads.** The GPU models and all in-memory state live
  in one process, so the config pins `workers = 1` with a thread pool
  (`LINA_THREADS`, default 16). The single GPU serializes ASR/TTS, so heavy
  simultaneous voice use queues — fine for a small tester group.
- **Per-user API keys.** The server boots unauthenticated; each visitor enters
  their own Anthropic key (validated on connect) and it's used only for them —
  one user connecting never affects another's key.
- **Per-browser sessions.** Each browser gets a random id (localStorage, sent as
  `X-Client-Id`) so testers only see their own sessions in the sidebar, and a
  fresh visitor starts a new session rather than landing in someone else's. This
  is a convenience boundary for testing, not a hard security control.
- Restrict the port with a firewall to your testers' IPs to minimize exposure.

Env knobs: `LINA_BIND` (default `0.0.0.0:7000`), `LINA_THREADS`, `LINA_TIMEOUT`,
`LINA_SSL_CERT` / `LINA_SSL_KEY`.

## Models

Default is `claude-sonnet-4-6`. You can switch to `claude-opus-4-7` for higher quality or `claude-haiku-4-5-20251001` for cheaper/faster testing — via the `--model` flag in CLI or the model dropdown in the GUI.

## Tuning the character

Open the files in `static/` and edit. The core identity (`person_setup.md`, `world.md`) is always in the system prompt; everything else is retrieved per-turn — so you can grow `sample_conversations.md` indefinitely without bloating every API call.

The behavior rules (knowledge boundary, AI self-reference ban, speaking style) live in `BEHAVIOR_RULES` inside `app/character.py`.
