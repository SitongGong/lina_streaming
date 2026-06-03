"""Gunicorn config for serving Lina on the public internet.

Run with:
    gunicorn -c gunicorn_conf.py

Why gunicorn instead of `python run_web.py`:
- The Flask/Werkzeug dev server spawns an UNBOUNDED thread per connection and
  has no idle timeouts, so internet scanners and half-open TLS probes pile up
  threads/file-descriptors over hours until the box is exhausted and the server
  becomes unreachable. Gunicorn uses a BOUNDED thread pool, enforces timeouts,
  and auto-restarts a dead worker.

Hard constraint — ONE worker:
- The app holds the GPU models (ASR on cuda:0, TTS on cuda:1) and all session /
  prompt state in process memory. Multiple workers would load the models once
  per worker (N× VRAM) and each get a separate, divergent copy of the state.
  So we run a single worker with many threads for concurrency. The single GPU
  serializes ASR/TTS under a lock, so heavy simultaneous voice use will queue —
  fine for a small tester group.

Everything below is env-overridable.
"""

import os

# The app factory. With this set, just run `gunicorn -c gunicorn_conf.py`.
wsgi_app = "app.web:create_app()"

bind = os.environ.get("LINA_BIND", "0.0.0.0:7000")

# MUST stay 1 — see the module docstring. Not env-overridable on purpose.
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("LINA_THREADS", "16"))

# Do NOT preload: the app must be imported (and CUDA initialized) inside the
# worker, after the fork — never in the master.
preload_app = False

# Worker-silence timeout. gthread heartbeats from its main loop independently of
# request handling, so long-lived SSE voice replies do NOT trip this.
timeout = int(os.environ.get("LINA_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# TLS — serve HTTPS directly (needed so the browser mic works). Point these at
# real cert files for a trusted cert; defaults to the bundled self-signed pair.
certfile = os.environ.get("LINA_SSL_CERT", "certs/cert.pem")
keyfile = os.environ.get("LINA_SSL_KEY", "certs/key.pem")
if not (certfile and os.path.exists(certfile) and keyfile and os.path.exists(keyfile)):
    # Fall back to plain HTTP if certs are missing (e.g. behind a TLS proxy).
    certfile = None
    keyfile = None

# Cap request line / header / field sizes so malformed scanner traffic can't
# allocate unbounded memory.
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 16380

accesslog = os.environ.get("LINA_ACCESS_LOG", "-")  # stdout
errorlog = os.environ.get("LINA_ERROR_LOG", "-")    # stderr
loglevel = os.environ.get("LINA_LOG_LEVEL", "info")
