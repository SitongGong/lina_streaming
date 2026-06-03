#!/usr/bin/env python3
"""Launch the web GUI.

Examples:
    python run_web.py
    python run_web.py --port 8080
    ANTHROPIC_API_KEY=sk-... python run_web.py

    # Serve over HTTPS so the browser microphone works from another machine
    # (getUserMedia requires a secure context — https:// or localhost):
    python run_web.py --host 0.0.0.0 --port 7000 --ssl-cert certs/cert.pem --ssl-key certs/key.pem
    python run_web.py --host 0.0.0.0 --port 7000 --ssl-adhoc   # throwaway self-signed
"""

import argparse

from app.web import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Web GUI for chatting with 西比莉娜.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ssl-cert", default=None, help="Path to TLS certificate (PEM).")
    parser.add_argument("--ssl-key", default=None, help="Path to TLS private key (PEM).")
    parser.add_argument(
        "--ssl-adhoc",
        action="store_true",
        help="Serve HTTPS with a throwaway self-signed cert (needs `cryptography`).",
    )
    args = parser.parse_args()

    # Resolve the TLS context. HTTPS is what lets the mic work off-localhost,
    # because browsers only expose getUserMedia in a secure context.
    ssl_context = None
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
    elif args.ssl_adhoc:
        ssl_context = "adhoc"
    elif args.ssl_cert or args.ssl_key:
        parser.error("--ssl-cert and --ssl-key must be given together.")

    scheme = "https" if ssl_context else "http"
    app = create_app()
    print(f"\n  → 打开浏览器访问 {scheme}://{args.host}:{args.port}\n")
    # threaded=True: SSE voice replies are long-lived, so concurrent status /
    # transcribe / interrupt requests must be served on separate threads.
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        ssl_context=ssl_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
