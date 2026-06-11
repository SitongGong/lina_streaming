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
import logging
import os
from pathlib import Path


def _load_dotenv() -> None:
    """把项目根目录 .env 里的键值加载进 os.environ（不覆盖已有的环境变量）。
    项目没装 python-dotenv，这里自己读，避免新增依赖。controller 的
    OPENAI_API_KEY 就靠这一步进环境——否则 controller 会退化成只走规则层。"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

from app.web import create_app

# 让 controller 决策 trace（logging.INFO）能打到日志里，方便复盘每轮判断。
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


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
