"""Config & API key resolution."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"

# ---- 可选账号登录（融合：登录用户走账号身份 + 独立目录；匿名用户走 client_id）----
USERS_DIR = PROJECT_ROOT / "users"
USERS_FILE = USERS_DIR / "users.json"
# 登录用户的会话存这里：conversations/users/<user_id>/，与匿名的平铺
# conversations/*.json 物理隔离。
USER_CONVERSATIONS_DIR = CONVERSATIONS_DIR / "users"


def resolve_secret_key() -> str:
    """Flask session 签名密钥。环境变量 LINA_SECRET_KEY 优先；否则读取/生成
    users/.secret_key（落盘复用，重启后已登录用户的 cookie 仍有效）。"""
    env = os.environ.get("LINA_SECRET_KEY")
    if env and env.strip():
        return env.strip()
    secret_path = USERS_DIR / ".secret_key"
    if secret_path.exists():
        try:
            content = secret_path.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    try:
        secret_path.write_text(token, encoding="utf-8")
    except Exception:
        pass
    return token


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve the API key from (in order): explicit arg, env var, ~/.lina_key file."""
    if explicit:
        return explicit.strip()
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env.strip()
    keyfile = Path.home() / ".lina_key"
    if keyfile.exists():
        try:
            content = keyfile.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception:
            pass
    return None


def resolve_openai_api_key(explicit: str | None = None) -> str | None:
    """解析 controller 用的 OpenAI key（默认走环境变量）。

    顺序：explicit 参数 → 环境变量 OPENAI_API_KEY → 不存在则返回 None。
    没有 key 时 controller 会自动退到「只走规则层 + fallback Plan」，
    所以这里返回 None 完全不影响 lina 正常聊天，只是少了 LLM 决策能力。
    """
    if explicit:
        return explicit.strip()
    env = os.environ.get("OPENAI_API_KEY")
    if env and env.strip():
        return env.strip()
    return None


def resolve_proactive_pacing() -> dict:
    """主动发言 / 续说的节奏参数（"magic numbers"）。

    全部可被环境变量覆盖，方便不改代码调体感。下发给前端驱动两套计时器：

    续说阶段（pending_segments 非空时）——快而稳，像连发微信：
      - continue_base_ms / continue_jitter

    主动发言阶段（段说完后）——指数退避 + 抖动，越没人理等得越久，最后告别：
      下一次等待 = engage_base_ms × engage_multiplier^已主动次数 × (1 ± jitter)
      - engage_base_ms      第一次主动前的基础等待
      - engage_multiplier   每多主动一次，间隔放大的倍数（>1 即递增/退避）
      - engage_jitter       ±比例随机抖动，破除机械感
      - max_nudges          主动到第几次改为告别并停止
    """
    def _num(env: str, default: float) -> float:
        raw = os.environ.get(env)
        if not raw or not raw.strip():
            return default
        try:
            return float(raw.strip())
        except ValueError:
            return default

    return {
        "continue_base_ms": _num("LINA_CONTINUE_BASE_MS", 3000),
        "continue_jitter": _num("LINA_CONTINUE_JITTER", 0.25),
        "engage_base_ms": _num("LINA_ENGAGE_BASE_MS", 30000),
        "engage_multiplier": _num("LINA_ENGAGE_MULTIPLIER", 1.4),
        "engage_jitter": _num("LINA_ENGAGE_JITTER", 0.3),
        "max_nudges": int(_num("LINA_MAX_NUDGES", 4)),
    }
