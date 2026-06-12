"""Config & API key resolution."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
CONVERSATIONS_DIR = PROJECT_ROOT / "conversations"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"
# 逐条消息的点赞/点踩反馈（一会话一 JSON）。
MESSAGE_FEEDBACK_DIR = PROJECT_ROOT / "message_feedback"


def resolve_admin_users() -> set[str]:
    """管理员用户名集合。来自环境变量 LINA_ADMIN_USERS，逗号/空格分隔。

    只有这些用户名登录后才能看到「汇总」统计页（问卷汇总 + 逐条点赞点踩汇总）。
    未配置则返回空集合 —— 此时没有任何人能看汇总，符合「统计仅管理员可见」的默认安全姿态。
    """
    raw = os.environ.get("LINA_ADMIN_USERS", "")
    return {u.strip() for u in re.split(r"[,\s]+", raw) if u.strip()}


def resolve_api_access_keys() -> set[str]:
    """OpenAI 兼容接口 /v1/chat/completions 的访问 key 集合。

    来自环境变量 LINA_API_KEYS，逗号/空格分隔，可配多个。
    - 未配置（空集合）→ **不鉴权**，任何人可调（适合内网/调试）。
    - 配置了 → 调用方必须在 Authorization: Bearer <key> 里带其中一个，否则 401。
    """
    raw = os.environ.get("LINA_API_KEYS", "")
    return {k.strip() for k in re.split(r"[,\s]+", raw) if k.strip()}

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


# Controller LLM provider settings. Anthropic exposes an OpenAI-compatible
# endpoint, so the same OpenAI-SDK-based controller can talk to Claude.
ANTHROPIC_OPENAI_COMPAT_BASE_URL = "https://api.anthropic.com/v1/"
_DEFAULT_OPENAI_CONTROLLER_MODEL = "gpt-5-mini"
_DEFAULT_ANTHROPIC_CONTROLLER_MODEL = "claude-haiku-4-5-20251001"


def resolve_controller_settings() -> dict:
    """决定 controller（规则层之上的 LLM 顾问 / 回复切分 / 自我记忆）用哪个 provider。

    **一律走 Claude**：所有 controller 的 LLM 请求都打到 Anthropic（默认 Claude
    Haiku，走其 OpenAI 兼容端点），**绝不向真正的 OpenAI 发请求**。因此：
    - 有 Anthropic key → anthropic。
    - 没有 Anthropic key → none（退到只走规则层）；**不再回落到 OpenAI**，
      即便设置了 OPENAI_API_KEY 也不会用。
    - LINA_CONTROLLER_PROVIDER / OPENAI_API_KEY 不再影响 provider 选择（保留环境
      变量只是为了兼容旧配置，不会再触发 OpenAI 调用）。

    覆盖项：LINA_CONTROLLER_MODEL、LINA_CONTROLLER_BASE_URL。
    返回 {provider, api_key, model, base_url}。
    """
    anthropic_key = resolve_api_key()
    model_override = (os.environ.get("LINA_CONTROLLER_MODEL") or "").strip() or None
    base_override = (os.environ.get("LINA_CONTROLLER_BASE_URL") or "").strip() or None
    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "model": model_override or _DEFAULT_ANTHROPIC_CONTROLLER_MODEL,
            "base_url": base_override or ANTHROPIC_OPENAI_COMPAT_BASE_URL,
        }
    return {
        "provider": "none",
        "api_key": None,
        "model": model_override or _DEFAULT_ANTHROPIC_CONTROLLER_MODEL,
        "base_url": base_override,
    }


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
        "engage_base_ms": _num("LINA_ENGAGE_BASE_MS", 20000),
        "engage_multiplier": _num("LINA_ENGAGE_MULTIPLIER", 1.4),
        "engage_jitter": _num("LINA_ENGAGE_JITTER", 0.3),
        "max_nudges": int(_num("LINA_MAX_NUDGES", 4)),
    }
