"""Flask web GUI for chatting with 西比莉娜.

Run via `python run_web.py` (see project root). The server keeps a single
shared CharacterEngine in memory once an API key is provided either via the
environment, ~/.lina_key, or the "/api/auth" endpoint at runtime.
"""

from __future__ import annotations

import base64
import difflib
import io
import os
import re
import threading
import time
import uuid
from pathlib import Path

import json

import anthropic
from flask import Flask, Response, jsonify, render_template, request, session

from . import character as character_mod
from .auth import UserStore
from .character import CharacterEngine, DEFAULT_MODEL
from .config import (
    CONVERSATIONS_DIR,
    PROJECT_ROOT,
    STATIC_DIR,
    USERS_FILE,
    resolve_api_key,
    resolve_openai_api_key,
    resolve_proactive_pacing,
    resolve_secret_key,
)
from .controller import LinaController, build_default_controller
from .conversation import ConversationStore
from .rag import retrieve_user_memory_chunks
from .self_facts import SelfFactsStore
from .voice import get_voice_engine, iter_sentences, mood_to_instruct


# ---- Engine pool: one CharacterEngine per pinned prompt version. The
# sentinel "_current_" represents the editable global overrides (the state
# shown in the 提示词 tab). Engines are created lazily on first chat and
# evicted when their underlying overrides change.
_CURRENT_KEY = "_current_"
# Credentials are PER CLIENT (per browser, keyed by X-Client-Id), not global —
# so on a shared public URL each tester uses their own API key and one tester
# connecting never overrides another's key. Engine-pool keys are prefixed with
# the client id (`<cid>\x00<prompt-source>`) so engines never cross clients.
_engine_pool: dict[str, CharacterEngine] = {}
_client_creds: dict[str, dict] = {}   # identity(user_id) -> {"key": str, "model": str}
_ANON_CLIENT = "_anon_"               # bucket for header-less (curl/admin) calls
_engine_lock = threading.Lock()
_store = ConversationStore(CONVERSATIONS_DIR)

# ---- 强制账号登录 ----
# 必须登录才能聊。登录态存 Flask 签名 cookie（session["user_id"]）。
# 登录后「身份」就是 user_id —— 会话按 user_id 打标隔离、检索按 user_id 隔离、
# 引擎凭证按 user_id 注册（统一用服务端 key，用户不再自己输 key）。
_user_store = UserStore(USERS_FILE)
# 莉娜对每个用户的「自我事实清单」（跨会话共享，按 user_id 存）。
_self_facts_store = SelfFactsStore(USERS_FILE.parent / "self_facts")


def _spawn_self_facts_update(user_id: str, current_facts: dict, slid_turns: list) -> None:
    """后台线程：让 controller LLM 把刚滑出窗口的几轮概括进该用户的自我事实清单。
    不阻塞 HTTP 响应。重新读一次清单避免覆盖并发写；store 自带锁。"""
    if not user_id or not slid_turns:
        return
    ctrl = _ensure_controller()
    if ctrl is None or not ctrl.has_llm:
        return

    def _work():
        try:
            base = _self_facts_store.load(user_id) or current_facts
            updated = ctrl.update_self_facts_sync(base, slid_turns)
            if updated is not None:
                _self_facts_store.save(user_id, updated)
        except Exception:
            pass  # 后台尽力而为，失败不影响主流程

    threading.Thread(target=_work, daemon=True).start()


def _spawn_self_facts_backfill(user_id: str) -> None:
    """登录后一次性回填：若清单还空、但该用户已有历史会话，把全部历史分批喂给
    controller LLM 提炼，补进自我事实清单。之后增量机制（滑出窗口）接管。
    后台线程跑，不阻塞登录响应；只在清单为空时跑，避免重复。"""
    if not user_id:
        return
    ctrl = _ensure_controller()
    if ctrl is None or not ctrl.has_llm:
        return
    if _self_facts_store.load(user_id):
        return  # 已有清单，不重复回填

    def _work():
        try:
            # 收集该用户所有会话的 (user, assistant) 对（跳过 system_trigger 占位）。
            pairs: list[tuple[str, str]] = []
            for conv in _store.iter_sessions():
                if getattr(conv, "client_id", None) != user_id:
                    continue
                pending_u = None
                for m in conv.messages:
                    if m.role == "user":
                        if m.meta and m.meta.get("system_trigger"):
                            continue
                        pending_u = (m.content or "").strip()
                    elif m.role == "assistant":
                        a = (m.content or "").strip()
                        pairs.append((pending_u or "", a))
                        pending_u = None
            if not pairs:
                return
            # 分批提炼（每批 6 轮），逐步合并进清单。
            facts: dict = {}
            BATCH = 6
            for i in range(0, len(pairs), BATCH):
                batch = pairs[i : i + BATCH]
                updated = ctrl.update_self_facts_sync(facts, batch)
                if updated is not None:
                    facts = updated
            if facts:
                _self_facts_store.save(user_id, facts)
        except Exception:
            pass

    threading.Thread(target=_work, daemon=True).start()
# 这些 /api 路径无需登录即可访问；其余 /api 一律要求已登录。
_PUBLIC_API_PATHS = {
    "/api/status",
    "/api/login",
    "/api/register",
    "/api/logout",
    "/api/whoami",
}

# Shared per-turn decision controller (rule layer + gpt-5-mini advisors).
# Stateless config — safe to share across all clients. Built once from
# OPENAI_API_KEY; `None`-LLM fallback is fine (rules-only still works).
_controller: LinaController | None = None


def _ensure_controller() -> LinaController | None:
    """Lazily build the shared LinaController. Safe to call repeatedly."""
    global _controller
    if _controller is not None:
        return _controller
    with _engine_lock:
        if _controller is None:
            _controller = build_default_controller(api_key=resolve_openai_api_key())
        return _controller

# ---------- Prompt-override layer ----------
# Editable from the web UI. Stored locally (gitignored), reapplied on
# engine creation and on every override edit.

PROMPT_COMPONENTS: list[tuple[str, str, str, str]] = [
    # (key, label, source, hint)
    ("person_setup.md", "角色核心设定", "file", "总是放进系统提示。最重要的身份信息。"),
    ("world.md", "世界观设定", "file", "总是放进系统提示。决定知识边界。"),
    ("sample_conversations.md", "示例对话", "file", "总是放进系统提示。她说话的腔调来源。"),
    ("personality.md", "人格问卷", "file", "RAG 检索：与用户话题相关时才出现。"),
    ("hobbies.md", "兴趣偏好", "file", "RAG 检索。"),
    ("others.md", "其他角色", "file", "RAG 检索。"),
    ("BEHAVIOR_RULES", "行为规则", "code", "代码常量。系统提示里的核心约束。"),
    ("MOOD_FORMAT_SPEC", "情绪标记格式", "code", "代码常量。决定 [mood: …] 的输出格式。"),
    ("SYSTEM_PROMPT_TEMPLATE", "系统提示模板", "code",
     "代码常量。包含占位符 {core_text} / {behavior_rules} / {mood_format_spec}。"),
]
_PROMPT_KEYS = {k for k, _, _, _ in PROMPT_COMPONENTS}

OVERRIDES_DIR = PROJECT_ROOT / "prompt_overrides"
OVERRIDES_FILE = OVERRIDES_DIR / "current.json"
VERSIONS_DIR = OVERRIDES_DIR / "versions"

_overrides: dict[str, str] = {}
_VERSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _new_version_id() -> str:
    """Sortable timestamp + short random suffix."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _version_path(version_id: str) -> Path | None:
    if not _VERSION_ID_RE.match(version_id):
        return None
    return VERSIONS_DIR / f"{version_id}.json"


def _save_version(name: str, note: str = "") -> dict:
    """Snapshot the current overrides dict as a new version file."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    vid = _new_version_id()
    record = {
        "version_id": vid,
        "name": (name or "").strip()[:80] or "（未命名）",
        "note": (note or "").strip()[:500],
        "created_at": time.time(),
        "overrides": dict(_overrides),
    }
    (VERSIONS_DIR / f"{vid}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def _load_version(version_id: str) -> dict | None:
    p = _version_path(version_id)
    if not p or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_versions() -> list[dict]:
    if not VERSIONS_DIR.exists():
        return []
    items: list[dict] = []
    for p in VERSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(
                {
                    "version_id": data.get("version_id", p.stem),
                    "name": data.get("name", ""),
                    "note": data.get("note", ""),
                    "created_at": data.get("created_at", p.stat().st_mtime),
                    "override_count": len(data.get("overrides", {})),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def _delete_version(version_id: str) -> bool:
    p = _version_path(version_id)
    if p and p.exists():
        p.unlink()
        return True
    return False


def _rename_version(version_id: str, name: str | None, note: str | None) -> dict | None:
    data = _load_version(version_id)
    if data is None:
        return None
    if name is not None:
        data["name"] = name.strip()[:80] or "（未命名）"
    if note is not None:
        data["note"] = note.strip()[:500]
    p = _version_path(version_id)
    if p is None:
        return None
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _diff_lines(a_text: str, b_text: str) -> list[dict]:
    """Align two texts line-by-line for side-by-side rendering."""
    a_lines = a_text.splitlines() if a_text else []
    b_lines = b_text.splitlines() if b_text else []
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    rows: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                rows.append({"a": a_lines[i1 + k], "b": b_lines[j1 + k], "type": "eq"})
        elif tag == "replace":
            la = a_lines[i1:i2]
            lb = b_lines[j1:j2]
            mx = max(len(la), len(lb))
            for k in range(mx):
                rows.append(
                    {
                        "a": la[k] if k < len(la) else None,
                        "b": lb[k] if k < len(lb) else None,
                        "type": "sub",
                    }
                )
        elif tag == "delete":
            for k in range(i2 - i1):
                rows.append({"a": a_lines[i1 + k], "b": None, "type": "del"})
        elif tag == "insert":
            for k in range(j2 - j1):
                rows.append({"a": None, "b": b_lines[j1 + k], "type": "add"})
    return rows


def _diff_versions(va: dict, vb: dict) -> dict:
    """Per-component diff between two saved versions."""
    overrides_a = va.get("overrides", {})
    overrides_b = vb.get("overrides", {})
    components: list[dict] = []
    for key, label, source, _hint in PROMPT_COMPONENTS:
        overridden_a = key in overrides_a
        overridden_b = key in overrides_b
        if not overridden_a and not overridden_b:
            components.append(
                {
                    "key": key,
                    "label": label,
                    "source": source,
                    "changed": False,
                    "a_overridden": False,
                    "b_overridden": False,
                    "rows": [],
                }
            )
            continue
        a_content = overrides_a.get(key, _default_value(key))
        b_content = overrides_b.get(key, _default_value(key))
        changed = a_content != b_content
        components.append(
            {
                "key": key,
                "label": label,
                "source": source,
                "changed": changed,
                "a_overridden": overridden_a,
                "b_overridden": overridden_b,
                "rows": _diff_lines(a_content, b_content) if changed else [],
            }
        )
    meta = lambda v: {
        "version_id": v.get("version_id"),
        "name": v.get("name", ""),
        "created_at": v.get("created_at", 0),
    }
    return {"a": meta(va), "b": meta(vb), "components": components}


def _load_overrides_from_disk() -> dict[str, str]:
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in _PROMPT_KEYS and isinstance(v, str)}
    except Exception:
        pass
    return {}


def _save_overrides_to_disk(d: dict[str, str]) -> None:
    OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDES_FILE.write_text(
        json.dumps(d, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sanitize_forced_state(fs: dict) -> dict:
    """Whitelist + clamp the user-submitted forced_state.

    Empty/invalid input yields {}, which the caller treats as a clear."""
    cleaned: dict = {}
    mood = fs.get("mood")
    if isinstance(mood, str) and mood.strip():
        cleaned["mood"] = mood.strip()[:40]
    for k in ("intensity", "trust"):
        v = fs.get(k)
        if v is None or v == "":
            continue
        try:
            iv = int(v)
        except (ValueError, TypeError):
            continue
        cleaned[k] = max(1, min(10, iv))
    return cleaned


def _default_value(key: str) -> str:
    if key.endswith(".md"):
        fpath = STATIC_DIR / key
        return fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    return {
        "BEHAVIOR_RULES": character_mod.BEHAVIOR_RULES,
        "MOOD_FORMAT_SPEC": character_mod.MOOD_FORMAT_SPEC,
        "SYSTEM_PROMPT_TEMPLATE": character_mod.SYSTEM_PROMPT_TEMPLATE,
    }.get(key, "")


def _client_ready(cid: str | None) -> bool:
    return cid is not None and cid in _client_creds


def _validate_key(key: str, model: str) -> tuple[bool, str | None]:
    """Verify an API key really works before accepting it, so a wrong key is
    rejected at connect time instead of silently failing on the first chat.

    A 1-token request is the cheapest probe. We only hard-reject on an auth
    error; transient/other failures are allowed through rather than locking a
    user out over a hiccup."""
    try:
        anthropic.Anthropic(api_key=key).messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, None
    except anthropic.AuthenticationError:
        return False, "API Key 无效或已过期。"
    except anthropic.PermissionDeniedError:
        return False, "该 API Key 无权访问。"
    except anthropic.NotFoundError:
        # Bad model name, but the key itself authenticated fine.
        return True, None
    except Exception:
        # Rate limit / network / etc. — don't block on a transient error.
        return True, None


def _client_id() -> str | None:
    """当前身份 = 登录用户的 user_id（来自签名 cookie）。

    强制登录后，整套下游（会话隔离、跨会话记忆、引擎凭证、引擎池 key）都以
    user_id 作为「身份」。未登录返回 None —— before_request 守卫会先把未登录
    的 /api 请求挡在外面，所以受守卫保护的端点里这里必定非空。"""
    return session.get("user_id")


def _cross_session_memory(cid: str | None, current_session_id: str, query: str, k: int = 3):
    """跨会话长期记忆：检索**同一 client_id** 的其它会话里讲过的事。

    严格按 client_id 过滤 → 不同用户（不同浏览器 id）的会话物理隔离，
    一个用户绝不会检索到另一个用户的记忆。cid 为空（匿名/curl）时不检索，
    避免把所有匿名会话混在一起。"""
    if not cid or not query.strip():
        return []
    own = [c for c in _store.iter_sessions() if getattr(c, "client_id", None) == cid]
    if not own:
        return []
    return retrieve_user_memory_chunks(own, query, k=k, current_session_id=current_session_id)


def _set_credentials(cid: str, api_key: str, model: str = DEFAULT_MODEL) -> None:
    """Set/replace one client's credentials and drop just that client's cached
    engines, so they rebuild with the new key/model. Other clients untouched."""
    with _engine_lock:
        _client_creds[cid] = {"key": api_key, "model": model}
        prefix = cid + "\x00"
        for k in [k for k in _engine_pool if k.startswith(prefix)]:
            _engine_pool.pop(k, None)


def _effective_overrides_for(conv) -> dict[str, str]:
    """Resolve a session to its effective overrides dict, taking mode into
    account. Private mode uses the session's own dict; shared mode follows
    the version pin (with a fallback to current globals if missing)."""
    if conv.prompt_mode == "private":
        return dict(conv.prompt_overrides or {})
    if conv.prompt_version_id:
        data = _load_version(conv.prompt_version_id)
        if data is not None:
            return dict(data.get("overrides", {}))
    return dict(_overrides)


def _engine_base_key(conv) -> str:
    """The prompt-source part of the pool key (shared across clients only in
    name; the full key is per-client). Private sessions get their own slot;
    shared sessions collapse onto the version (or _current_)."""
    if conv.prompt_mode == "private":
        return f"sess:{conv.session_id}"
    if conv.prompt_version_id and _load_version(conv.prompt_version_id) is not None:
        return conv.prompt_version_id
    return _CURRENT_KEY


def _engine_for_session(conv, cid: str) -> CharacterEngine | None:
    """Returns a cached engine for this client + session's prompt mode/source,
    built with THAT client's API key and model."""
    creds = _client_creds.get(cid)
    if not creds:
        return None
    key = cid + "\x00" + _engine_base_key(conv)
    with _engine_lock:
        engine = _engine_pool.get(key)
        if engine is not None:
            return engine
        engine = CharacterEngine(
            api_key=creds["key"],
            static_dir=STATIC_DIR,
            model=creds["model"],
            overrides=_effective_overrides_for(conv),
            controller=_ensure_controller(),
        )
        _engine_pool[key] = engine
        return engine


def _pop_engines_with_suffix(suffix: str) -> None:
    """Drop pooled engines whose key ends with `suffix`, across ALL clients —
    used when a shared prompt source changes (affects everyone using it)."""
    with _engine_lock:
        for k in [k for k in _engine_pool if k.endswith(suffix)]:
            _engine_pool.pop(k, None)


def _invalidate_current_engine() -> None:
    """Drop every client's engine using the editable global overrides."""
    _pop_engines_with_suffix("\x00" + _CURRENT_KEY)


def _invalidate_engine(version_id: str) -> None:
    _pop_engines_with_suffix("\x00" + version_id)


def _invalidate_session_engine(session_id: str) -> None:
    """Drop the engine for a private session after its overrides change."""
    _pop_engines_with_suffix("\x00sess:" + session_id)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static_web"),
    )
    # 签名 cookie 密钥（登录态存这里）。
    app.secret_key = resolve_secret_key()

    # 强制登录：除少数公开端点外，所有 /api/ 都要求已登录。
    @app.before_request
    def _enforce_login():
        p = request.path
        if p.startswith("/api/") and p not in _PUBLIC_API_PATHS:
            if not session.get("user_id"):
                return jsonify({"ok": False, "error": "请先登录。"}), 401
            # 登录后仍需各自连接 API Key（自带 key，或用 magic「0」走服务端 key），
            # 见 /api/auth。这里不再自动注册服务端 key。

    # Load any persisted overrides from the gitignored local folder.
    global _overrides
    _overrides = _load_overrides_from_disk()

    # Build the shared controller eagerly so any startup error (bad OpenAI
    # key, missing dep) shows in the server log, not on the first chat.
    _ensure_controller()

    # Pre-load the ASR/TTS models at startup (in a background thread) so the
    # mic is ready without a cold ~30s wait on the first click. Set
    # LINA_VOICE_PRELOAD=0 to keep the old lazy-on-click behavior.
    if os.environ.get("LINA_VOICE_PRELOAD", "1") not in ("0", "false", "False", ""):
        try:
            get_voice_engine().ensure_loading()
        except Exception:
            pass

    @app.route("/")
    def index():
        return render_template("chat.html")

    # ---------- 账号登录 / 注册 ----------

    @app.route("/api/whoami", methods=["GET"])
    def whoami():
        uid = session.get("user_id")
        return jsonify({"logged_in": bool(uid), "user_id": uid, "username": session.get("username")})

    @app.route("/api/register", methods=["POST"])
    def register():
        data = request.get_json(force=True, silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        ok, result = _user_store.register(username, password)
        if not ok:
            return jsonify({"ok": False, "error": result}), 400
        # 注册即登录。
        session["user_id"] = result
        session["username"] = username
        session.permanent = True
        # 不自动注册服务端 key：登录后用户需自行在 /api/auth 连接 API Key（或「0」）。
        return jsonify({"ok": True, "user_id": result, "username": username})

    @app.route("/api/login", methods=["POST"])
    def login():
        data = request.get_json(force=True, silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        user_id = _user_store.verify(username, password)
        if not user_id:
            return jsonify({"ok": False, "error": "用户名或密码错误"}), 401
        session["user_id"] = user_id
        session["username"] = username
        session.permanent = True
        # 登录后一次性回填：把该用户的历史会话提炼进自我事实清单（仅清单空时）。
        # 注：沿用 upstream 的鉴权模型——不自动注册服务端 key，用户自行 /api/auth 连接。
        _spawn_self_facts_backfill(user_id)
        return jsonify({"ok": True, "user_id": user_id, "username": username})

    @app.route("/api/logout", methods=["POST"])
    def logout():
        session.pop("user_id", None)
        session.pop("username", None)
        return jsonify({"ok": True})

    @app.route("/api/status")
    def status():
        uid = session.get("user_id")
        has_key = _client_ready(uid)
        ctrl = _ensure_controller()
        return jsonify(
            {
                "ready": has_key,            # 已连接 API Key 才算 ready（≠ 仅登录）
                "logged_in": bool(uid),
                "username": session.get("username"),
                "model": (_client_creds.get(uid) or {}).get("model") if has_key else None,
                "default_model": DEFAULT_MODEL,
                "controller": {
                    "enabled": ctrl is not None,
                    "has_llm": bool(ctrl and ctrl.has_llm),
                },
                "proactive_pacing": resolve_proactive_pacing(),
            }
        )

    @app.route("/api/auth", methods=["POST"])
    def auth():
        cid = _client_id() or _ANON_CLIENT
        data = request.get_json(force=True, silent=True) or {}
        key = (data.get("api_key") or "").strip()
        model = (data.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        if not key:
            return jsonify({"ok": False, "error": "缺少 api_key"}), 400
        # Shortcut: a literal "0" means "use the server-side default key"
        # (env ANTHROPIC_API_KEY / ~/.lina_key). The real key is resolved
        # entirely server-side and is NEVER sent back to the browser. The
        # default key is trusted, so we skip validation for it.
        if key == "0":
            default_key = resolve_api_key()
            if not default_key:
                return jsonify({"ok": False, "error": "服务器未配置默认密钥。"}), 400
            key = default_key
        else:
            # A user-supplied key must actually authenticate before we accept it.
            ok, err = _validate_key(key, model)
            if not ok:
                return jsonify({"ok": False, "error": err}), 401
        _set_credentials(cid, key, model=model)
        return jsonify({"ok": True, "model": model})

    def _enrich_sessions(sessions: list[dict]) -> list[dict]:
        """Attach resolved prompt labels so the UI can render badges
        without per-row lookups."""
        versions_by_id = {v["version_id"]: v for v in _list_versions()}
        for s in sessions:
            mode = s.get("prompt_mode", "shared")
            if mode == "private":
                s["prompt_label"] = "专属"
                s["prompt_version_name"] = None
                s["prompt_version_missing"] = False
                continue
            vid = s.get("prompt_version_id")
            if vid is None:
                s["prompt_label"] = None
                s["prompt_version_name"] = None
                s["prompt_version_missing"] = False
            elif vid in versions_by_id:
                s["prompt_label"] = versions_by_id[vid]["name"]
                s["prompt_version_name"] = versions_by_id[vid]["name"]
                s["prompt_version_missing"] = False
            else:
                s["prompt_label"] = "(已删除)"
                s["prompt_version_name"] = "(已删除)"
                s["prompt_version_missing"] = True
        return sessions

    @app.route("/api/sessions", methods=["GET"])
    def list_sessions():
        sessions = _store.list_sessions()
        cid = _client_id()
        # With a client id (the normal UI path) scope to that browser's own
        # sessions. Without one (curl/admin), return everything for debugging.
        if cid is not None:
            sessions = [s for s in sessions if s.get("client_id") == cid]
        return jsonify({"sessions": _enrich_sessions(sessions)})

    @app.route("/api/sessions", methods=["POST"])
    def new_session():
        data = request.get_json(force=True, silent=True) or {}
        requested = (data.get("session_id") or "").strip() or None
        version_id = data.get("prompt_version_id")
        if isinstance(version_id, str):
            version_id = version_id.strip() or None
        else:
            version_id = None
        if version_id is not None and _load_version(version_id) is None:
            return jsonify({"ok": False, "error": "指定的版本不存在"}), 400
        conv = _store.new_session(requested)
        conv.prompt_version_id = version_id
        conv.client_id = _client_id()  # tag owner so the sidebar stays private
        _store.save(conv)
        return jsonify(
            {
                "session_id": conv.session_id,
                "messages": [],
                "prompt_version_id": conv.prompt_version_id,
            }
        )

    def _session_payload(conv) -> dict:
        version_name = None
        version_missing = False
        if conv.prompt_mode == "shared" and conv.prompt_version_id:
            v = _load_version(conv.prompt_version_id)
            if v is None:
                version_name = "(已删除)"
                version_missing = True
            else:
                version_name = v.get("name", "")
        if conv.prompt_mode == "private":
            label = "专属"
        elif version_name:
            label = version_name
        else:
            label = None
        return {
            "prompt_mode": conv.prompt_mode,
            "prompt_version_id": conv.prompt_version_id,
            "prompt_version_name": version_name,
            "prompt_version_missing": version_missing,
            "prompt_label": label,
            "prompt_override_count": len(conv.prompt_overrides or {}),
        }

    @app.route("/api/sessions/<session_id>", methods=["GET"])
    def get_session(session_id: str):
        conv = _store.load(session_id)
        return jsonify(
            {
                "session_id": conv.session_id,
                "title": conv.title,
                **_session_payload(conv),
                "forced_state": conv.forced_state,
                "last_meta": conv.last_assistant_meta(),
                "messages": [m.to_dict() for m in conv.messages],
            }
        )

    @app.route("/api/sessions/<session_id>", methods=["PATCH"])
    def patch_session(session_id: str):
        data = request.get_json(force=True, silent=True) or {}
        conv = _store.load(session_id)
        if "prompt_mode" in data:
            new_mode = data["prompt_mode"]
            if new_mode not in ("shared", "private"):
                return jsonify({"ok": False, "error": "prompt_mode 必须是 shared 或 private"}), 400
            if new_mode == "private" and conv.prompt_mode != "private":
                # Seed the session's private overrides from whatever it was
                # using before, so the user has a baseline to tweak.
                if conv.prompt_overrides is None:
                    conv.prompt_overrides = _effective_overrides_for(conv)
            conv.prompt_mode = new_mode
        if "prompt_version_id" in data:
            v = data["prompt_version_id"]
            if v is None or (isinstance(v, str) and v.strip() == ""):
                conv.prompt_version_id = None
            elif not isinstance(v, str):
                return jsonify({"ok": False, "error": "prompt_version_id 必须是字符串或 null"}), 400
            elif _load_version(v) is None:
                return jsonify({"ok": False, "error": "指定的版本不存在"}), 400
            else:
                conv.prompt_version_id = v
        if "forced_state" in data:
            fs = data["forced_state"]
            if fs is None:
                conv.forced_state = None
            elif isinstance(fs, dict):
                cleaned = _sanitize_forced_state(fs)
                conv.forced_state = cleaned or None
            else:
                return jsonify({"ok": False, "error": "forced_state 必须是对象或 null"}), 400
        _store.save(conv)
        # Mode/version flip changes the effective engine; evict relevant cache.
        _invalidate_session_engine(conv.session_id)
        return jsonify(
            {
                "ok": True,
                **_session_payload(conv),
                "forced_state": conv.forced_state,
            }
        )

    # ---------- Per-session prompt override editor ----------
    # Same shape as /api/prompts/* but scoped to a single session's
    # private prompt_overrides dict. Only meaningful in prompt_mode=private,
    # but we don't block reads — the UI surfaces both states clearly.

    @app.route("/api/sessions/<session_id>/overrides", methods=["GET"])
    def session_overrides_list(session_id: str):
        conv = _store.load(session_id)
        overrides = conv.prompt_overrides or {}
        items = []
        for key, label, source, hint in PROMPT_COMPONENTS:
            default = _default_value(key)
            overridden = key in overrides
            items.append(
                {
                    "key": key,
                    "label": label,
                    "source": source,
                    "hint": hint,
                    "overridden": overridden,
                    "current": overrides[key] if overridden else default,
                    "default": default,
                }
            )
        return jsonify(
            {
                "session_id": session_id,
                "prompt_mode": conv.prompt_mode,
                "components": items,
                "override_count": len(overrides),
            }
        )

    @app.route("/api/sessions/<session_id>/overrides/<path:key>", methods=["PUT"])
    def session_override_set(session_id: str, key: str):
        if key not in _PROMPT_KEYS:
            return jsonify({"ok": False, "error": f"未知组件: {key}"}), 400
        data = request.get_json(force=True, silent=True) or {}
        content = data.get("content")
        if not isinstance(content, str):
            return jsonify({"ok": False, "error": "content 必须是字符串"}), 400
        conv = _store.load(session_id)
        overrides = dict(conv.prompt_overrides or {})
        if content == _default_value(key):
            overrides.pop(key, None)
        else:
            overrides[key] = content
        conv.prompt_overrides = overrides or None
        _store.save(conv)
        _invalidate_session_engine(session_id)
        return jsonify(
            {
                "ok": True,
                "overridden": key in overrides,
                "override_count": len(overrides),
            }
        )

    @app.route("/api/sessions/<session_id>/overrides/<path:key>", methods=["DELETE"])
    def session_override_clear_one(session_id: str, key: str):
        if key not in _PROMPT_KEYS:
            return jsonify({"ok": False, "error": f"未知组件: {key}"}), 400
        conv = _store.load(session_id)
        overrides = dict(conv.prompt_overrides or {})
        overrides.pop(key, None)
        conv.prompt_overrides = overrides or None
        _store.save(conv)
        _invalidate_session_engine(session_id)
        return jsonify({"ok": True, "override_count": len(overrides)})

    @app.route("/api/sessions/<session_id>/overrides/reset-all", methods=["POST"])
    def session_override_reset_all(session_id: str):
        conv = _store.load(session_id)
        conv.prompt_overrides = None
        _store.save(conv)
        _invalidate_session_engine(session_id)
        return jsonify({"ok": True})

    @app.route("/api/sessions/<session_id>/overrides/export", methods=["GET"])
    def session_override_export(session_id: str):
        conv = _store.load(session_id)
        payload = json.dumps(conv.prompt_overrides or {}, ensure_ascii=False, indent=2)
        filename = f"prompt_overrides_{conv.session_id}.json"
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id: str):
        ok = _store.delete(session_id)
        return jsonify({"ok": ok})

    @app.route("/api/sessions/<session_id>/export", methods=["GET"])
    def export_session(session_id: str):
        conv = _store.load(session_id)
        payload = json.dumps(conv.to_dict(), ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{conv.session_id}.json"'
                ),
            },
        )

    @app.route("/api/sessions/<session_id>/reset", methods=["POST"])
    def reset_session(session_id: str):
        conv = _store.load(session_id)
        conv.messages.clear()
        conv.title = ""
        _store.save(conv)
        return jsonify({"ok": True})

    @app.route("/api/chat", methods=["POST"])
    def chat():
        cid = _client_id() or _ANON_CLIENT
        if not _client_ready(cid):
            return jsonify({"ok": False, "error": "请先在右上角输入 Anthropic API Key。"}), 401

        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        message = (data.get("message") or "").strip()
        # 可选：用户像微信那样引用了莉娜之前的某条消息来回复。限长防滥用。
        quoted_text = (data.get("quoted_text") or "").strip()[:500]
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if not message:
            return jsonify({"ok": False, "error": "消息为空"}), 400

        conv = _store.load(session_id)
        # Detect a stale pin (shared mode, version deleted) so the UI can warn.
        pinned_version_missing = (
            conv.prompt_mode == "shared"
            and conv.prompt_version_id is not None
            and _load_version(conv.prompt_version_id) is None
        )

        engine = _engine_for_session(conv, cid)
        if engine is None:
            return jsonify({"ok": False, "error": "引擎未就绪"}), 401

        # 跨会话用户记忆：让莉娜记住该用户（同 client_id）在别的会话里讲过的事。
        extra_memory = _cross_session_memory(cid, session_id, message)
        # 莉娜的自我事实清单（按 user_id 跨会话共享）——注入让她对自己说过的话一致。
        self_facts = _self_facts_store.load(cid)

        try:
            result = engine.chat(
                conv, message, extra_memory_chunks=extra_memory,
                quoted_text=quoted_text, self_facts=self_facts,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        _store.save(conv)
        # 自我事实清单：若有轮次刚滑出窗口，在**后台线程**概括更新（不阻塞本次回复）。
        # 概括的是旧对话，晚一两秒、下一轮生效完全无妨。
        if result.slid_out_turns:
            _spawn_self_facts_update(cid, dict(self_facts or {}), list(result.slid_out_turns))

        return jsonify(
            {
                "ok": True,
                "reply": result.text,
                "mood": result.mood,
                "prompt_version_id": conv.prompt_version_id,
                "prompt_version_fallback": "current" if pinned_version_missing else None,
                "forced_state": conv.forced_state,  # None after one-shot consumption
                "retrieved": [
                    {"source": c.source, "heading": c.heading, "text": c.text}
                    for c in result.retrieved
                ],
                "retrieved_history": [
                    {"source": c.source, "heading": c.heading, "text": c.text}
                    for c in result.retrieved_history
                ],
                "plan": result.plan,
                "controller_trace": result.controller_trace,
                "pending_segments": result.pending_segments,
                "has_more_segments": bool(result.pending_segments),
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_creation_input_tokens": result.cache_creation_tokens,
                    "cache_read_input_tokens": result.cache_read_tokens,
                },
            }
        )

    @app.route("/api/continue", methods=["POST"])
    def continue_segment():
        """续说：把上一条回复没说完的下一小段说出来。前端在收到
        has_more_segments=true 后用短计时器调它；用户发新消息会自然作废。"""
        cid = _client_id() or _ANON_CLIENT
        if not _client_ready(cid):
            return jsonify({"ok": False, "error": "引擎未就绪。"}), 401
        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400

        conv = _store.load(session_id)
        if not conv.pending_segments:
            return jsonify({"ok": True, "skipped": True, "reason": "no_pending_segments"})

        engine = _engine_for_session(conv, cid)
        if engine is None:
            return jsonify({"ok": False, "error": "引擎未就绪"}), 401
        try:
            result = engine.continue_segment(conv)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        if result is None:
            return jsonify({"ok": True, "skipped": True, "reason": "no_pending_segments"})
        _store.save(conv)

        return jsonify(
            {
                "ok": True,
                "continuation": True,
                "reply": result.text,
                "mood": result.mood,
                "plan": result.plan,
                "controller_trace": result.controller_trace,
                "pending_segments": result.pending_segments,
                "has_more_segments": bool(result.pending_segments),
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_creation_input_tokens": result.cache_creation_tokens,
                    "cache_read_input_tokens": result.cache_read_tokens,
                },
            }
        )

    @app.route("/api/proactive", methods=["POST"])
    def proactive():
        """用户闲置时由前端计时器触发：莉娜主动发言。
        body.mode: "engage"（默认，抛话头）/ "farewell"（多次未回应后告别）。"""
        cid = _client_id() or _ANON_CLIENT
        if not _client_ready(cid):
            return jsonify({"ok": False, "error": "引擎未就绪。"}), 401
        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        mode = (data.get("mode") or "engage").strip().lower()
        if mode not in ("engage", "farewell"):
            mode = "engage"

        conv = _store.load(session_id)
        # 没有任何用户历史就不主动发言（避免硬造话题）。
        if not any(m.role == "user" for m in conv.messages):
            return jsonify({"ok": True, "skipped": True, "reason": "no_history"})

        # 权威停止：本轮（上一条真实用户消息之后）如果已经告别过，就不再主动。
        # 防止前端计数漂移导致告别后还反复发起。
        for m in reversed(conv.messages):
            if m.role == "user" and not (m.meta and m.meta.get("system_trigger")):
                break
            if m.role == "assistant" and m.meta and m.meta.get("farewell"):
                return jsonify({"ok": True, "skipped": True, "reason": "already_farewelled"})

        engine = _engine_for_session(conv, cid)
        if engine is None:
            return jsonify({"ok": False, "error": "引擎未就绪"}), 401
        try:
            if mode == "farewell":
                result = engine.proactive_farewell(conv)
            else:
                result = engine.proactive(conv)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        _store.save(conv)
        # 莉娜主动讲的（尤其自己的经历）也记进自我事实清单（后台异步）。
        if result.slid_out_turns:
            _spawn_self_facts_update(cid, dict(_self_facts_store.load(cid)), list(result.slid_out_turns))

        # 真正是不是告别，由后端权威计数决定（可能覆盖了请求的 mode）。读最后一条
        # assistant 的 meta 为准，让前端据此停止后续主动。
        did_farewell = bool((conv.last_assistant_meta() or {}).get("farewell"))
        return jsonify(
            {
                "ok": True,
                "proactive": True,
                "mode": "farewell" if did_farewell else "engage",
                "farewell": did_farewell,
                "reply": result.text,
                "mood": result.mood,
                "plan": result.plan,
                "controller_trace": result.controller_trace,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_creation_input_tokens": result.cache_creation_tokens,
                    "cache_read_input_tokens": result.cache_read_tokens,
                },
            }
        )

    # ---------- Voice pipeline (local ASR + streaming TTS) ----------

    def _sse(obj: dict) -> str:
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    @app.route("/api/voice/status", methods=["GET"])
    def voice_status():
        return jsonify(get_voice_engine().status())

    @app.route("/api/voice/init", methods=["POST"])
    def voice_init():
        """Kick off the (lazy, background) load of the ASR/TTS models."""
        return jsonify(get_voice_engine().ensure_loading())

    @app.route("/api/voice/transcribe", methods=["POST"])
    def voice_transcribe():
        """Push-to-talk ASR: a WAV blob in → recognized text out.

        The browser records via Web Audio and uploads a 16 kHz mono WAV, so we
        decode with soundfile (no ffmpeg needed) and hand the waveform to the
        local Qwen3-ASR model."""
        ve = get_voice_engine()
        if ve.status()["state"] != "ready":
            return jsonify({"ok": False, "error": "语音模型尚未就绪。"}), 409
        file = request.files.get("audio")
        if file is None:
            return jsonify({"ok": False, "error": "缺少音频。"}), 400
        try:
            import soundfile as sf

            wav, sr = sf.read(io.BytesIO(file.read()), dtype="float32")
        except Exception as e:
            return jsonify({"ok": False, "error": f"音频解码失败：{e}"}), 400
        try:
            text = ve.transcribe(wav, sr)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "text": text})

    @app.route("/api/voice/say", methods=["POST"])
    def voice_say():
        """Synthesize arbitrary text to speech — used by the per-message replay
        (🔊) button. Re-synthesizes from the stored message text + its mood, so
        it works for any assistant message: live, text-typed, or loaded from
        history. Returns a single WAV. Gated on a connected client to avoid
        anonymous GPU abuse on a public endpoint."""
        cid = _client_id() or _ANON_CLIENT
        if not _client_ready(cid):
            return jsonify({"ok": False, "error": "请先连接 Anthropic API Key。"}), 401
        ve = get_voice_engine()
        if ve.status()["state"] != "ready":
            return jsonify({"ok": False, "error": "语音模型尚未就绪。"}), 409
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "文本为空"}), 400
        text = text[:2000]  # cap to keep one synth call bounded
        mood = data.get("mood") if isinstance(data.get("mood"), dict) else None
        try:
            wav_bytes, _sr = ve.synthesize(text, mood_to_instruct(mood))
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return Response(wav_bytes, mimetype="audio/wav")

    @app.route("/api/voice/chat", methods=["POST"])
    def voice_chat():
        """Streaming voice reply over SSE.

        Streams Claude's tokens as `delta` events for live captions, and — as
        each spoken sentence completes — synthesizes it and emits the audio as
        a base64 `audio` event so playback begins on sentence #1 rather than
        after the whole reply. If the client aborts the request (the barge-in
        interrupt), the generator is closed, the upstream Claude stream is
        aborted, and nothing is persisted."""
        cid = _client_id() or _ANON_CLIENT
        if not _client_ready(cid):
            return jsonify({"ok": False, "error": "请先连接 Anthropic API Key。"}), 401
        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        message = (data.get("message") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        if not message:
            return jsonify({"ok": False, "error": "消息为空"}), 400

        conv = _store.load(session_id)
        engine = _engine_for_session(conv, cid)
        if engine is None:
            return jsonify({"ok": False, "error": "引擎未就绪"}), 401

        ve = get_voice_engine()
        tts_ready = ve.status()["state"] == "ready"
        voice_self_facts = _self_facts_store.load(cid)  # 注入自我事实（语音路径只读不更新）

        def event_stream():
            gen = engine.chat_stream(conv, message, self_facts=voice_self_facts)
            pending = ""        # visible text not yet flushed to a TTS sentence
            cur_mood = None     # latest parsed mood — drives TTS delivery style
            idx = 0             # audio chunk ordering index

            def synth(sentence: str):
                nonlocal idx
                if not tts_ready:
                    return None
                try:
                    wav_bytes, _sr = ve.synthesize(sentence, mood_to_instruct(cur_mood))
                except Exception:
                    return None
                if not wav_bytes:
                    return None
                ev = {
                    "type": "audio",
                    "index": idx,
                    "text": sentence,
                    "mime": "audio/wav",
                    "audio": base64.b64encode(wav_bytes).decode("ascii"),
                }
                idx += 1
                return ev

            try:
                for ev in gen:
                    etype = ev.get("type")
                    if etype == "mood":
                        cur_mood = ev.get("mood")
                        yield _sse({"type": "mood", "mood": cur_mood})
                    elif etype == "delta":
                        yield _sse({"type": "delta", "text": ev["text"]})
                        pending += ev["text"]
                        sentences, pending = iter_sentences(pending)
                        for s in sentences:
                            audio_ev = synth(s)
                            if audio_ev:
                                yield _sse(audio_ev)
                    elif etype == "done":
                        # Speak whatever's left in the buffer as a last chunk.
                        sentences, pending = iter_sentences(pending, flush=True)
                        for s in sentences:
                            audio_ev = synth(s)
                            if audio_ev:
                                yield _sse(audio_ev)
                        # engine.chat_stream has now mutated conv; persist it.
                        _store.save(conv)
                        yield _sse(
                            {
                                "type": "done",
                                "mood": ev.get("mood"),
                                "text": ev.get("text", ""),
                                "forced_state": conv.forced_state,  # None after one-shot consumption
                                "usage": ev.get("usage"),
                                "retrieved": [
                                    {"source": c.source, "heading": c.heading, "text": c.text}
                                    for c in ev.get("retrieved", [])
                                ],
                                "retrieved_history": [
                                    {"source": c.source, "heading": c.heading, "text": c.text}
                                    for c in ev.get("retrieved_history", [])
                                ],
                            }
                        )
            except Exception as e:  # noqa: BLE001 — surface mid-stream errors
                yield _sse({"type": "error", "error": str(e)})
            finally:
                # Closing the engine generator aborts the upstream Claude
                # request if the client bailed (barge-in). On a clean finish
                # this is a harmless no-op.
                gen.close()

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ---------- Prompt override endpoints ----------

    @app.route("/api/prompts", methods=["GET"])
    def list_prompts():
        items = []
        for key, label, source, hint in PROMPT_COMPONENTS:
            default = _default_value(key)
            overridden = key in _overrides
            current = _overrides[key] if overridden else default
            items.append(
                {
                    "key": key,
                    "label": label,
                    "source": source,
                    "hint": hint,
                    "overridden": overridden,
                    "current": current,
                    "default": default,
                }
            )
        return jsonify({"components": items, "override_count": len(_overrides)})

    @app.route("/api/prompts/<path:key>", methods=["PUT"])
    def set_prompt(key: str):
        if key not in _PROMPT_KEYS:
            return jsonify({"ok": False, "error": f"未知组件: {key}"}), 400
        data = request.get_json(force=True, silent=True) or {}
        content = data.get("content")
        if not isinstance(content, str):
            return jsonify({"ok": False, "error": "content 必须是字符串"}), 400
        # If the user submits the exact default, treat as a reset (cleaner UX).
        if content == _default_value(key):
            _overrides.pop(key, None)
        else:
            _overrides[key] = content
        _save_overrides_to_disk(_overrides)
        _invalidate_current_engine()
        return jsonify({"ok": True, "overridden": key in _overrides})

    @app.route("/api/prompts/<path:key>", methods=["DELETE"])
    def reset_prompt(key: str):
        if key not in _PROMPT_KEYS:
            return jsonify({"ok": False, "error": f"未知组件: {key}"}), 400
        _overrides.pop(key, None)
        _save_overrides_to_disk(_overrides)
        _invalidate_current_engine()
        return jsonify({"ok": True})

    @app.route("/api/prompts/reset-all", methods=["POST"])
    def reset_all_prompts():
        _overrides.clear()
        _save_overrides_to_disk(_overrides)
        _invalidate_current_engine()
        return jsonify({"ok": True})

    @app.route("/api/prompts/export", methods=["GET"])
    def export_overrides_endpoint():
        payload = json.dumps(_overrides, ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="prompt_overrides.json"',
            },
        )

    # ---------- Prompt version snapshots ----------

    @app.route("/api/prompts/versions", methods=["GET"])
    def versions_list():
        return jsonify({"versions": _list_versions()})

    @app.route("/api/prompts/versions", methods=["POST"])
    def versions_save():
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name") or ""
        note = data.get("note") or ""
        record = _save_version(name=name, note=note)
        return jsonify(
            {
                "ok": True,
                "version": {
                    "version_id": record["version_id"],
                    "name": record["name"],
                    "note": record["note"],
                    "created_at": record["created_at"],
                    "override_count": len(record["overrides"]),
                },
            }
        )

    @app.route("/api/prompts/versions/diff", methods=["GET"])
    def versions_diff():
        a_id = request.args.get("a", "").strip()
        b_id = request.args.get("b", "").strip()
        va = _load_version(a_id)
        vb = _load_version(b_id)
        if va is None or vb is None:
            return jsonify({"ok": False, "error": "找不到指定的版本"}), 404
        return jsonify(_diff_versions(va, vb))

    @app.route("/api/prompts/versions/<version_id>", methods=["GET"])
    def versions_get(version_id: str):
        data = _load_version(version_id)
        if data is None:
            return jsonify({"ok": False, "error": "版本不存在"}), 404
        return jsonify(data)

    @app.route("/api/prompts/versions/<version_id>", methods=["DELETE"])
    def versions_delete(version_id: str):
        ok = _delete_version(version_id)
        if not ok:
            return jsonify({"ok": False, "error": "版本不存在"}), 404
        _invalidate_engine(version_id)
        return jsonify({"ok": True})

    @app.route("/api/prompts/versions/<version_id>", methods=["PATCH"])
    def versions_patch(version_id: str):
        body = request.get_json(force=True, silent=True) or {}
        name = body.get("name") if isinstance(body.get("name"), str) else None
        note = body.get("note") if isinstance(body.get("note"), str) else None
        data = _rename_version(version_id, name=name, note=note)
        if data is None:
            return jsonify({"ok": False, "error": "版本不存在"}), 404
        return jsonify({"ok": True, "name": data["name"], "note": data["note"]})

    @app.route("/api/prompts/versions/<version_id>/restore", methods=["POST"])
    def versions_restore(version_id: str):
        data = _load_version(version_id)
        if data is None:
            return jsonify({"ok": False, "error": "版本不存在"}), 404
        _overrides.clear()
        _overrides.update(data.get("overrides", {}))
        _save_overrides_to_disk(_overrides)
        _invalidate_current_engine()
        return jsonify({"ok": True, "override_count": len(_overrides)})

    @app.route("/api/prompts/versions/<version_id>/export", methods=["GET"])
    def versions_export(version_id: str):
        data = _load_version(version_id)
        if data is None:
            return jsonify({"ok": False, "error": "版本不存在"}), 404
        safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", data.get("name", "")).strip("_") or version_id
        filename = f"prompt_version_{safe_name}.json"
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.route("/api/prompts/import", methods=["POST"])
    def import_overrides_endpoint():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify({"ok": False, "error": "需要 JSON 对象 {key: content, ...}"}), 400
        filtered = {
            k: v for k, v in data.items()
            if k in _PROMPT_KEYS and isinstance(v, str)
        }
        ignored = [k for k in data.keys() if k not in _PROMPT_KEYS]
        _overrides.clear()
        _overrides.update(filtered)
        _save_overrides_to_disk(_overrides)
        _invalidate_current_engine()
        return jsonify({"ok": True, "imported": list(filtered.keys()), "ignored": ignored})

    return app
