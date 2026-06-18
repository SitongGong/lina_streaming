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
    FEEDBACK_DIR,
    MESSAGE_FEEDBACK_DIR,
    PROJECT_ROOT,
    STATIC_DIR,
    USERS_FILE,
    resolve_admin_users,
    resolve_api_access_keys,
    resolve_api_key,
    resolve_controller_settings,
    resolve_openai_api_key,
    resolve_proactive_pacing,
    resolve_secret_key,
)
from .controller import (
    LinaController,
    build_default_controller,
    read_prompt_file,
    read_json_value,
    make_json_key,
    reload_rule_patterns,
    set_prompt_overrides,
)
from .controller.merged_controller import MergedController
from .conversation import Conversation, ConversationStore
from .feedback import DIMENSIONS, FeedbackStore, MessageFeedbackStore
from .rag import retrieve_user_memory_chunks
from .self_facts import SelfFactsStore
from .user_facts import UserFactsStore
from .voice import get_voice_engine, iter_sentences, mood_to_instruct
from . import asta_tts


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
_feedback_store = FeedbackStore(FEEDBACK_DIR)
_msg_feedback_store = MessageFeedbackStore(MESSAGE_FEEDBACK_DIR)
# 管理员用户名集合（来自环境变量 LINA_ADMIN_USERS）。只有这些账号能看汇总统计。
_admin_users = resolve_admin_users()

# ---- 强制账号登录 ----
# 必须登录才能聊。登录态存 Flask 签名 cookie（session["user_id"]）。
# 登录后「身份」就是 user_id —— 会话按 user_id 打标隔离、检索按 user_id 隔离、
# 引擎凭证按 user_id 注册（统一用服务端 key，用户不再自己输 key）。
_user_store = UserStore(USERS_FILE)
# 莉娜对每个用户的「自我事实清单」（跨会话共享，按 user_id 存）。
_self_facts_store = SelfFactsStore(USERS_FILE.parent / "self_facts")
# 「用户事实清单」——记用户讲过的、关于他自己的稳定事实（跨会话共享，按 user_id 存）。
# 解决「聊久了把用户的事忘了/记混/编造」。
_user_facts_store = UserFactsStore(USERS_FILE.parent / "user_facts")


def _spawn_user_facts_update(user_id: str, current_facts: dict, slid_turns: list) -> None:
    """后台线程：把刚滑出窗口的几轮里**用户**讲过的稳定事实概括进用户事实清单。"""
    if not user_id or not slid_turns:
        return
    ctrl = _ensure_controller()
    if ctrl is None or not ctrl.has_llm:
        return

    def _work():
        try:
            base = _user_facts_store.load(user_id) or current_facts
            updated = ctrl.update_user_facts_sync(base, slid_turns)
            if updated is not None:
                _user_facts_store.save(user_id, updated)
        except Exception:
            pass

    threading.Thread(target=_work, daemon=True).start()


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
_controller: LinaController | None = None            # 原版（20 路并发 advisor）
_controller_merged: LinaController | None = None     # 合并版（4 组 advisor，提速实验）
# 「合并 advisor」开关——**按用户独立**：集合里有该 user_id = 该用户用合并版(4 advisor)，
# 否则用原版(20 advisor)。A 用合并版、B 同时用原版，互不影响。
_merged_users: set[str] = set()
# controller 专用锁——**不能用 _engine_lock**：_ensure_controller 会在
# _engine_for_session 已持有 _engine_lock 时被调用，复用同一把（非重入）锁会自锁死。
_controller_lock = threading.Lock()


def _user_uses_merged(cid: str | None) -> bool:
    return bool(cid) and cid in _merged_users


def _build_original_controller() -> LinaController:
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                cfg = resolve_controller_settings()
                _controller = build_default_controller(
                    api_key=cfg["api_key"], model=cfg["model"],
                    base_url=cfg["base_url"], provider=cfg["provider"],
                )
    return _controller


def _build_merged_controller() -> LinaController:
    global _controller_merged
    if _controller_merged is None:
        with _controller_lock:
            if _controller_merged is None:
                cfg = resolve_controller_settings()
                _controller_merged = MergedController(
                    openai_client=_build_original_controller()._client,
                    model_name=cfg["model"],
                )
    return _controller_merged


def _controller_for(cid: str | None) -> LinaController | None:
    """按用户返回 controller：该用户开了合并版→合并版，否则原版。
    两个版本都是无状态判断器，各建一个全局实例、所有用户共享，不重复构造。"""
    return _build_merged_controller() if _user_uses_merged(cid) else _build_original_controller()


def _ensure_controller() -> LinaController | None:
    """无用户上下文时（后台提炼、启动预建等）用的默认 controller = 原版。"""
    return _build_original_controller()

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
    ("BEHAVIOR_RULES", "行为规则", "code", "主模型：系统提示里的核心约束。"),
    ("MOOD_FORMAT_SPEC", "情绪标记格式", "code", "主模型：决定 [mood: …] 的输出格式。"),
    ("SYSTEM_PROMPT_TEMPLATE", "系统提示模板", "code",
     "主模型：包含占位符 {core_text} / {behavior_rules} / {mood_format_spec}，以及 controller 填充位说明。"),
    ("SEGMENT_PROTOCOL_SPEC", "分段说话机制", "code", "主模型：决定何时把回复拆成多小段。"),
    ("PROACTIVE_INSTRUCTION", "主动发言·开口指令", "code", "用户沉默时让莉娜主动开口的指令。"),
    ("FAREWELL_INSTRUCTION", "主动发言·告别指令", "code", "连续没回应后让莉娜自然收束的指令。"),
    ("CONTINUE_INSTRUCTION", "续说·展开指令", "code", "把没说完的要点展开成一小段的指令（含 {point} 占位）。"),

    # —— controller 侧（按场景注入的 prompt，key = prompts/ 下相对路径）——
    # few-shot 示例库（controller 按场景挑）
    ("controller/fewshot/typo_tolerance.txt", "few-shot·错字容错", "controller", "用户打错字时怎么领会的范例。"),
    ("controller/fewshot/no_trailing_question.txt", "few-shot·别连环提问", "controller", "抑制句尾连甩问号的范例。"),
    ("controller/fewshot/positive_response.txt", "few-shot·报喜回应", "controller", "用户报喜时替他高兴、不浇冷水的范例。"),
    ("controller/fewshot/comfort.txt", "few-shot·安抚", "controller", "用户低落时先接情绪的范例。"),
    ("controller/fewshot/modern_boundary.txt", "few-shot·现代边界", "controller", "现代请求时茫然以对的范例。"),
    # 差异化场景模块
    ("modules/user_vent.txt", "模块·安抚陪伴", "controller", "安抚场景注入。"),
    ("modules/action_boundary.txt", "模块·知识边界", "controller", "现代请求/AI自指场景注入。"),
    ("modules/world_immersion.txt", "模块·兴奋点沉浸", "controller", "古代语/遗物/戏剧/香草场景注入。"),
    ("modules/relationship_recall.txt", "模块·关系回访", "controller", "“还记得吗/上次”场景注入。"),
    ("modules/self_introspection.txt", "模块·自我反思", "controller", "问她身份/性格/来历场景注入。"),
    ("modules/welcome_back.txt", "模块·久别重逢", "controller", "用户隔了较久回来时注入。"),
    ("modules/continuation.txt", "模块·续说", "controller", "把没说完的段接着说时注入。"),
    ("modules/hook_concrete_example.txt", "模块·具体细节钩子", "controller", "要求带专名/场景细节。"),
    ("modules/hook_callback.txt", "模块·回勾话头", "controller", "回勾最近未聊完的话头。"),
    ("modules/hook_history_recall.txt", "模块·引用历史", "controller", "引用用户讲过的事。"),
    # controller 专用模板
    ("controller/proactive_topic.txt", "主动发言·挑话头", "controller", "用户沉默时挑哪个话头主动开口。"),
    ("controller/self_facts.txt", "自我清单·提炼规则", "controller", "把莉娜说过的自我事实概括进清单的规则。"),
    ("controller/control_flag.txt", "微顾问·布尔判官模板", "controller", "每个布尔字段微顾问的 prompt 模板。"),
    ("controller/control_int.txt", "微顾问·数值判官模板", "controller", "数值字段微顾问的 prompt 模板。"),
    ("controller/control_text.txt", "微顾问·文本判官模板", "controller", "文本字段微顾问的 prompt 模板。"),
]
# 主模型 4 个 prompt 现在也是文件（prompts/main/），override 仍按这些 KEY 走
# character 的 overrides 机制；它们与文件的映射：
_MAIN_PROMPT_FILES = {
    "BEHAVIOR_RULES": "main/behavior_rules.txt",
    "MOOD_FORMAT_SPEC": "main/mood_format_spec.txt",
    "SYSTEM_PROMPT_TEMPLATE": "main/system_prompt_template.txt",
    "SEGMENT_PROTOCOL_SPEC": "main/segment_protocol_spec.txt",
    "PROACTIVE_INSTRUCTION": "main/proactive_instruction.txt",
    "FAREWELL_INSTRUCTION": "main/farewell_instruction.txt",
    "CONTINUE_INSTRUCTION": "main/continue_instruction.txt",
}


def _generate_json_components() -> list[tuple[str, str, str, str]]:
    """把 4 个结构化 JSON 拆成细粒度可编辑组件（key = 'file#dotpath'）。
    读 JSON 自动生成，加新规则只需改 JSON、无需改这里。"""
    import json as _json
    specs = [
        ("controller/advisor_rules.json", "微顾问规则", "advisor", ["target_desc", "decision_rules", "range_desc"]),
        ("controller/rules.json", "规则层关键词", "rule", None),          # 每个场景一条（值是数组）
        ("controller/proactive_stages.json", "主动发言策略", "stage", None),  # 每个 stage 一条
        ("controller/constraints.json", "本轮约束句", "constraint", None),    # 每条约束一条
    ]
    comps: list[tuple[str, str, str, str]] = []
    base = PROJECT_ROOT / "prompts"
    for rel, label_prefix, source, subfields in specs:
        try:
            data = _json.loads((base / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for top_key, val in data.items():
            if top_key.startswith("_"):
                continue
            if subfields and isinstance(val, dict):
                # advisor：每个子字段一条
                for sf in subfields:
                    if sf in val:
                        comps.append((f"{rel}#{top_key}.{sf}",
                                      f"{label_prefix}·{top_key}·{sf}", source, f"{rel} 里 {top_key}.{sf}"))
            else:
                comps.append((f"{rel}#{top_key}", f"{label_prefix}·{top_key}", source, f"{rel} 里 {top_key}"))
    return comps


PROMPT_COMPONENTS += _generate_json_components()
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


def _sync_controller_overrides() -> None:
    """把 _overrides 里属于 controller（key 含 '/' 的相对路径）的项，
    推到 controller 的全局 override 层，让 load_prompt 即时生效。
    主模型 4 个 KEY 不在此处——它们经 character 的 overrides 机制走。"""
    ctrl_ov = {k: v for k, v in _overrides.items() if "/" in k}
    set_prompt_overrides(ctrl_ov)
    # 规则层正则在 import 时已编译进模块全局，需显式重载才能让 override 生效。
    reload_rule_patterns()


def _match_default_version() -> dict | None:
    """If the active global default (`_overrides`) exactly equals some saved
    version's overrides, return that version's {version_id, name}; else None
    ("自定义"). Self-correcting: editing the default in the 提示词 tab makes it
    stop matching, so the UI never shows a stale version name."""
    cur = dict(_overrides)
    for v in _list_versions():  # newest first
        data = _load_version(v["version_id"])
        if data is not None and dict(data.get("overrides", {})) == cur:
            return {"version_id": v["version_id"], "name": v.get("name", "")}
    return None


def _effective_prompt_version_id(conv) -> str | None:
    """这个会话实际生效的 prompt 版本 id，用于给问卷/逐条反馈打标。

    - 共享模式 + 已 pin 某版本 → 该版本 id。
    - 共享模式 + 未 pin（跟随全局默认）→ 当前全局默认对应的已保存版本 id；
      若默认是「自定义」（不匹配任何已保存版本）则为 None。
      ——这一支正是之前漏标的常见情况：直接存 conv.prompt_version_id 会得到 None。
    - 私有模式 → None（用的是会话内联自定义 prompt，由 prompt_mode="private" 标识）。
    """
    if conv.prompt_mode == "shared":
        if conv.prompt_version_id:
            return conv.prompt_version_id
        matched = _match_default_version()
        return matched["version_id"] if matched else None
    return None


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
    # 静态人设 .md（在 static/）
    if key.endswith(".md"):
        fpath = STATIC_DIR / key
        return fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    # 主模型 4 个 prompt：现在是 prompts/main/ 下的文件
    if key in _MAIN_PROMPT_FILES:
        return read_prompt_file(_MAIN_PROMPT_FILES[key])
    # JSON 细粒度字段：key = "rel_path#dotpath"
    if "#" in key:
        rel, dotpath = key.split("#", 1)
        # rules.json 的场景值是数组 → 以 JSON 数组字符串呈现给编辑器
        import json as _json
        try:
            data = _json.loads((PROJECT_ROOT / "prompts" / rel).read_text(encoding="utf-8"))
            cur = data
            for part in dotpath.split("."):
                cur = cur[part]
            return cur if isinstance(cur, str) else _json.dumps(cur, ensure_ascii=False, indent=2)
        except (OSError, ValueError, KeyError, TypeError):
            return ""
    # controller 侧整文件：key 本身就是 prompts/ 下的相对路径
    if "/" in key:
        return read_prompt_file(key)
    return ""


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


def _is_admin() -> bool:
    """当前登录用户是否管理员（用户名在 LINA_ADMIN_USERS 配置里）。

    管理员判定用 username（而非 user_id），与环境变量里写的人类可读用户名一致。"""
    return (session.get("username") or "") in _admin_users


def _require_admin():
    """返回一个 403 响应（用于 admin-only 端点的开头守卫），已是管理员则返回 None。"""
    if not _is_admin():
        return jsonify({"ok": False, "error": "仅管理员可查看统计数据。"}), 403
    return None


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
    # key 带上 controller 标记：同一用户切换 merged 会用不同 engine，
    # 不同用户用不同 controller 也不会共用同一个 engine（避免串扰）。
    ctrl_tag = "merged" if _user_uses_merged(cid) else "orig"
    key = cid + "\x00" + _engine_base_key(conv) + "\x00" + ctrl_tag
    with _engine_lock:
        engine = _engine_pool.get(key)
        if engine is not None:
            return engine
        engine = CharacterEngine(
            api_key=creds["key"],
            static_dir=STATIC_DIR,
            model=creds["model"],
            overrides=_effective_overrides_for(conv),
            controller=_controller_for(cid),   # 按该用户选原版/合并版
        )
        _engine_pool[key] = engine
        return engine


# --- OpenAI 兼容公开接口用的共享引擎 ---------------------------------------
# 外部调用方（用 OpenAI SDK）不走网页的 per-client key 机制，这里用服务端默认
# Anthropic key 构造一个共享引擎，懒加载、全局复用。
_openai_api_engine: CharacterEngine | None = None
_openai_api_engine_lock = threading.Lock()


def _openai_compat_engine() -> CharacterEngine | None:
    """懒加载一个共享引擎，给 /v1/chat/completions 用。无服务端 key → None。"""
    global _openai_api_engine
    if _openai_api_engine is not None:
        return _openai_api_engine
    with _openai_api_engine_lock:
        if _openai_api_engine is None:
            key = resolve_api_key()  # 服务端默认 Anthropic key
            if not key:
                return None
            _openai_api_engine = CharacterEngine(
                api_key=key,
                static_dir=STATIC_DIR,
                model=DEFAULT_MODEL,
                controller=_ensure_controller(),
            )
        return _openai_api_engine


def _conv_from_openai_messages(messages: list) -> tuple[Conversation, str]:
    """把 OpenAI 的 messages 数组转成 (临时会话, 最后一条用户消息)。

    system 消息忽略（Lina 的人设由自己的 prompt 决定，不接受外部 system 覆盖）。
    除最后一条 user 外的 user/assistant 都作为历史塞进临时会话；最后一条 user
    作为本轮输入返回。临时会话不入库（每次调用无状态，符合 OpenAI 接口习惯）。
    """
    conv = Conversation(session_id="openai-api-ephemeral")
    norm = [
        (str(m.get("role", "")), str(m.get("content", "")))
        for m in (messages or [])
        if isinstance(m, dict) and str(m.get("content", "")).strip()
    ]
    # 找最后一条 user 作为本轮输入；它之前的都是历史。
    last_user_idx = max(
        (i for i, (r, _) in enumerate(norm) if r == "user"), default=-1
    )
    if last_user_idx < 0:
        return conv, ""
    for i, (role, content) in enumerate(norm):
        if i == last_user_idx:
            continue
        if role in ("user", "assistant"):
            conv.add(role, content)
    return conv, norm[last_user_idx][1]


def _pop_engines_with_suffix(suffix: str) -> None:
    """Drop pooled engines whose key ends with `suffix`, across ALL clients —
    used when a shared prompt source changes (affects everyone using it)."""
    with _engine_lock:
        for k in [k for k in _engine_pool if k.endswith(suffix)]:
            _engine_pool.pop(k, None)


def _invalidate_current_engine() -> None:
    """Drop every client's engine using the editable global overrides."""
    _pop_engines_with_suffix("\x00" + _CURRENT_KEY)
    # 全局 override 变了 → 同步 controller 的 load_prompt override 层（即时生效）。
    _sync_controller_overrides()


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
    _sync_controller_overrides()  # 启动即把已保存的 controller override 注册进 load_prompt

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
        return jsonify({"logged_in": bool(uid), "user_id": uid, "username": session.get("username"), "is_admin": _is_admin()})

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
        return jsonify({"ok": True, "user_id": result, "username": username, "is_admin": _is_admin()})

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
        # 沿用 upstream 鉴权模型——不自动注册服务端 key，用户自行 /api/auth 连接。
        _spawn_self_facts_backfill(user_id)
        return jsonify({"ok": True, "user_id": user_id, "username": username, "is_admin": _is_admin()})

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
                "is_admin": _is_admin(),
                "model": (_client_creds.get(uid) or {}).get("model") if has_key else None,
                "default_model": DEFAULT_MODEL,
                "controller": {
                    "enabled": ctrl is not None,
                    "has_llm": bool(ctrl and ctrl.has_llm),
                    "merged": _user_uses_merged(uid),   # 该用户是否用合并版（4 advisor）
                },
                "proactive_pacing": resolve_proactive_pacing(),
                "default_prompt": _match_default_version(),
            }
        )

    @app.route("/api/controller_mode", methods=["POST"])
    def controller_mode():
        """切换当前用户的 controller：合并版(4 advisor) ↔ 原版(20 advisor)。
        **仅对该用户生效**，不影响别人。切换只清该用户的 engine（让其按新 controller 重建）。"""
        cid = session.get("user_id")
        if not cid:
            return jsonify({"ok": False, "error": "请先登录"}), 401
        data = request.get_json(force=True, silent=True) or {}
        merged = bool(data.get("merged", False))
        if merged:
            _merged_users.add(cid)
        else:
            _merged_users.discard(cid)
        # 只清该用户的 engine（key 以 "<cid>\x00" 开头），不动别人。
        with _engine_lock:
            for k in [k for k in _engine_pool if k.startswith(cid + "\x00")]:
                _engine_pool.pop(k, None)
        mode = "合并版(4 advisor)" if merged else "原版(20 advisor)"
        return jsonify({"ok": True, "merged": merged, "mode": mode})

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
        # 用户事实清单——注入让她记得用户是谁、聊过什么（跨上百轮不忘/不混/不编）。
        user_facts = _user_facts_store.load(cid)

        try:
            result = engine.chat(
                conv, message, extra_memory_chunks=extra_memory,
                quoted_text=quoted_text, self_facts=self_facts, user_facts=user_facts,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        _store.save(conv)
        # 自我/用户事实清单：若有轮次刚滑出窗口，在**后台线程**概括更新（不阻塞本次回复）。
        if result.slid_out_turns:
            _spawn_self_facts_update(cid, dict(self_facts or {}), list(result.slid_out_turns))
            _spawn_user_facts_update(cid, dict(user_facts or {}), list(result.slid_out_turns))

        return jsonify(
            {
                "ok": True,
                "reply": result.text,
                "ts": conv.messages[-1].ts if conv.messages else None,
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
                "ts": conv.messages[-1].ts if conv.messages else None,
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
        proactive_user_facts = _user_facts_store.load(cid)   # 参考用户长期记忆
        proactive_self_facts = _self_facts_store.load(cid)   # 参考莉娜自己的设定，避免自相矛盾
        try:
            if mode == "farewell":
                result = engine.proactive_farewell(conv, user_facts=proactive_user_facts, self_facts=proactive_self_facts)
            else:
                result = engine.proactive(conv, user_facts=proactive_user_facts, self_facts=proactive_self_facts)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        _store.save(conv)
        # 注意：主动发言的内容**不再喂进记忆清单**（_run_proactive 已返回空 slid_out），
        # 避免即兴话头污染记忆 / 与之前说过的冲突。下面这个分支因此基本不触发。
        if result.slid_out_turns:
            _spawn_self_facts_update(cid, dict(_self_facts_store.load(cid)), list(result.slid_out_turns))
            _spawn_user_facts_update(cid, dict(_user_facts_store.load(cid)), list(result.slid_out_turns))

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
                "ts": conv.messages[-1].ts if conv.messages else None,
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
                                "ts": conv.messages[-1].ts if conv.messages else None,
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

    @app.route("/api/tts/speak", methods=["POST"])
    def tts_speak():
        """Voice an already-generated TEXT reply sentence-by-sentence via the
        remote *asta* TTS engine with a fixed emotion (default joy). Streams
        base64 `audio` events over SSE — one per sentence — so playback starts
        on sentence #1. Unlike /api/voice/say (local-GPU qwen-tts, whole reply),
        this needs no GPU on the host and is per-sentence + fixed joy. Gated on a
        connected client, same as /api/voice/say, to avoid anonymous abuse."""
        cid = _client_id() or _ANON_CLIENT
        if not _client_ready(cid):
            return jsonify({"ok": False, "error": "请先连接 Anthropic API Key。"}), 401
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "文本为空"}), 400
        text = text[:4000]  # bound total synthesis work per request

        def event_stream():
            try:
                sentences, _ = iter_sentences(text, flush=True)
                idx = 0
                for s in sentences:
                    try:
                        wav = asta_tts.synthesize(s)
                    except Exception as e:  # noqa: BLE001 — report, keep going
                        yield _sse({"type": "error", "index": idx, "error": str(e)})
                        continue
                    if not wav:
                        continue
                    yield _sse(
                        {
                            "type": "audio",
                            "index": idx,
                            "text": s,
                            "mime": "audio/wav",
                            "audio": base64.b64encode(wav).decode("ascii"),
                        }
                    )
                    idx += 1
                yield _sse({"type": "done"})
            except Exception as e:  # noqa: BLE001
                yield _sse({"type": "error", "error": str(e)})

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ---------- Post-evaluation questionnaire ----------
    # A tester rates Lina on nine dimensions (good/bad + reason) plus a
    # free-text note, one questionnaire per session. Stored as JSON (one file
    # per session) under feedback/. No Anthropic call, so these are NOT gated
    # on _client_ready — the chat being evaluated already happened.

    @app.route("/api/feedback/schema", methods=["GET"])
    def feedback_schema():
        return jsonify(
            {"dimensions": [{"key": k, "label": label} for k, label in DIMENSIONS]}
        )

    @app.route("/api/feedback/summary", methods=["GET"])
    def feedback_summary():
        # 统计数据仅管理员可见。聚合全部测评者的问卷。
        denied = _require_admin()
        if denied:
            return denied
        return jsonify(_feedback_store.summary(None))

    @app.route("/api/feedback/export", methods=["GET"])
    def feedback_export():
        denied = _require_admin()
        if denied:
            return denied
        records = [r.to_dict() for r in _feedback_store.list_records(None)]
        payload = json.dumps(records, ensure_ascii=False, indent=2)
        return Response(
            payload,
            mimetype="application/json; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="feedback.json"'},
        )

    @app.route("/api/feedback", methods=["POST"])
    def feedback_submit():
        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        # Copy the conversation's title in for human-readable indexing — but
        # only if that session actually exists; never auto-create one here
        # (ConversationStore.load() would otherwise spawn an empty session).
        # 解析定位信息：标题 / prompt 版本 / 聊到第几轮（提交时的消息数）。
        session_title = ""
        prompt_version_id = None
        prompt_mode = ""
        message_count = 0
        last_message_ts = None
        if _store._path(session_id).exists():
            conv = _store.load(session_id)
            session_title = conv.title
            # 跟随全局默认的共享会话，要解析出当时实际生效的默认版本，别只存 None。
            prompt_version_id = _effective_prompt_version_id(conv)
            prompt_mode = conv.prompt_mode
            message_count = len(conv.messages)
            if conv.messages:
                last_message_ts = conv.messages[-1].ts
        try:
            record = _feedback_store.submit(
                session_id=session_id,
                dimensions=data.get("dimensions"),
                other=data.get("other"),
                client_id=_client_id(),
                session_title=session_title,
                prompt_version_id=prompt_version_id,
                prompt_mode=prompt_mode,
                message_count=message_count,
                last_message_ts=last_message_ts,
            )
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "feedback": record.to_dict()})

    @app.route("/api/feedback/<session_id>", methods=["GET"])
    def feedback_get(session_id: str):
        record = _feedback_store.load(session_id)
        return jsonify({"feedback": record.to_dict() if record else None})

    # ---------- Per-message thumbs feedback ----------
    # 每条助手回复上的 👍/👎 + 可选理由。任意登录用户都可对自己会话里的消息打分；
    # 但聚合统计（/summary）仅管理员可见。一会话一 JSON，按消息 ts 作 key。

    @app.route("/api/message-feedback/summary", methods=["GET"])
    def message_feedback_summary():
        denied = _require_admin()
        if denied:
            return denied
        return jsonify(_msg_feedback_store.summary())

    @app.route("/api/message-feedback", methods=["POST"])
    def message_feedback_set():
        data = request.get_json(force=True, silent=True) or {}
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "缺少 session_id"}), 400
        session_title = ""
        prompt_version_id = None
        if _store._path(session_id).exists():
            conv = _store.load(session_id)
            session_title = conv.title
            prompt_version_id = _effective_prompt_version_id(conv)
        try:
            entry = _msg_feedback_store.set(
                session_id=session_id,
                message_ts=data.get("message_ts"),
                rating=data.get("rating"),          # "up" | "down" | "" (clear)
                reason=data.get("reason") or "",
                user_id=_client_id(),
                text=data.get("text") or "",
                session_title=session_title,
                dimension=data.get("dimension") or "",
                prompt_version_id=prompt_version_id,
            )
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "entry": entry})

    @app.route("/api/message-feedback/<session_id>", methods=["GET"])
    def message_feedback_for_session(session_id: str):
        # 回显某会话里已打分的消息（按 ts），供前端加载历史时还原按钮状态。
        return jsonify({"items": _msg_feedback_store.list_for_session(session_id)})

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

    # ---------- OpenAI 兼容接口：/v1/chat/completions ----------
    # 让外部用 OpenAI SDK / 任意兼容客户端直接调 Lina。支持 stream=true。
    # 只返回纯文本回复（mood / [segments] 已由引擎剥除）。暂不鉴权。
    @app.route("/v1/chat/completions", methods=["POST"])
    def openai_chat_completions():
        # 鉴权：配了 LINA_API_KEYS 才校验；没配则放行（内网/调试）。
        allowed_keys = resolve_api_access_keys()
        if allowed_keys:
            auth = request.headers.get("Authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if token not in allowed_keys:
                return jsonify({
                    "error": {"message": "无效或缺失的 API key", "type": "invalid_request_error",
                              "code": "invalid_api_key"}
                }), 401

        engine = _openai_compat_engine()
        if engine is None:
            return jsonify({
                "error": {"message": "服务端未配置默认 API key", "type": "server_error"}
            }), 503

        data = request.get_json(force=True, silent=True) or {}
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return jsonify({
                "error": {"message": "messages 不能为空", "type": "invalid_request_error"}
            }), 400
        stream = bool(data.get("stream", False))
        model_name = str(data.get("model") or "lina")

        conv, user_message = _conv_from_openai_messages(messages)
        if not user_message:
            return jsonify({
                "error": {"message": "messages 里没有 user 消息", "type": "invalid_request_error"}
            }), 400

        # 固定/伪造的 id 和时间戳（OpenAI 响应需要这些字段）。
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if stream:
            def event_stream():
                # 首个 chunk 带 role，符合 OpenAI 流式约定。
                first = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                try:
                    for ev in engine.chat_stream(conv, user_message):
                        if ev.get("type") != "delta":
                            continue  # mood/done 不外露，只流文本
                        text = ev.get("text") or ""
                        if not text:
                            continue
                        chunk = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                except Exception as e:  # noqa: BLE001
                    err = {"error": {"message": str(e), "type": "server_error"}}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                # 收尾 chunk：finish_reason=stop + [DONE]，OpenAI 客户端据此结束。
                done = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(event_stream(), mimetype="text/event-stream")

        # 非流式：跑完整 chat()，一次性返回。
        try:
            result = engine.chat(conv, user_message)
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500
        return jsonify({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
        })

    return app
