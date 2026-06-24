"""Character engine: builds the system prompt and calls the Claude API.

Strategy:
- The "core" system prompt (character setup + world + behavior rules) is
  marked with `cache_control` so it's reused across turns at low cost.
- Each turn, the RAG layer retrieves a few chunks from the supporting files
  and prepends them as a brief "参考资料" section in the user message. This
  keeps situation-specific context in the model's view without bloating the
  cached prefix.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from .controller import (
    LinaController,
    LinaPromptComposer,
    LinaPromptPlan,
    LinaTurnContext,
)
from .conversation import Conversation
from .rag import CharacterRAG, Chunk, retrieve_history_chunks


# Matches a leading mood tag like:
#   [mood: 好奇 | 7 | 信任=4]
# Tolerant of full-width punctuation and either 信任/trust label.
MOOD_TAG_RE = re.compile(
    r"^\s*\[\s*mood\s*[:：]\s*"
    r"(?P<mood>[^\|｜\]]+?)\s*[\|｜]\s*"
    r"(?P<intensity>\d+)\s*[\|｜]\s*"
    r"(?:信任|trust)\s*[=＝]\s*"
    r"(?P<trust>\d+)\s*\]\s*\n?",
    re.IGNORECASE,
)


# Matches an optional trailing segment-plan tag like:
#   [segments: 里面有半张烧焦的羊皮纸 || 上面的字我只认出三个]
# The body before this tag is the FIRST segment (shown now); the items here
# are the REMAINING segments' short outline points, parked for later
# proactive continuation. Stripped from what the user sees.
SEGMENTS_TAG_RE = re.compile(
    r"\n?\s*\[\s*segments?\s*[:：]\s*(?P<body>.+?)\s*\]\s*$",
    re.IGNORECASE | re.DOTALL,
)

# How many remaining segments we allow. Caps "split long replies" so it
# doesn't overshoot into "lina monologues for 6 turns".
MAX_PENDING_SEGMENTS = 3


def parse_segments_tag(raw: str) -> tuple[str, list[str]]:
    """Strip a trailing [segments: A || B] tag from `raw`.

    Returns (text_without_tag, [outline_points]). Absent tag → (raw, []).
    Splits on `||` (also full-width ｜｜). Empty points dropped, capped at
    MAX_PENDING_SEGMENTS.
    """
    if not raw:
        return raw, []
    m = SEGMENTS_TAG_RE.search(raw)
    if not m:
        return raw, []
    body = m.group("body")
    cleaned = raw[: m.start()].rstrip()
    parts = [p.strip() for p in re.split(r"\s*\|\|\s*|\s*｜｜\s*", body)]
    points = [p for p in parts if p][:MAX_PENDING_SEGMENTS]
    return cleaned, points


DEFAULT_MODEL = "claude-sonnet-4-6"

# 本地「滑出窗口旧对话」BM25 检索（retrieve_history_chunks）开关：默认【关】。
# 用户对话历史记忆已由 EverMem 云接管，这条本地 BM25 历史检索与之重复、且会与
# EverMem 召回打架，故默认禁用（代码保留，置 LINA_HISTORY_RECALL=1 可启用，
# 如不接 EverMem 时作兜底）。
import os as _os
_HISTORY_RECALL_ON = _os.environ.get("LINA_HISTORY_RECALL", "").strip() in ("1", "true", "True")


# 主模型 prompt 已外置到 prompts/main/ 下的文件，便于修改（不再写死在代码里）。
# 内容与历史版本一字不差，只是改为从文件加载。overrides 机制不变。
_MAIN_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "main"


def _load_main_prompt(filename: str, fallback: str = "") -> str:
    """读 prompts/main/<filename>。缺失则返回 fallback（容错，不崩）。"""
    fp = _MAIN_PROMPTS_DIR / filename
    try:
        return fp.read_text(encoding="utf-8")
    except OSError:
        return fallback


# 注入主模型 user 轮的各上下文块固定文案，外置到 prompts/main/user_content_instructions.json，
# 支持网页提示词区按 main/user_content_instructions.json#<键> 编辑、override 即时生效。
_UC_REL = "main/user_content_instructions.json"


def _uc(key: str, fallback: str = "", **fmt) -> str:
    """读 user_content_instructions.json 的某条文案（经 controller 的 override 层），
    再用 **fmt 填占位（{mood} 等）。文件/字段缺失则用 fallback，保证不崩。"""
    try:
        from .controller._prompts import load_json_value
        text = load_json_value(_UC_REL, key, fallback=fallback)
    except Exception:
        text = fallback
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text


BEHAVIOR_RULES = _load_main_prompt("behavior_rules.txt")
MOOD_FORMAT_SPEC = _load_main_prompt("mood_format_spec.txt")
SEGMENT_PROTOCOL_SPEC = _load_main_prompt("segment_protocol_spec.txt")
SYSTEM_PROMPT_TEMPLATE = _load_main_prompt("system_prompt_template.txt")


# behavior_rules 按块拆分（prompts/main/behavior_blocks/）：base 常驻进缓存块，
# 其余按本轮场景注入，减短每轮 prompt。块文件缺失则回退用整份 BEHAVIOR_RULES。
def _load_behavior_block(name: str) -> str:
    fp = _MAIN_PROMPTS_DIR / "behavior_blocks" / f"{name}.txt"
    try:
        return fp.read_text(encoding="utf-8")
    except OSError:
        return ""


_BEHAVIOR_BASE = _load_behavior_block("base")
_BEHAVIOR_SCENE_BLOCKS = {
    "continuation_skills": _load_behavior_block("continuation_skills"),
    "empathy_scenes": _load_behavior_block("empathy_scenes"),
    "humor": _load_behavior_block("humor"),
}
# 拆分是否可用：base 块存在才启用按场景注入；否则回退整份（安全兜底）。
_BEHAVIOR_SPLIT_OK = bool(_BEHAVIOR_BASE.strip())

# 规则检索（EVEROS_RULES=1）：behavior 全走检索、mood 情绪表走检索，每轮按用户话
# 注入 top-k 条；缓存块里 behavior 留空、mood 只留「格式头+三条铁律」常驻（机制底线，
# 漏检会让输出格式崩）。检索关闭时一切照旧（整份注入）。
_RULES_RETRIEVAL_ON = _os.environ.get("EVEROS_RULES", "").strip() in ("1", "true", "True")


def _mood_format_head() -> str:
    """情绪标记格式说明 + 三条铁律（到「情绪表」前），机制底线，常驻。"""
    import re as _re
    head = _re.split(r"\*\*情绪表\*\*", MOOD_FORMAT_SPEC, maxsplit=1)[0].strip()
    return head or MOOD_FORMAT_SPEC


_MOOD_HEAD = _mood_format_head()


def _block_text(block) -> str:
    """从 system block / message content 里抽出纯文本（兼容 str 与 {type,text} dict / list）。"""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return block.get("text", "") or ""
    if isinstance(block, list):
        return "\n".join(_block_text(b) for b in block)
    return str(block)


def _render_final_prompt(system_blocks, api_messages) -> str:
    """把最终发给主模型的 system + messages 原样拼成可读文本（仅调试展示，不影响发送内容）。"""
    parts: list[str] = ["================ SYSTEM ================"]
    for i, blk in enumerate(system_blocks or [], 1):
        cache = ""
        if isinstance(blk, dict) and blk.get("cache_control"):
            cache = "  [cache]"
        parts.append(f"\n----- system block #{i}{cache} -----\n{_block_text(blk)}")
    parts.append("\n================ MESSAGES ================")
    for msg in api_messages or []:
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        content = msg.get("content") if isinstance(msg, dict) else msg
        parts.append(f"\n----- [{role}] -----\n{_block_text(content)}")
    return "\n".join(parts)


@dataclass
class ChatResult:
    text: str
    retrieved: list[Chunk]
    retrieved_history: list[Chunk]
    mood: dict | None = None  # {"mood": str, "intensity": int, "trust": int}
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    # Per-turn controller decision + trace, surfaced for the inspector.
    plan: dict | None = None
    controller_trace: dict | None = None
    # 日记/话题检索决策（供右侧调试面板展示）：need_diary/query_type/新旧话题/命中编号。
    diary_debug: dict | None = None
    # 最终拼给主模型的完整 prompt（system + messages 原样拼成可读文本，供右侧面板滚动查看）。
    final_prompt: str | None = None
    # 各阶段耗时（毫秒）：controller_ms / retrieval_ms / inference_ms。
    timings: dict | None = None
    # Remaining segment outline points this reply parked for later
    # continuation (empty = not split). Web layer arms the continue timer
    # on a non-empty list.
    pending_segments: list[str] = field(default_factory=list)
    # Marks a reply produced by continue_segment() (a follow-up chunk).
    is_continuation: bool = False
    # 「刚滑出窗口、待概括进自我事实清单」的几轮对话对（空 = 无）。
    # web 层据此在后台异步跑概括，不阻塞回复。
    slid_out_turns: list = field(default_factory=list)


def parse_mood_tag(raw: str) -> tuple[str, dict | None]:
    """Strip a leading mood tag from `raw`. Returns (cleaned_text, mood_dict | None).

    Tolerant: if the tag is absent or malformed, returns the text unchanged
    and `None` for mood — we still show whatever Claude said, but the UI
    will indicate that no mood was reported.
    """
    if not raw:
        return raw, None
    m = MOOD_TAG_RE.match(raw)
    if not m:
        return raw.strip(), None
    cleaned = raw[m.end() :].lstrip("\n").rstrip()
    try:
        intensity = max(1, min(10, int(m.group("intensity"))))
        trust = max(1, min(10, int(m.group("trust"))))
    except ValueError:
        return raw.strip(), None
    mood = {
        "mood": m.group("mood").strip(),
        "intensity": intensity,
        "trust": trust,
    }
    return cleaned, mood


class CharacterEngine:
    def __init__(
        self,
        api_key: str,
        static_dir: str | Path,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        history_window: int = 30,
        retrieve_k: int = 4,
        history_retrieve_k: int = 3,
        overrides: dict[str, str] | None = None,
        controller: LinaController | None = None,
    ):
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        self.api_key = api_key
        self.static_dir = Path(static_dir)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        # With a controller wired in, the per-turn Plan supplies these
        # (overriding the instance defaults). Without one, they're used as-is.
        self.history_window = history_window
        self.retrieve_k = retrieve_k
        self.history_retrieve_k = history_retrieve_k
        self.overrides: dict[str, str] = dict(overrides or {})
        self.rag = CharacterRAG(self.static_dir, file_overrides=self._file_overrides())
        # 可检索人设走本地 EverOS（agent 轨道语义检索）。启用时取代 self.rag 的
        # BM25 检索人设；未启用（EVEROS_PERSONA 未置）则 enabled=False，回退 self.rag。
        # CORE 人设（self.rag.core_text）不受影响，仍全量进 prompt。
        from .persona_memory import PersonaMemory
        self.persona_memory = PersonaMemory()
        from .rules_memory import RulesMemory
        self.rules_memory = RulesMemory()
        # 日记/谈资两级检索（本地 EverOS）。由 plan.need_diary 门控（controller 判定
        # 本轮该调莉娜生活经历时才检索）。未启用（EVEROS_DIARY 未置）则 enabled=False。
        from .diary_memory import DiaryMemory
        self.diary_memory = DiaryMemory()
        self._system_blocks = self._build_system_blocks()
        # Controller is optional. None → pre-controller behavior preserved.
        self._controller = controller
        self._composer = LinaPromptComposer() if controller is not None else None

    def _file_overrides(self) -> dict[str, str]:
        """Subset of overrides that target static .md files."""
        return {k: v for k, v in self.overrides.items() if k.endswith(".md")}

    def apply_overrides(self, overrides: dict[str, str] | None) -> None:
        """Replace overrides and rebuild RAG + cached system prompt in place."""
        self.overrides = dict(overrides or {})
        self.rag = CharacterRAG(self.static_dir, file_overrides=self._file_overrides())
        self._system_blocks = self._build_system_blocks()

    def _build_system_blocks(self) -> list[dict]:
        # 规则检索启用时：缓存块里 behavior 留空（全走检索，由动态块按当前话注入
        # top-k 条），mood 只留「格式头+三条铁律」常驻（机制底线）。网页 override 了
        # 对应 KEY 时尊重 override（整份），不走检索。
        if _RULES_RETRIEVAL_ON and "BEHAVIOR_RULES" not in self.overrides:
            behavior = ""
        else:
            behavior = self.overrides.get("BEHAVIOR_RULES", BEHAVIOR_RULES)
        if _RULES_RETRIEVAL_ON and "MOOD_FORMAT_SPEC" not in self.overrides:
            mood = _MOOD_HEAD
        else:
            mood = self.overrides.get("MOOD_FORMAT_SPEC", MOOD_FORMAT_SPEC)
        template = self.overrides.get("SYSTEM_PROMPT_TEMPLATE", SYSTEM_PROMPT_TEMPLATE)
        try:
            prompt = template.format(
                core_text=self.rag.core_text,
                behavior_rules=behavior,
                mood_format_spec=mood,
            )
        except (KeyError, IndexError):
            # User-provided template may have removed/renamed placeholders.
            # Fall back to a simple concatenation rather than crashing.
            prompt = (
                f"{template}\n\n{self.rag.core_text}\n\n{behavior}\n\n{mood}"
            )
        # Append the (optional) segment-splitting protocol. Stable text, so it
        # stays inside the cached block and doesn't hurt cache hits. The model
        # only emits a [segments:…] tag when splitting is appropriate.
        segment_spec = self.overrides.get("SEGMENT_PROTOCOL_SPEC", SEGMENT_PROTOCOL_SPEC)
        prompt = f"{prompt}\n\n================\n# 六、{segment_spec}"
        # Single cached system block. Prompt caching needs at least ~1024
        # tokens; the character corpus is well above that.
        return [
            {
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _history_pairs_for_controller(self, conversation: Conversation) -> tuple[tuple[str, str], ...]:
        """Turn the linear message list into (user, assistant) pairs for the
        controller context. System-triggered placeholders (proactive /
        farewell) are skipped so the controller sees only real exchanges."""
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for m in conversation.messages:
            if m.role == "user":
                if m.meta and m.meta.get("system_trigger"):
                    continue
                if pending_user is not None:
                    pairs.append((pending_user, ""))
                pending_user = (m.content or "").strip()
            elif m.role == "assistant":
                a_text = (m.content or "").strip()
                if pending_user is not None:
                    pairs.append((pending_user, a_text))
                    pending_user = None
                else:
                    pairs.append(("", a_text))
        if pending_user is not None:
            pairs.append((pending_user, ""))
        return tuple(pairs)

    @staticmethod
    def _gap_seconds(conversation: Conversation) -> float:
        """Seconds since the last stored message. 0 if no history.

        Used by the controller's welcome_back rule. We read the most recent
        message's `ts` and diff against now. A genuinely fresh conversation
        (no messages) returns 0 → never triggers welcome_back."""
        if not conversation.messages:
            return 0.0
        last_ts = getattr(conversation.messages[-1], "ts", 0.0) or 0.0
        return max(0.0, time.time() - float(last_ts))

    def _make_turn_context(
        self,
        user_message: str,
        conversation: Conversation,
        *,
        is_proactive: bool = False,
        is_farewell: bool = False,
        is_continuation: bool = False,
        has_cross_session_memory: bool = False,
    ) -> LinaTurnContext:
        """构造 controller 的 turn context（dispatch 和 match_topic 并发时复用）。"""
        has_prior_assistant = any(m.role == "assistant" for m in conversation.messages)
        last_meta = conversation.last_assistant_meta() or {}
        return LinaTurnContext(
            user_text=user_message,
            history=self._history_pairs_for_controller(conversation),
            session_id=getattr(conversation, "session_id", "") or "",
            is_proactive=is_proactive,
            is_farewell=is_farewell,
            is_continuation=is_continuation,
            is_first_turn=not has_prior_assistant,
            has_cross_session_memory=has_cross_session_memory,
            prior_trust=int(last_meta.get("trust", 3) or 3),
            gap_seconds=self._gap_seconds(conversation),
        )

    def _dispatch_controller(
        self,
        user_message: str,
        conversation: Conversation,
        *,
        is_proactive: bool = False,
        is_farewell: bool = False,
        is_continuation: bool = False,
        has_cross_session_memory: bool = False,
    ) -> tuple[LinaPromptPlan, dict[str, Any] | None]:
        """Run controller and return (plan, trace). Without a controller
        wired in, returns a `LinaPromptPlan` populated with the engine's
        legacy defaults so the rest of the code path is identical."""
        if self._controller is None:
            plan = LinaPromptPlan(
                retrieve_k=self.retrieve_k,
                history_recall_k=self.history_retrieve_k,
                history_window=self.history_window,
                use_cross_session_memory=has_cross_session_memory,
                trace_source="disabled",
                matched_rule="controller_disabled",
            )
            return plan, None

        ctx = self._make_turn_context(
            user_message, conversation,
            is_proactive=is_proactive, is_farewell=is_farewell,
            is_continuation=is_continuation, has_cross_session_memory=has_cross_session_memory,
        )
        plan = self._controller.dispatch_sync(ctx)
        # 决策可观测：每轮把关键判断打到日志，方便不开断点就复盘 controller 行为。
        try:
            import logging
            logging.getLogger("lina.controller").info(
                "[plan] rule=%s src=%s | suppress_q=%s lenient_typo=%s allow_seg=%s "
                "self_facts=%s tone=%s sent=%s chars=%s | u=%r",
                plan.matched_rule, plan.trace_source,
                plan.suppress_trailing_question, plan.lenient_typos, plan.allow_segment,
                plan.use_self_facts, plan.tone_hint, plan.sentences, plan.max_reply_chars,
                (user_message or "")[:40],
            )
        except Exception:
            pass
        return plan, self._controller.last_trace

    @staticmethod
    def _filter_chunks_by_plan(chunks: list[Chunk], plan: LinaPromptPlan) -> list[Chunk]:
        """Drop static chunks whose source file is disabled in the plan.

        Run AFTER RAG retrieval so we don't change BM25 scoring; we just
        suppress chunks the plan said we shouldn't use.

        注意：本过滤器是为本地 md 文件 RAG（source 形如 'personality.md'）设计的。
        EverOS agent 轨道（PersonaMemory）回来的 chunk source 不带 .md 后缀
        （如 'personality'/'world'），不在 static_sources 命名体系内——这类
        chunk 由语义检索 + controller 的 query_hint 自然导向，不参与本过滤、
        原样放行（否则会被全部误丢，导致「检索资料」恒空）。"""
        allowed = set(plan.static_sources)
        # allowed 为空 = controller 未对来源设限 → 不过滤，全保留。
        if not allowed:
            return list(chunks)
        # 只对「带 .md 后缀的本地 md 来源」做白名单过滤；EverOS 来源（无 .md）放行。
        return [
            c
            for c in chunks
            if (not c.source.endswith(".md")) or (c.source in allowed)
        ]

    def _build_dynamic_system_blocks(self, plan: LinaPromptPlan, rules_query: str = "") -> list[dict]:
        """Build optional second system block from the controller plan.

        Returns an empty list when there's nothing dynamic to add (caller
        then uses just the cached block — the pre-controller payload)."""
        parts: list[str] = []
        if self._composer is not None:
            bundle = self._composer.compose(plan)
            if bundle.tail_text.strip():
                parts.append(bundle.tail_text)
        # 规则检索：按本轮用户话语义检索 behavior top-8 + mood 情绪 top-6 注入
        # （EVEROS_RULES=1 且未 override 整份时）。缓存块里 behavior 已留空、mood 仅留
        # 格式头，所以这里把「本轮用得上的」规则补进来，省掉全量 2 万字。
        if _RULES_RETRIEVAL_ON and rules_query.strip() and self.rules_memory.enabled:
            if "BEHAVIOR_RULES" not in self.overrides:
                bh = self.rules_memory.retrieve_behavior(rules_query, k=8)
                if bh.strip():
                    parts.append("# 行为规则（本轮相关，必须遵守）\n\n" + bh)
            if "MOOD_FORMAT_SPEC" not in self.overrides:
                md = self.rules_memory.retrieve_mood(rules_query, k=6)
                if md.strip():
                    parts.append("# 情绪表（本轮可能用到的情绪）\n\n" + md)
        if not parts:
            return []
        # Second block is NOT cache_control'd: it changes per turn.
        return [{"type": "text", "text": "\n\n".join(parts)}]

    @staticmethod
    def _scene_behavior_blocks(plan: LinaPromptPlan) -> list[str]:
        """[已弃用] 旧的按场景挑 behavior_blocks 逻辑，规则检索启用后不再调用。
        没有 controller（plan 为默认）时，为安全起见全注入（等价整份）。"""
        if plan is None:
            return ["continuation_skills", "empathy_scenes", "humor"]
        picked: list[str] = []
        # 续聊技巧：本轮要勾话头/给细节/引用历史时。
        if getattr(plan, "hook_concrete_example", False) or getattr(plan, "hook_callback", False) \
                or getattr(plan, "hook_history_recall", False):
            picked.append("continuation_skills")
        # 共情/关系场景应对：安抚或关系回访时。
        if getattr(plan, "module_user_vent", False) or getattr(plan, "module_relationship_recall", False):
            picked.append("empathy_scenes")
        # 幽默：轻松/兴奋点场景，或语气偏轻松调侃时。
        tone = (getattr(plan, "tone_hint", "") or "")
        # 幽默：兴奋点场景，或明确轻松/调侃语气（不含「自然」——它几乎每轮都有，会失效）。
        if getattr(plan, "module_world_immersion", False) or any(
            k in tone for k in ("轻松", "雀跃", "调侃", "俏皮", "幽默")
        ):
            picked.append("humor")
        return picked

    def _build_user_content(
        self,
        user_message: str,
        retrieved: list[Chunk],
        retrieved_history: list[Chunk],
        prior_mood: dict | None,
        is_forced: bool = False,
        is_first_turn: bool = False,
        quoted_text: str = "",
        self_facts_text: str = "",
        user_facts_text: str = "",
        diary_text: str = "",
        diary_continuation: bool = False,
        pending_segments: list[str] | None = None,
    ) -> str:
        sections: list[str] = []

        if prior_mood:
            if is_forced:
                sections.append(_uc(
                    "mood_forced",
                    fallback="<状态强制设定 — 用户刚刚手动设定了莉娜此刻的情绪和信任度。"
                    "请直接从这个状态出发回复，不要质疑、纠正或试图「回到上一轮的状态」。"
                    "这一轮她就是这个状态。>\n情绪：{mood} / 强度 {intensity}\n"
                    "对该用户的信任度：{trust} / 10\n</状态强制设定>",
                    mood=prior_mood.get('mood', '平静'), intensity=prior_mood.get('intensity', 5),
                    trust=prior_mood.get('trust', 3),
                ))
            elif is_first_turn:
                sections.append(_uc(
                    "mood_first_turn",
                    fallback="<起始状态 — 这是莉娜与该用户的第一次接触，请用以下默认值作为起点>\n"
                    "情绪：{mood} / 强度 {intensity}\n对该用户的信任度：{trust} / 10\n"
                    "**强制规则**：信任度 3 是陌生人的标准起始值。"
                    "**不要**因为用户开场白礼貌或友好就在第一轮就抬高到 5/7/8。"
                    "信任度的调整严格遵循「情绪连续性」规则：用户共情/真懂古代文化才 +1/+2；"
                    "用户矛盾/冒犯才 -1/-2。如果用户只是简单打招呼或客套，这一轮的信任度"
                    "应仍然是 3（最多 4）。\n</起始状态>",
                    mood=prior_mood.get('mood', '平静'), intensity=prior_mood.get('intensity', 5),
                    trust=prior_mood.get('trust', 3),
                ))
            else:
                sections.append(_uc(
                    "mood_ongoing",
                    fallback="<近况 — 莉娜目前的状态，影响这一轮的回复>\n"
                    "上一轮情绪：{mood} / 强度 {intensity}\n当前对该用户的信任度：{trust} / 10\n"
                    "请从这个状态出发，让本轮回复带着上一轮的情绪余韵；"
                    "信任度只能小幅 (±1/±2) 调整，不要突变。\n</近况>",
                    mood=prior_mood.get('mood', '?'), intensity=prior_mood.get('intensity', '?'),
                    trust=prior_mood.get('trust', '?'),
                ))

        # 莉娜的「自我事实清单」命中项——由 controller 决定本轮是否检索（第 4 个
        # 检索库），按当前话题 BM25 检出相关几条（不再整份常驻）。她亲口说过、
        # 人设文件里没写的稳定事实（养猫、承诺、喜好…），保证自我一致。
        if self_facts_text:
            _h = _uc(
                "self_facts_header",
                fallback="<莉娜的自我设定记忆 — 你（莉娜）之前亲口说过的、与本轮相关的关于你自己的事实。"
                "务必与这些保持一致，不要自相矛盾；自然引用即可，不要生硬复述。>",
            )
            sections.append(f"{_h}\n{self_facts_text}\n</莉娜的自我设定记忆>")

        # 莉娜日记/谈资命中项——由 controller 判定本轮该找谈资时，从预设日记库（两级
        # 检索：话题→日记）调出的莉娜真实经历。用途：用户提到相关话题时有真事可聊、
        # 主动找话有真素材，**不凭空编造**。只作背景素材，自然化用，别照搬复述。
        if diary_text:
            _cont = _uc(
                "diary_continuation_note",
                fallback="（⚠️ 本轮用户是在**接着上一个话题继续追问**，比如用代词/省略问「他还好吗」"
                "「那现在呢」。下面就是上次聊这个话题时用的同一份日记——**接着上次说过的口径往下答，"
                "保持前后一致，不要改口、不要重新编一套**。）\n",
            ) if diary_continuation else ""
            _h = _uc(
                "diary_header",
                fallback="<莉娜的谈资与日记 — 这是为本轮话题匹配到的【话题卡】+ 卡里精确关联的【真实日记】。\n"
                "{continuation}话题卡是你这个角色的「谈资说明书」；日记是你真实经历过的事。\n"
                "**铁律（必须遵守）：**\n1. **给了内容就一定要用它来聊，不许回避**。\n"
                "2. **绝不编造与日记冲突的事实**。\n3. 照着话题卡的「暴露边界」「共情方式」来。\n"
                "4. 用户倾诉/难过时先共情他、再自然带出你的类似经历。\n"
                "5. 自然化用，用你自己的语气说，不要整段照搬。>",
                continuation=_cont,
            )
            sections.append(f"{_h}\n{diary_text}\n</莉娜的谈资与日记>")

        # 「用户事实清单」命中项——用户在过往（可能很多轮前、已滑出窗口）讲过的、
        # 关于他自己的稳定事实。这是解决「聊久了把用户的事忘了/记混/编造」的关键：
        # 按当前话题 BM25 检出相关几条注入，让莉娜记得用户是谁、聊过什么。
        if user_facts_text:
            _h = _uc(
                "user_memory_header",
                fallback="<关于用户的记忆 — 用户之前讲过的、关于他自己的事实（可能是很久以前说的）。"
                "请把这些当作你确实记得的事，自然地体现出「记得他」；但**只用这里列出的**，"
                "不要凭印象编造用户没说过的事，也不要生硬复述。>",
            )
            sections.append(f"{_h}\n{user_facts_text}\n</关于用户的记忆>")
        else:
            # 没有关于这个用户的任何记忆（如全新会话）。**诚实**：不要装熟、不要编
            # 「你来过几次」「眼熟」这种。用户若问「还记得我吗」，就如实说一时想不起来/
            # 我们好像还没怎么聊过——不假装记得一个其实没有记录的人。
            sections.append(_uc(
                "user_memory_empty",
                fallback="<关于用户的记忆 — （空：你这边没有关于这个用户的任何记忆。）\n"
                "**这意味着你其实不认识他、或还没怎么聊过。诚实对待：不要装熟、不要编造"
                "「你来过几次」「有点眼熟」「记得你」这类。若他问你是否记得他，就如实说"
                "一时想不起来 / 我们好像还没正式聊过。>",
            ))

        if retrieved:
            static_text = "\n\n".join(c.render() for c in retrieved)
            _h = _uc(
                "char_ref_header",
                fallback="<角色设定参考 — 与本轮对话相关的设定细节，仅作背景，不要照搬其措辞或括号动作>",
            )
            sections.append(f"{_h}\n{static_text}\n</角色设定参考>")
        if retrieved_history:
            hist_text = "\n\n".join(c.render() for c in retrieved_history)
            _h = _uc(
                "history_recall_header",
                fallback="<历史回忆 — 来自本会话更早轮次的相关片段。这些都是已经发生过的对话，"
                "用户之前讲过的事实/偏好/承诺，请记住并保持一致；不要重复或复述。>",
            )
            sections.append(f"{_h}\n{hist_text}\n</历史回忆>")
        # 用户像微信那样"引用"了莉娜之前的某条消息来回复——明确告诉模型
        # 这一轮是针对哪句话说的，回复要承接那句，而不是泛泛而谈。
        quoted = str(quoted_text or "").strip()
        if quoted:
            _h = _uc(
                "quoted_header",
                fallback="<用户引用了你之前说过的这句话来回复 — 本轮请明确承接、回应这句，不要答非所问>",
            )
            sections.append(f"{_h}\n{quoted}\n</用户引用>")
        # 打断接续判断：上一条莉娜还有没说完的段（park 在 pending_segments），
        # 用户这时插了新话。不直接作废，而是把没说完的要点作为背景交给主模型，
        # 由它自己判断——新消息跟这些要点相关就先回应、再自然接着说完；岔开了
        # 就放下。判断权在主模型，不再机械「插话即丢弃」。
        pending = [str(p).strip() for p in (pending_segments or []) if str(p or "").strip()]
        if pending:
            points = "、".join(pending)
            _h = _uc(
                "pending_segments_header",
                fallback="<你上一条还没说完的话 — 你之前分段说话时，还剩下面这些要点没说，"
                "用户这会儿插了新消息进来。请你自己判断：\n"
                "- 如果用户的新消息跟这些要点还相关，就先回应新消息，再自然把相关那点接着说完；\n"
                "- 如果用户明显岔开、换了话题，就放下这些要点，专心回应新消息。\n"
                "不管接不接，都不要提到「我刚才还想说」这类元叙述。>",
            )
            sections.append(f"{_h}\n{points}\n</你上一条还没说完的话>")
        sections.append(f"<用户发言>\n{user_message}\n</用户发言>")
        if not sections[:-1]:  # only user message present, no context blocks
            return user_message
        return "\n\n".join(sections)

    def _prepare(
        self,
        conversation: Conversation,
        user_message: str,
        *,
        extra_memory_chunks: list[Chunk] | None = None,
        quoted_text: str = "",
        self_facts: dict | None = None,
        user_facts: dict | None = None,
        user_memory_text: str = "",
        diary_text: str = "",
    ) -> dict:
        """Shared setup for chat() and chat_stream(): controller dispatch,
        plan-driven RAG, mood seeding, the assembled API `messages` list, and
        the (two-block) system payload. Pure (no mutation of the
        conversation), so streaming can abort without side effects."""
        import logging as _logging
        _dlog = _logging.getLogger("lina.controller")
        diary_continuation = False
        diary_debug: dict = {"enabled": bool(self.diary_memory.enabled)}

        # 各阶段耗时（毫秒），供右侧面板展示。controller=判定+判别器并发段，
        # retrieval=日记/话题/人设检索段，inference 由 chat()/chat_stream() 另填。
        timings: dict = {}
        _t_ctrl0 = time.perf_counter()

        # 1) Controller 判定（plan）+ 话题连续性判断器（m）。
        # 二者互不依赖：当 controller 在 + 日记启用时，**一次并发**同时发两个 LLM 调用
        # （dispatch_and_match_sync），省串行往返延迟；否则退回单独 dispatch。
        m = None
        run_diary = (not diary_text) and self.diary_memory.enabled and self._controller is not None
        if run_diary:
            pool = conversation.topic_pool or {}
            recent_pairs = list(self._history_pairs_for_controller(conversation))[-6:]
            ctx_cm = self._make_turn_context(user_message, conversation,
                                             has_cross_session_memory=bool(extra_memory_chunks))
            try:
                plan, m = self._controller.dispatch_and_match_sync(
                    ctx_cm, user_message, recent_pairs, pool)
                plan_trace = self._controller.last_trace
            except Exception as e:
                _dlog.info("[diary] 并发 dispatch+match 失败，退回串行: %s", e)
                plan, plan_trace = self._dispatch_controller(
                    user_message, conversation, has_cross_session_memory=bool(extra_memory_chunks))
                m = None
        else:
            plan, plan_trace = self._dispatch_controller(
                user_message, conversation, has_cross_session_memory=bool(extra_memory_chunks))
        # controller + 判别器并发段结束。
        timings["controller_ms"] = round((time.perf_counter() - _t_ctrl0) * 1000, 1)
        _t_retr0 = time.perf_counter()

        # 日记/谈资两级检索：用并发拿到的 m（话题连续性判断器结果）走分支。
        if run_diary and m is not None:
            diary_debug.update({
                "need_diary": bool(m.get("need_diary")),
                "query_type": m.get("query_type"),
                "search_query": m.get("search_query"),
            })
            mid = m.get("matched_topic_id")
            if mid and mid in pool:
                # 接续老话题 → 复用缓存（保连续）+ 直检补更具体的日记（保深入）。
                # **关键（治状态漂移 C10）**：同一话题已经用过的日记ID 锁在 locked_ids 里，
                # 直检补充的日记**累加进锁定集、之后一直带着**——所以培根第一次锁了哪几篇
                # （昏迷篇等），后续每轮都用同一批，不会这轮昏迷、下轮换成告别篇导致状态飘。
                diary_continuation = True
                pool[mid].setdefault("turns", []).append(len(conversation.messages))
                locked_ids = pool[mid].get("locked_ids") or list(pool[mid].get("diary_ids") or [])
                try:
                    extra = self.diary_memory.search_diaries_direct(
                        m.get("search_query") or user_message, k=3
                    )
                except Exception:
                    extra = []
                added = 0
                for d in extra:
                    did = d.get("name", "").replace("diary_", "")
                    if did and did not in locked_ids:
                        locked_ids.append(did)
                        added += 1
                # 用锁定集里的【全部】日记重建注入文本（话题卡正文 + 所有锁定日记），稳定不漂
                texts = [pool[mid].get("diary_text", "")]
                got = self.diary_memory.get_diaries_by_ids(locked_ids)
                for did in locked_ids:
                    if did in got and got[did] not in texts[0]:
                        texts.append(f"〔莉娜的相关经历〕{self.diary_memory._clean(got[did])}")
                diary_text = "\n\n———\n\n".join(t for t in texts if t.strip())
                pool[mid]["locked_ids"] = locked_ids  # 锁回缓存，下轮继续带
                conversation.topic_pool = pool
                diary_debug.update({
                    "mode": "接续老话题", "topic": pool[mid].get("title", mid),
                    "locked_diary_ids": list(locked_ids), "new_added": added,
                })
                _dlog.info("[diary] 接续老话题 %s（锁定 %d 篇，本轮新增 %d）",
                           pool[mid].get("title", mid), len(locked_ids), added)
            elif m.get("need_diary"):
                # 新话题 → 检索 + 存池。
                # 检索词用**判断器产出的干净 search_query**（它看了上下文、把代词/省略补全成
                # 自包含的查询，如「他还好吗」→「培根老师的近况」）。不再把前几轮原话拼进
                # 检索——那会稀释当前问题、召回跑偏（培根→炼金器具的 bug）。判断器没给
                # query 时退回当前句。
                ctx = m.get("search_query") or (user_message or "").strip()
                try:
                    cards = self.diary_memory.retrieve_cards(ctx, query_type=m.get("query_type") or "global")
                except Exception as e:
                    cards = []
                    _dlog.info("[diary] 检索失败: %s", e)
                if cards:
                    diary_text = "\n\n———\n\n".join(c["text"] for c in cards)
                    new_pool = dict(pool)
                    for c in cards:
                        new_pool[c["topic_id"]] = {
                            "title": c["title"], "diary_text": c["text"],
                            "diary_ids": c["diary_ids"], "turns": [len(conversation.messages)],
                        }
                    conversation.topic_pool = new_pool
                    diary_debug.update({
                        "mode": "新话题检索",
                        "hit_topics": [c["title"] for c in cards],
                        "hit_diary_ids": [d for c in cards for d in c.get("diary_ids", [])],
                    })
                    _dlog.info("[diary] 新话题检索+存池: %s", [c["title"] for c in cards])
                else:
                    diary_debug["mode"] = "需日记但无命中→回退人设"
                    _dlog.info("[diary] need_diary=True 但无完整话题卡命中 → 回退人设")
            else:
                diary_debug["mode"] = "本轮不需日记"
                _dlog.info("[diary] 判断器: 本轮不需日记")

        # 2) RAG controlled by the plan (falls back to instance defaults when
        #    no controller is wired in).
        rag_query = plan.query_hint or user_message
        # 自我事实清单作为第 4 个检索库：仅当 plan.use_self_facts 时，按当前话题
        # BM25 检索命中的几条注入（不再整份常驻），省 token、避免无关注入。
        self_facts_text = ""
        if self_facts and (plan.use_self_facts or self._controller is None):
            try:
                from .self_facts import SelfFactsStore
                self_facts_text = SelfFactsStore.search(self_facts, rag_query, k=5)
            except Exception:
                self_facts_text = ""
        # 用户事实清单：长期记住用户的核心。清单本就很短（上限 ~40 条），所以
        # **较短时整份注入**（不靠 BM25），避免「换个说法问就检不到」——让主模型
        # 自己从完整清单里找。只有清单异常大时才退回 BM25 检索控量。
        # 用户记忆注入。**优先用外部传入的 user_memory_text**（web 层从 EverMem
        # 语义检索得到的「关于用户的记忆」文本）——这条路取代了原 user_facts 的
        # BM25/整份注入，但**注入位置完全相同**（仍是下面 _build_user_content 的
        # 「关于用户的记忆」块），主模型 prompt 模板一字不动。
        # 没传 user_memory_text 时，回退原 user_facts 逻辑（兼容、兜底）。
        user_facts_text = ""
        if user_memory_text.strip():
            user_facts_text = user_memory_text.strip()
        elif user_facts:
            try:
                from .user_facts import UserFactsStore
                total = sum(len(v) for v in user_facts.values() if isinstance(v, list))
                if total <= 40:
                    user_facts_text = UserFactsStore.render(user_facts)   # 整份注入
                else:
                    user_facts_text = UserFactsStore.search(user_facts, rag_query, k=10)
            except Exception:
                user_facts_text = ""
        retrieve_k = plan.retrieve_k if self._controller is not None else self.retrieve_k
        # 人设检索：启用 EverOS 时走 agent 轨道语义检索（取代 BM25），失败/空回退
        # self.rag.retrieve。注入位置不变（下面「角色设定参考」块），CORE 人设不动。
        retrieved_raw: list[Chunk] = []
        if retrieve_k > 0:
            if self.persona_memory.enabled:
                retrieved_raw = self.persona_memory.retrieve(rag_query, k=retrieve_k)
            if not retrieved_raw:
                retrieved_raw = self.rag.retrieve(rag_query, k=retrieve_k)
        retrieved = (
            self._filter_chunks_by_plan(retrieved_raw, plan)
            if self._controller is not None
            else retrieved_raw
        )

        history_window = plan.history_window if self._controller is not None else self.history_window
        history_recall_k = (
            plan.history_recall_k if self._controller is not None else self.history_retrieve_k
        )
        retrieved_history: list[Chunk] = []
        if _HISTORY_RECALL_ON and plan.use_history_recall and history_recall_k > 0:
            retrieved_history = retrieve_history_chunks(
                conversation.messages,
                rag_query,
                k=history_recall_k,
                exclude_recent_count=history_window,
            )
        # Cross-session user memory passed in by the caller (web layer).
        if extra_memory_chunks and plan.use_cross_session_memory:
            retrieved_history = list(extra_memory_chunks) + retrieved_history
        # 检索段结束（日记/话题/人设/历史）。
        timings["retrieval_ms"] = round((time.perf_counter() - _t_retr0) * 1000, 1)

        # 3) Mood / first-turn / forced-state seeding (unchanged).
        prior_mood = conversation.last_assistant_meta()
        forced = conversation.forced_state
        has_prior_assistant = any(m.role == "assistant" for m in conversation.messages)
        is_first_turn = False
        if forced:
            merged = {"mood": "平静", "intensity": 5, "trust": 3}
            if prior_mood:
                merged.update(prior_mood)
            merged.update(forced)
            prior_mood = merged
        elif not has_prior_assistant:
            prior_mood = {"mood": "平静", "intensity": 5, "trust": 3}
            is_first_turn = True

        # 4) Build the API messages: prior history (windowed) + new user turn.
        prior: list[dict] = []
        for m in conversation.messages:
            if m.role == "assistant" and m.meta:
                tag = f"[mood: {m.meta.get('mood', '?')} | {m.meta.get('intensity', 5)} | 信任={m.meta.get('trust', 3)}]"
                prior.append({"role": "assistant", "content": f"{tag}\n{m.content}"})
            else:
                prior.append({"role": m.role, "content": m.content})
        if history_window > 0:
            prior = prior[-history_window:]
        api_messages = prior + [
            {
                "role": "user",
                "content": self._build_user_content(
                    user_message,
                    retrieved,
                    retrieved_history,
                    prior_mood,
                    is_forced=bool(forced),
                    is_first_turn=is_first_turn,
                    quoted_text=quoted_text,
                    self_facts_text=self_facts_text,
                    user_facts_text=user_facts_text,
                    diary_text=diary_text,
                    diary_continuation=diary_continuation,
                    # 上一轮 park 下来、还没说完的段（用户这轮插话）。交给主模型
                    # 自己判断接不接（见 _build_user_content）。仅 controller 模式下启用，
                    # 避免无 controller 的简单模式行为变化。
                    pending_segments=(
                        list(conversation.pending_segments or [])
                        if self._controller is not None
                        else None
                    ),
                ),
            }
        ]
        # 5) Two-segment system: cached block + plan-derived tail.
        system_blocks = self._system_blocks + self._build_dynamic_system_blocks(plan, rules_query=user_message)
        return {
            "retrieved": retrieved,
            "retrieved_history": retrieved_history,
            "forced": forced,
            "api_messages": api_messages,
            "system_blocks": system_blocks,
            "plan": plan,
            "plan_trace": plan_trace,
            "diary_debug": diary_debug,
            "diary_text": diary_text,
            # 调试用：把最终发给主模型的 system + messages 原样拼成可读文本，供右侧面板展示。
            # 只读拼接，不改任何发送内容。
            "final_prompt": _render_final_prompt(system_blocks, api_messages),
            "timings": timings,
        }

    def _slid_out_turns(
        self, conversation: Conversation, history_window: int
    ) -> list[tuple[str, str]]:
        """返回「刚滑出 history_window、值得抢救进自我事实清单」的几轮对话对。
        纯计算、无 LLM 调用——真正的概括由调用方在后台异步跑（不阻塞用户）。

        只取窗口边界往外的一小段（最多 4 轮），既不重复窗口内原文，又在它彻底
        消失前抓住关键自我陈述。"""
        if self._controller is None or not self._controller.has_llm:
            return []
        pairs = self._history_pairs_for_controller(conversation)
        if len(pairs) <= history_window:
            return []  # 还没滑出任何轮次
        slid_end = len(pairs) - history_window
        slid_start = max(0, slid_end - 4)
        return list(pairs[slid_start:slid_end])

    def chat(
        self,
        conversation: Conversation,
        user_message: str,
        extra_memory_chunks: list[Chunk] | None = None,
        quoted_text: str = "",
        self_facts: dict | None = None,
        user_facts: dict | None = None,
        user_memory_text: str = "",
        diary_text: str = "",
    ) -> ChatResult:
        prep = self._prepare(
            conversation,
            user_message,
            extra_memory_chunks=extra_memory_chunks,
            quoted_text=quoted_text,
            self_facts=self_facts,
            user_facts=user_facts,
            user_memory_text=user_memory_text,
            diary_text=diary_text,
        )
        retrieved = prep["retrieved"]
        retrieved_history = prep["retrieved_history"]
        forced = prep["forced"]
        plan = prep["plan"]
        plan_trace = prep["plan_trace"]
        diary_debug = prep.get("diary_debug")
        final_prompt = prep.get("final_prompt")
        timings = dict(prep.get("timings") or {})
        history_window = plan.history_window if self._controller is not None else self.history_window

        _t_inf0 = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prep["system_blocks"],
            messages=prep["api_messages"],
        )
        timings["inference_ms"] = round((time.perf_counter() - _t_inf0) * 1000, 1)

        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        raw_reply = "".join(text_parts).strip()
        # Strip leading mood tag, then trailing segment-plan tag.
        cleaned_reply, mood = parse_mood_tag(raw_reply)
        cleaned_reply, segments = parse_segments_tag(cleaned_reply)
        # 尊重 controller 的切分预算：本轮不准拆 → 丢弃模型可能误带的段；
        # 准拆 → 按 plan.max_segments 截断。（parse 已先剥掉标记，不会外露。）
        if self._controller is not None and not plan.allow_segment:
            segments = []
        else:
            segments = segments[: max(0, plan.max_segments - 1)]  # 第一段已发，余下 max-1 段
        # A fresh user turn invalidates any earlier split plan, installs this one.
        conversation.pending_segments = segments or None

        # Persist user (raw) and assistant (cleaned, with mood as meta).
        conversation.add("user", user_message)
        conversation.add("assistant", cleaned_reply, meta=mood)
        # forced_state is one-shot: consumed by the turn it influenced.
        if forced:
            conversation.forced_state = None

        # 自我事实清单：算出「刚滑出窗口、待概括」的几轮（纯计算，不调 LLM）。
        # 真正的概括由 web 层在后台线程异步跑，不阻塞本次回复返回。
        slid_out = self._slid_out_turns(conversation, history_window)

        usage = response.usage
        return ChatResult(
            text=cleaned_reply,
            retrieved=retrieved,
            retrieved_history=retrieved_history,
            mood=mood,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            plan=plan.to_dict(),
            controller_trace=plan_trace,
            diary_debug=diary_debug,
            final_prompt=final_prompt,
            timings=timings,
            pending_segments=list(segments),
            slid_out_turns=slid_out,
        )

    def chat_stream(self, conversation: Conversation, user_message: str, self_facts: dict | None = None, user_facts: dict | None = None, user_memory_text: str = "", diary_text: str = ""):
        """Streaming counterpart to chat(). A generator yielding event dicts:

            {"type": "mood",  "mood": {...}|None}   — emitted once, as soon as
                                                       the leading [mood:] line
                                                       is parsed off the front.
            {"type": "delta", "text": "..."}        — incremental visible text,
                                                       mood line already stripped.
            {"type": "done",  "mood", "retrieved",
             "retrieved_history", "usage"}          — final summary.

        The conversation is persisted (user + assistant turns) ONLY when the
        stream runs to completion. If the consumer stops iterating early — the
        interrupt case — `GeneratorExit` propagates through the open stream,
        aborting the Anthropic request, and nothing is saved. That matches the
        product rule: a barged-in turn is treated as a mistake and discarded.
        """
        prep = self._prepare(conversation, user_message, self_facts=self_facts, user_facts=user_facts, user_memory_text=user_memory_text, diary_text=diary_text)
        retrieved = prep["retrieved"]
        retrieved_history = prep["retrieved_history"]
        forced = prep["forced"]
        timings = dict(prep.get("timings") or {})
        _t_inf0 = time.perf_counter()

        header_done = False  # have we parsed/stripped the [mood:] header line?
        _ = prep  # system_blocks/plan used below
        header_buf = ""
        mood: dict | None = None
        cleaned_parts: list[str] = []
        # Once a '[' appears in the body it MIGHT be the start of a trailing
        # [segments:…] tag. We hold back text from that '[' onward (instead of
        # streaming it as a delta) so the tag never shows in captions or gets
        # spoken by TTS. At the end we decide: real segments tag → drop it;
        # stray '[' → flush it as a final delta.
        tail_buf = ""

        def _emit_header(buf: str):
            """Parse the mood tag off `buf`; return (mood, cleaned_visible_text)."""
            return parse_mood_tag(buf)

        def _push_body(text: str):
            """Stream body text as deltas, but hold back from any '[' (possible
            segments-tag start). Yields delta events; returns nothing."""
            nonlocal tail_buf
            out = []
            if tail_buf:
                # Already holding back — keep accumulating, emit nothing.
                tail_buf += text
                return out
            br = text.find("[")
            if br < 0:
                cleaned_parts.append(text)
                out.append({"type": "delta", "text": text})
            else:
                before = text[:br]
                if before:
                    cleaned_parts.append(before)
                    out.append({"type": "delta", "text": before})
                tail_buf = text[br:]   # start holding back from the '['
            return out

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prep["system_blocks"],
            messages=prep["api_messages"],
        ) as stream:
            for delta in stream.text_stream:
                if not delta:
                    continue
                if not header_done:
                    header_buf += delta
                    # The mood tag occupies the first line; once we've seen a
                    # newline the header is complete and we can strip it.
                    if "\n" in header_buf:
                        cleaned, mood = _emit_header(header_buf)
                        header_done = True
                        yield {"type": "mood", "mood": mood}
                        for ev in _push_body(cleaned):
                            yield ev
                    continue
                for ev in _push_body(delta):
                    yield ev

            # Stream finished. If we never saw a newline (single-line reply or
            # a dropped tag), parse whatever we buffered now.
            if not header_done:
                cleaned, mood = _emit_header(header_buf)
                yield {"type": "mood", "mood": mood}
                for ev in _push_body(cleaned):
                    yield ev

            final = stream.get_final_message()

        # Resolve the held-back tail: if it's a [segments:…] tag, strip it; if
        # it was just stray bracketed text, flush it as one final delta.
        segments: list[str] = []
        if tail_buf:
            tail_clean, segments = parse_segments_tag(tail_buf)
            if tail_clean:
                cleaned_parts.append(tail_clean)
                yield {"type": "delta", "text": tail_clean}

        cleaned_full = "".join(cleaned_parts).strip()
        conversation.pending_segments = segments or None

        # Reached only on normal completion — safe to persist.
        conversation.add("user", user_message)
        conversation.add("assistant", cleaned_full, meta=mood)
        if forced:
            conversation.forced_state = None

        usage = getattr(final, "usage", None)
        timings["inference_ms"] = round((time.perf_counter() - _t_inf0) * 1000, 1)
        yield {
            "type": "done",
            "text": cleaned_full,
            "mood": mood,
            "retrieved": retrieved,
            "retrieved_history": retrieved_history,
            "pending_segments": list(segments),
            # 调试面板字段（与非流式 /api/chat 对齐）：controller 判定、日记检索、最终 prompt、耗时。
            "plan": prep["plan"].to_dict() if prep.get("plan") else None,
            "diary_debug": prep.get("diary_debug"),
            "final_prompt": prep.get("final_prompt"),
            "timings": timings,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            },
        }

    # 主动发言指令：用户沉默时让莉娜根据近况主动开口。作为一条临时 user
    # 轮注入，不入库；只有她的回复会被保存。
    PROACTIVE_INSTRUCTION = _load_main_prompt("proactive_instruction.txt")

    def proactive(
        self,
        conversation: Conversation,
        extra_memory_chunks: list[Chunk] | None = None,
        user_facts: dict | None = None,
        self_facts: dict | None = None,
    ) -> ChatResult:
        return self._run_proactive(
            conversation,
            instruction=self.overrides.get("PROACTIVE_INSTRUCTION", self.PROACTIVE_INSTRUCTION),
            mode="engage",
            extra_memory_chunks=extra_memory_chunks,
            is_farewell=False,
            user_facts=user_facts,
            self_facts=self_facts,
        )

    # 告别指令：用户连续多次没回应主动搭话时，让莉娜自然地结束话题。
    # 与 PROACTIVE_INSTRUCTION 并列，作为另一种"主动开口"的语气。
    FAREWELL_INSTRUCTION = _load_main_prompt("farewell_instruction.txt")

    def proactive_farewell(
        self,
        conversation: Conversation,
        extra_memory_chunks: list[Chunk] | None = None,
        user_facts: dict | None = None,
        self_facts: dict | None = None,
    ) -> ChatResult:
        return self._run_proactive(
            conversation,
            instruction=self.overrides.get("FAREWELL_INSTRUCTION", self.FAREWELL_INSTRUCTION),
            mode="farewell",
            extra_memory_chunks=extra_memory_chunks,
            is_farewell=True,
            user_facts=user_facts,
            self_facts=self_facts,
        )

    def _run_proactive(
        self,
        conversation: Conversation,
        *,
        instruction: str,
        mode: str,
        extra_memory_chunks: list[Chunk] | None,
        is_farewell: bool,
        user_facts: dict | None = None,
        self_facts: dict | None = None,
    ) -> ChatResult:
        """Shared body for proactive() and proactive_farewell().

        Same shape as chat() but injects a synthetic user instruction
        instead of a real user message. The controller is invoked with
        is_proactive=True (and is_farewell=True for the farewell path)
        so the rule layer can fire `proactive_engage` / `proactive_farewell`.
        """
        last_user_text = ""
        for m in reversed(conversation.messages):
            if m.role == "user" and not (m.meta and m.meta.get("system_trigger")):
                last_user_text = m.content or ""
                break

        # 统计「上一条真实用户消息之后」已经发生了几次主动发言。这是**权威计数**
        # （存在会话历史里），不依赖前端内存——前端刷新/不同步都不会让它出错。
        prior_proactive = 0
        for m in reversed(conversation.messages):
            if m.role == "user" and not (m.meta and m.meta.get("system_trigger")):
                break  # 到了最近一条真实用户消息，停止往回数
            if m.role == "assistant" and m.meta and m.meta.get("proactive"):
                prior_proactive += 1

        # 告别由后端按权威计数决定，覆盖调用方传入的 is_farewell：已经主动过
        # (max_nudges - 1) 次，这一次就是第 max_nudges 次 → 告别。这样无论前端
        # 计数怎么漂，都不会出现「主动很多次也不告别」。
        from .config import resolve_proactive_pacing
        max_nudges = int(resolve_proactive_pacing().get("max_nudges", 4) or 4)
        if prior_proactive >= max_nudges - 1:
            is_farewell = True

        plan, plan_trace = self._dispatch_controller(
            last_user_text,
            conversation,
            is_proactive=True,
            is_farewell=is_farewell,
            has_cross_session_memory=bool(extra_memory_chunks),
        )

        # 分级话题来源：第 1 次接最近话题、第 2 次翻更早话题、第 3 次起说莉娜
        # 自己的经历。（告别路径不挑话头。）
        stage = {0: "recent", 1: "earlier"}.get(prior_proactive, "self")

        # B 级（MapDia/PaRT）：engage 路径上，先让 controller 按当前 stage 从历史里
        # 挑一个最值得重提的话头（或 self 级抛莉娜自己的事），驱动这次主动开口。
        topic_hook = ""
        topic_query = ""
        avoid_hooks: list[str] = []
        if (not is_farewell) and self._controller is not None and self._controller.has_llm:
            try:
                # 收集本会话之前主动发言已经抛过的话头，传给挑选器去重，
                # 避免连续几次主动发言内容雷同。
                avoid_hooks = [
                    m.meta["topic_hook"]
                    for m in conversation.messages
                    if m.role == "assistant" and m.meta and m.meta.get("topic_hook")
                ]
                ctx = LinaTurnContext(
                    user_text=last_user_text,
                    history=self._history_pairs_for_controller(conversation),
                    is_proactive=True,
                )
                picked = self._controller.pick_proactive_topic_sync(
                    ctx, avoid_hooks=avoid_hooks, stage=stage, user_facts=user_facts
                )
                topic_hook = (picked or {}).get("topic_hook", "") or ""
                topic_query = (picked or {}).get("query_hint", "") or ""
            except Exception:
                pass  # 挑话头失败不影响主动发言，退回规则的静态 query_hint
        # 主动发言可观测：每次把 stage / 计数 / 选中话头 / 已避免话头打到日志，
        # 方便复盘"为什么又重复了同一个话题"。
        try:
            import logging
            logging.getLogger("lina.controller").info(
                "[proactive] prior=%s stage=%s farewell=%s hook=%r avoid=%r",
                prior_proactive, stage, is_farewell, topic_hook,
                [h[:20] for h in avoid_hooks],
            )
        except Exception:
            pass

        # 挑到话头：① 检索 query 优先用它（更聚焦）；② 指令里点名让莉娜去捡它。
        if topic_hook:
            instruction = (
                f"{instruction}\n"
                f"这次主动开口，就顺着这个话头来——「{topic_hook}」。"
                f"自然地把它重新捡起来，让用户感觉你记着、惦记着；不要生硬地宣布换话题。"
            )

        rag_query = topic_query or plan.query_hint or last_user_text
        retrieve_k = plan.retrieve_k if self._controller is not None else self.retrieve_k
        retrieved_raw = (
            self.rag.retrieve(rag_query, k=retrieve_k) if rag_query and retrieve_k > 0 else []
        )
        retrieved = (
            self._filter_chunks_by_plan(retrieved_raw, plan)
            if self._controller is not None
            else retrieved_raw
        )

        history_window = plan.history_window if self._controller is not None else self.history_window
        history_recall_k = (
            plan.history_recall_k if self._controller is not None else self.history_retrieve_k
        )
        retrieved_history: list[Chunk] = []
        if _HISTORY_RECALL_ON and plan.use_history_recall and history_recall_k > 0 and rag_query:
            retrieved_history = retrieve_history_chunks(
                conversation.messages,
                rag_query,
                k=history_recall_k,
                exclude_recent_count=history_window,
            )
        if extra_memory_chunks and plan.use_cross_session_memory:
            retrieved_history = list(extra_memory_chunks) + retrieved_history

        prior_mood = conversation.last_assistant_meta()
        if not prior_mood:
            prior_mood = {"mood": "平静", "intensity": 5, "trust": 3}

        prior: list[dict] = []
        for m in conversation.messages:
            if m.role == "assistant" and m.meta:
                tag = f"[mood: {m.meta.get('mood', '?')} | {m.meta.get('intensity', 5)} | 信任={m.meta.get('trust', 3)}]"
                prior.append({"role": "assistant", "content": f"{tag}\n{m.content}"})
            else:
                prior.append({"role": m.role, "content": m.content})
        if history_window > 0:
            prior = prior[-history_window:]

        # 注入用户长期记忆，让主动开口时也"记得用户"。清单短→整份注入。
        uf_text = ""
        if user_facts:
            try:
                from .user_facts import UserFactsStore
                total = sum(len(v) for v in user_facts.values() if isinstance(v, list))
                uf_text = (UserFactsStore.render(user_facts) if total <= 40
                           else UserFactsStore.search(user_facts, topic_query or last_user_text, k=10))
            except Exception:
                uf_text = ""
        # 注入莉娜自己的事实清单，让她主动开口时**不和自己以前说过的话冲突**。
        sf_text = ""
        if self_facts:
            try:
                from .self_facts import SelfFactsStore
                sf_text = SelfFactsStore.render(self_facts)
            except Exception:
                sf_text = ""
        composed_instruction = self._build_user_content(
            instruction, retrieved, retrieved_history, prior_mood,
            self_facts_text=sf_text, user_facts_text=uf_text
        )
        api_messages = prior + [{"role": "user", "content": composed_instruction}]

        system_blocks = self._system_blocks + self._build_dynamic_system_blocks(plan, rules_query=instruction)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=api_messages,
        )

        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        raw_reply = "".join(text_parts).strip()
        cleaned_reply, mood = parse_mood_tag(raw_reply)
        # 主动发言本身就是「主动开口一小句」，不该再分段。剥掉并丢弃模型若误带的
        # [segments:…] 标记，避免它露在气泡里。
        cleaned_reply, _ = parse_segments_tag(cleaned_reply)

        conversation.add("user", instruction, meta={"system_trigger": True, "mode": mode})
        meta = dict(mood) if mood else {}
        meta["proactive"] = True
        if is_farewell:
            meta["farewell"] = True
        if topic_hook:
            meta["topic_hook"] = topic_hook   # 记下本次话头，供下次去重
        conversation.add("assistant", cleaned_reply, meta=meta)

        # 主动发言**不喂进记忆清单**：它是为了找话头而即兴抛出的内容（小八卦、随口
        # 联想），不是用户/莉娜确认过的稳定事实。之前把它当 slid_out 提炼入库，会让
        # 这些即兴内容污染记忆、和她真说过的话冲突、并干扰之后的对话。故这里返回空。
        slid_out: list = []

        usage = response.usage
        return ChatResult(
            text=cleaned_reply,
            retrieved=retrieved,
            retrieved_history=retrieved_history,
            mood=mood,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            plan=plan.to_dict(),
            controller_trace=plan_trace,
            slid_out_turns=slid_out,
        )

    # 续说指令：把上一条回复 parse 出来的某个小段要点，自然展开成一小段。
    # {point} 由 continue_segment() 填入。
    CONTINUE_INSTRUCTION = _load_main_prompt("continue_instruction.txt")

    def continue_segment(
        self,
        conversation: Conversation,
        extra_memory_chunks: list[Chunk] | None = None,
    ) -> ChatResult | None:
        """Deliver the next parked segment as a short follow-up message.

        Pops the first outline point from `conversation.pending_segments`,
        asks the model to expand it into one short line, and persists it
        like a proactive turn. Returns None if there's nothing pending
        (the web layer treats None as "stop the continue timer").

        Structurally mirrors _run_proactive but routes through the
        controller with is_continuation=True (→ the `continuation` rule:
        very short, no re-retrieval, tone follows the prior segment).
        """
        pending = list(conversation.pending_segments or [])
        if not pending:
            return None
        point = pending.pop(0)
        # Consume immediately so a crash mid-turn doesn't replay this point.
        conversation.pending_segments = pending or None

        plan, plan_trace = self._dispatch_controller(
            point,
            conversation,
            is_continuation=True,
            has_cross_session_memory=bool(extra_memory_chunks),
        )

        prior_mood = conversation.last_assistant_meta()
        if not prior_mood:
            prior_mood = {"mood": "平静", "intensity": 5, "trust": 3}

        history_window = plan.history_window if self._controller is not None else self.history_window
        prior: list[dict] = []
        for m in conversation.messages:
            if m.role == "assistant" and m.meta:
                tag = f"[mood: {m.meta.get('mood', '?')} | {m.meta.get('intensity', 5)} | 信任={m.meta.get('trust', 3)}]"
                prior.append({"role": "assistant", "content": f"{tag}\n{m.content}"})
            else:
                prior.append({"role": m.role, "content": m.content})
        if history_window > 0:
            prior = prior[-history_window:]

        # Continuation doesn't re-retrieve (the material was fixed last turn),
        # so no static/history chunks — just the state + the instruction.
        continue_tpl = self.overrides.get("CONTINUE_INSTRUCTION", self.CONTINUE_INSTRUCTION)
        instruction = continue_tpl.format(point=point)
        composed = self._build_user_content(instruction, [], [], prior_mood)
        api_messages = prior + [{"role": "user", "content": composed}]

        system_blocks = self._system_blocks + self._build_dynamic_system_blocks(plan, rules_query=instruction)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=api_messages,
        )

        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        raw_reply = "".join(text_parts).strip()
        cleaned_reply, mood = parse_mood_tag(raw_reply)
        # A continuation must not itself spawn more segments — strip & ignore
        # any segment tag it might emit, so we don't recurse indefinitely.
        cleaned_reply, _ = parse_segments_tag(cleaned_reply)

        conversation.add(
            "user", instruction, meta={"system_trigger": True, "mode": "continue"}
        )
        meta = dict(mood) if mood else {}
        # 续说只标 continuation（不是主动找话，不标 proactive）——前端据此显示
        # 「· 接着说」而非「· 主动」。
        meta["continuation"] = True
        conversation.add("assistant", cleaned_reply, meta=meta)

        usage = response.usage
        return ChatResult(
            text=cleaned_reply,
            retrieved=[],
            retrieved_history=[],
            mood=mood,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            plan=plan.to_dict(),
            controller_trace=plan_trace,
            pending_segments=list(conversation.pending_segments or []),
            is_continuation=True,
        )
