"""Controller data structures.

`LinaTurnContext` is the read-only input to controller decisions.
`LinaPromptPlan` is the structured output every downstream component reads.

Field set is tailored to lina (西比莉娜), not Mio:
- No tsundere/punchline/EverMemOS fields.
- Length budgets are wider (sentences 1-10, chars 20-300) because lina's
  excitement topics (古代语 / 遗物 / 戏剧 / 香草) explicitly allow 5-10
  short lines, per her behavior rules.
- Memory sources mirror lina's actual RAG layout: per-file static
  (personality/hobbies/others), per-session history recall, and
  cross-session user memory.
- A `mood_continuity` flag exists because lina has explicit
  mood/trust persistence rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_VALID_MODULES = frozenset({
    "plain_greeting",
    "short_reaction",
    "user_vent",
    "action_boundary",
    "world_immersion",
    "relationship_recall",
    "self_introspection",
    "welcome_back",
    "continuation",
    "hook_concrete_example",
    "hook_callback",
    "hook_history_recall",
})


# few-shot 示例库白名单：Plan.fewshot_tags 只接受这些 tag，对应
# prompts/controller/fewshot/<tag>.txt。加新场景就往这里加一个 tag + 一个文件。
_VALID_FEWSHOT = frozenset({
    "typo_tolerance",        # 善意理解错别字
    "no_trailing_question",  # 别连环甩问号
    "positive_response",     # 报喜要共情、别浇冷水
    "comfort",               # 低落时先接住情绪、别说教
    "modern_boundary",       # 现代请求茫然以对、别出戏
})
# 一轮最多注入几组示例，避免撑爆 token。放到 2 易把场景库挤掉（场景库 +
# 通用库 no_trailing_question/typo_tolerance 常一起触发），放宽到 3。
_MAX_FEWSHOT_TAGS = 3


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _unique_keep_order(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = _normalize_text(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _normalize_text(value).lower()
    if text in {"1", "true", "yes", "y", "on", "是"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", ""}:
        return False
    return bool(value)


@dataclass(frozen=True)
class LinaTurnContext:
    """All signals available when deciding how to handle this turn.

    `history` is a tuple of (user_text, assistant_text) pairs in chronological
    order; the most recent pair is last. Pairs with missing assistant text
    pass an empty string for the second slot.
    """

    user_text: str
    history: tuple[tuple[str, str], ...] = ()
    user_id: str = ""
    session_id: str = ""
    is_proactive: bool = False
    is_farewell: bool = False
    is_first_turn: bool = False
    has_cross_session_memory: bool = False
    prior_trust: int = 3
    # 这一轮是不是「续说」——把上一条回复没说完的小段接着说。
    is_continuation: bool = False
    # 距离上一条消息过去了多少秒（用于 welcome_back 判断）。
    # 0 表示未知 / 首轮 / 紧接着的连续对话。
    gap_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_text", _normalize_text(self.user_text))
        object.__setattr__(self, "user_id", _normalize_text(self.user_id))
        object.__setattr__(self, "session_id", _normalize_text(self.session_id))
        object.__setattr__(self, "is_proactive", _coerce_bool(self.is_proactive))
        object.__setattr__(self, "is_farewell", _coerce_bool(self.is_farewell))
        object.__setattr__(self, "is_first_turn", _coerce_bool(self.is_first_turn))
        object.__setattr__(
            self, "has_cross_session_memory", _coerce_bool(self.has_cross_session_memory)
        )
        object.__setattr__(
            self, "prior_trust", _clamp_int(self.prior_trust, default=3, minimum=1, maximum=10)
        )
        object.__setattr__(self, "is_continuation", _coerce_bool(self.is_continuation))
        try:
            object.__setattr__(self, "gap_seconds", max(0.0, float(self.gap_seconds)))
        except (TypeError, ValueError):
            object.__setattr__(self, "gap_seconds", 0.0)
        normalized: list[tuple[str, str]] = []
        for turn in self.history or ():
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            normalized.append((_normalize_text(turn[0]), _normalize_text(turn[1])))
        object.__setattr__(self, "history", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LinaPromptPlan:
    """Structured per-turn plan consumed by RAG layer and prompt composer.

    Field groups:
    - Retrieval/memory: which sources to query, what query hint to use,
      how many chunks to return, how much history window to keep.
    - Modules: which differentiated module blocks (comfort / world
      immersion / etc.) to inject into the dynamic system tail.
    - Style: sentences, char cap, tone hint.
    - Hooks: optional content shaping (concrete example, callback,
      history recall, doubt wrap).
    - Mood: whether to enforce strict continuity with last turn's mood.
    - Trace: source + matched rule, used purely for observability.
    """

    # retrieval / memory
    use_static_personality: bool = True
    use_static_hobbies: bool = True
    use_static_others: bool = True
    # world.md（世界观设定）/ sample_conversations.md（说话范例）现在按场景检索，
    # 不再永远全量。默认开（保守：不确定时仍给），由规则/顾问按场景关掉省 token。
    use_world: bool = True
    use_sample_conversations: bool = True
    use_history_recall: bool = True
    use_cross_session_memory: bool = True
    # 是否检索「莉娜自我事实清单」（她亲口说过的关于自己的事）。默认不查——
    # 只在用户问及莉娜自身、或需要保持自我一致时才开，省 token、避免无关注入。
    use_self_facts: bool = False
    query_hint: str = ""
    retrieve_k: int = 4
    history_recall_k: int = 3
    history_window: int = 30

    # modules
    module_user_vent: bool = False
    module_action_boundary: bool = False
    module_world_immersion: bool = False
    module_relationship_recall: bool = False
    module_self_introspection: bool = False
    module_welcome_back: bool = False
    module_continuation: bool = False

    # hooks
    hook_concrete_example: bool = False
    hook_callback: bool = False
    hook_history_recall: bool = False
    allow_doubt_wrap: bool = True

    # style / length
    # 默认偏短：普通闲聊应该是"一两口气"的短回复。长内容靠切段分多次说，
    # 而不是堆在一段里。需要展开的场景（兴奋点/安抚/关系回访）由对应规则
    # 显式调高，不依赖这个默认。
    sentences: int = 2
    max_reply_chars: int = 45
    tone_hint: str = ""

    # 切段预算（由 controller 给，主模型在预算内自己决定实际切几段）：
    # allow_segment=本轮准不准把回复拆成多小段；max_segments=最多拆几段（含已发的第一段）。
    # 短反应/问候/告别/能力边界等场景关掉，避免硬凑分段；可展开的场景才开。
    allow_segment: bool = True
    max_segments: int = 3

    # 行为微调（controller 按场景动态注入到本轮约束块，不改主模型 prompt）：
    # suppress_trailing_question=本轮抑制"句尾强行甩问号"的习惯（主模型人设里
    #   "追问/问句多"指令太强，导致每条结尾都硬问；开了就压一压）。
    # lenient_typos=本轮善意理解用户的错别字/笔误，按最合理意思接住，不揪着错字
    #   反复追问纠错（主模型 prompt 缺这条容错指令）。
    suppress_trailing_question: bool = False
    lenient_typos: bool = False
    # 用户本轮在报喜/表达好转——开启则带出 positive_response 示例（替对方高兴、别浇冷水）。
    user_positive: bool = False
    # 本轮要注入哪些 few-shot 示例（按 tag，对应 fewshot/<tag>.txt）。由 suppress/
    # lenient 等开关自动带出，也可由规则/顾问直接指定。post_init 去重+白名单+截断。
    fewshot_tags: tuple[str, ...] = ()

    # mood continuity
    enforce_mood_continuity: bool = True

    # trace (purely informational)
    trace_source: str = "fallback"
    matched_rule: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "use_static_personality", _coerce_bool(self.use_static_personality))
        object.__setattr__(self, "use_static_hobbies", _coerce_bool(self.use_static_hobbies))
        object.__setattr__(self, "use_static_others", _coerce_bool(self.use_static_others))
        object.__setattr__(self, "use_world", _coerce_bool(self.use_world))
        object.__setattr__(self, "use_sample_conversations", _coerce_bool(self.use_sample_conversations))
        object.__setattr__(self, "use_history_recall", _coerce_bool(self.use_history_recall))
        object.__setattr__(
            self, "use_cross_session_memory", _coerce_bool(self.use_cross_session_memory)
        )
        object.__setattr__(self, "use_self_facts", _coerce_bool(self.use_self_facts))
        object.__setattr__(self, "query_hint", _normalize_text(self.query_hint)[:32])
        object.__setattr__(self, "retrieve_k", _clamp_int(self.retrieve_k, 4, 0, 8))
        object.__setattr__(self, "history_recall_k", _clamp_int(self.history_recall_k, 3, 0, 8))
        object.__setattr__(self, "history_window", _clamp_int(self.history_window, 30, 0, 60))
        object.__setattr__(self, "module_user_vent", _coerce_bool(self.module_user_vent))
        object.__setattr__(self, "module_action_boundary", _coerce_bool(self.module_action_boundary))
        object.__setattr__(self, "module_world_immersion", _coerce_bool(self.module_world_immersion))
        object.__setattr__(
            self, "module_relationship_recall", _coerce_bool(self.module_relationship_recall)
        )
        object.__setattr__(
            self, "module_self_introspection", _coerce_bool(self.module_self_introspection)
        )
        object.__setattr__(self, "module_welcome_back", _coerce_bool(self.module_welcome_back))
        object.__setattr__(self, "module_continuation", _coerce_bool(self.module_continuation))
        object.__setattr__(self, "hook_concrete_example", _coerce_bool(self.hook_concrete_example))
        object.__setattr__(self, "hook_callback", _coerce_bool(self.hook_callback))
        object.__setattr__(self, "hook_history_recall", _coerce_bool(self.hook_history_recall))
        object.__setattr__(self, "allow_doubt_wrap", _coerce_bool(self.allow_doubt_wrap))
        object.__setattr__(self, "sentences", _clamp_int(self.sentences, 2, 1, 10))
        object.__setattr__(self, "max_reply_chars", _clamp_int(self.max_reply_chars, 45, 20, 300))
        object.__setattr__(self, "allow_segment", _coerce_bool(self.allow_segment))
        object.__setattr__(self, "max_segments", _clamp_int(self.max_segments, 3, 1, 3))
        object.__setattr__(self, "tone_hint", _normalize_text(self.tone_hint)[:12])
        object.__setattr__(self, "suppress_trailing_question", _coerce_bool(self.suppress_trailing_question))
        object.__setattr__(self, "lenient_typos", _coerce_bool(self.lenient_typos))
        object.__setattr__(self, "user_positive", _coerce_bool(self.user_positive))
        tags = tuple(t for t in _unique_keep_order(self.fewshot_tags) if t in _VALID_FEWSHOT)
        object.__setattr__(self, "fewshot_tags", tags[:_MAX_FEWSHOT_TAGS])
        object.__setattr__(self, "enforce_mood_continuity", _coerce_bool(self.enforce_mood_continuity))
        object.__setattr__(self, "trace_source", _normalize_text(self.trace_source) or "fallback")
        object.__setattr__(self, "matched_rule", _normalize_text(self.matched_rule))

    @property
    def module_names(self) -> tuple[str, ...]:
        """Ordered list of module ids to inject for this turn."""
        names: list[str] = []
        if self.module_user_vent:
            names.append("user_vent")
        if self.module_action_boundary:
            names.append("action_boundary")
        if self.module_world_immersion:
            names.append("world_immersion")
        if self.module_relationship_recall:
            names.append("relationship_recall")
        if self.module_self_introspection:
            names.append("self_introspection")
        if self.module_welcome_back:
            names.append("welcome_back")
        if self.module_continuation:
            names.append("continuation")
        if self.hook_concrete_example:
            names.append("hook_concrete_example")
        if self.hook_callback:
            names.append("hook_callback")
        if self.hook_history_recall:
            names.append("hook_history_recall")
        return tuple(n for n in names if n in _VALID_MODULES)

    @property
    def static_sources(self) -> tuple[str, ...]:
        """Which static .md files RAG should be allowed to draw from."""
        sources: list[str] = []
        if self.use_static_personality:
            sources.append("personality.md")
        if self.use_static_hobbies:
            sources.append("hobbies.md")
        if self.use_static_others:
            sources.append("others.md")
        if self.use_world:
            sources.append("world.md")
        if self.use_sample_conversations:
            sources.append("sample_conversations.md")
        return tuple(sources)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
