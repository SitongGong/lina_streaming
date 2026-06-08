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
    use_history_recall: bool = True
    use_cross_session_memory: bool = True
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

    # mood continuity
    enforce_mood_continuity: bool = True

    # trace (purely informational)
    trace_source: str = "fallback"
    matched_rule: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "use_static_personality", _coerce_bool(self.use_static_personality))
        object.__setattr__(self, "use_static_hobbies", _coerce_bool(self.use_static_hobbies))
        object.__setattr__(self, "use_static_others", _coerce_bool(self.use_static_others))
        object.__setattr__(self, "use_history_recall", _coerce_bool(self.use_history_recall))
        object.__setattr__(
            self, "use_cross_session_memory", _coerce_bool(self.use_cross_session_memory)
        )
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
        object.__setattr__(self, "tone_hint", _normalize_text(self.tone_hint)[:12])
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
        return tuple(sources)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
