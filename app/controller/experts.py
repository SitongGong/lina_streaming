"""Fan-out micro-advisors.

Each advisor judges exactly one field of `LinaPromptPlan`. Splitting the
decision per field gives:
- small, even prompt sizes (good for batch / rate-limit budgeting)
- per-field defaults on timeout (we never block a turn on a slow judge)
- per-field traceability (the controller logs each one separately)

All advisors share three template files (`control_flag` / `control_int` /
`control_text`); they differ only in the `target_desc` / `decision_rules`
strings injected into the template.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ._prompts import load_prompt
from .schema import LinaTurnContext


logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?|```$", re.I | re.M)


def _normalize_json_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = _THINK_RE.sub("", cleaned).strip()
    cleaned = _CODE_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "是"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", ""}:
        return False
    return bool(value)


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Tolerant JSON-object parser.

    LLMs sometimes wrap output in code fences or add trailing prose; we
    strip those and also try to grab the first balanced {...} block as a
    last resort. Uses `json_repair` if available, otherwise falls back to
    the stdlib parser.
    """
    text = _normalize_json_text(raw)
    if not text:
        raise ValueError("empty advisor output")
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    try:
        import json_repair  # type: ignore
    except Exception:
        json_repair = None

    last_error: Exception | None = None
    for cand in candidates:
        try:
            payload = json.loads(cand)
        except json.JSONDecodeError:
            if json_repair is None:
                last_error = ValueError("invalid JSON and json_repair unavailable")
                continue
            try:
                payload = json_repair.loads(cand)
            except Exception as exc:
                last_error = exc
                continue
        if isinstance(payload, dict):
            return payload
        last_error = ValueError(f"advisor output is not an object: {type(payload).__name__}")
    raise last_error or ValueError("failed to parse advisor JSON")


def _render_history(history: tuple[tuple[str, str], ...], limit: int = 3) -> str:
    if not history:
        return "(无历史)"
    visible = history[-limit:]
    lines: list[str] = []
    for idx, (u, a) in enumerate(visible, start=1):
        lines.append(f"{idx}. U: {u[:120]}")
        lines.append(f"   A: {a[:120]}")
    return "\n".join(lines)


@dataclass(frozen=True)
class AdvisorResult:
    name: str
    fields: dict[str, Any] = field(default_factory=dict)
    source: str = "default"
    latency_ms: float = 0.0
    raw_output: str = ""
    error: str = ""


class _AdvisorBase:
    """Common scaffolding shared by all field advisors.

    Subclasses override `_target_desc` / `_decision_rules` / `defaults`
    and `_normalize_fields`. They all call the same OpenAI chat
    completion endpoint with `response_format=json_object`.
    """

    name = "base"
    template_path = ""
    defaults: dict[str, Any] = {}

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        timeout: float = 1.8,
        run_condition: Optional[Callable[[LinaTurnContext], bool]] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = max(0.2, float(timeout or 1.8))
        self._template = load_prompt(self.template_path)
        self._run_condition = run_condition

    def should_run(self, ctx: LinaTurnContext) -> bool:
        if self._run_condition is None:
            return True
        return bool(self._run_condition(ctx))

    async def judge(self, ctx: LinaTurnContext) -> AdvisorResult:
        if not self.should_run(ctx):
            return AdvisorResult(name=self.name, fields=dict(self.defaults), source="skipped")
        if self._client is None or not self._template:
            return AdvisorResult(name=self.name, fields=dict(self.defaults), source="no_client")
        prompt = self._render_prompt(ctx)
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    # GPT-5 系是 reasoning 模型：
                    # - 用 max_completion_tokens 取代 max_tokens；上限要给够，
                    #   因为这个预算同时覆盖「内部推理」和「正式输出」，给太小
                    #   会被推理吃光导致 content 为空（finish_reason=length）。
                    # - reasoning_effort='minimal'：单字段判断不需要深度推理，
                    #   关掉推理既快又省，content 直接出 JSON。
                    # - 不传 temperature：GPT-5 系只接受默认值。
                    max_completion_tokens=512,
                    reasoning_effort="minimal",
                    response_format={"type": "json_object"},
                ),
                timeout=self._timeout,
            )
            raw = (response.choices[0].message.content or "").strip()
            fields = self._normalize_fields(_parse_json_object(raw))
            return AdvisorResult(
                name=self.name,
                fields=fields,
                source="llm",
                latency_ms=(time.monotonic() - started) * 1000,
                raw_output=raw,
            )
        except Exception as exc:  # asyncio.TimeoutError, OpenAI errors, parse errors
            # 用 类型名: 消息 的形式记录——asyncio.TimeoutError 的 str() 是空字符串，
            # 单打 %s 会得到 "failed: " 这种看不出根因的日志。
            detail = f"{type(exc).__name__}: {exc}".rstrip(": ")
            logger.warning("advisor %s failed: %s", self.name, detail)
            return AdvisorResult(
                name=self.name,
                fields=dict(self.defaults),
                source="default_error",
                latency_ms=(time.monotonic() - started) * 1000,
                error=detail,
            )

    def _render_prompt(self, ctx: LinaTurnContext) -> str:
        raise NotImplementedError

    def _normalize_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class BoolAdvisor(_AdvisorBase):
    template_path = "controller/control_flag.txt"

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        name: str,
        field_name: str,
        target_desc: str,
        decision_rules: str,
        default: bool = False,
        timeout: float = 1.8,
        run_condition: Optional[Callable[[LinaTurnContext], bool]] = None,
    ) -> None:
        self.name = name
        self.field_name = field_name
        self.defaults = {field_name: default}
        self._target_desc = target_desc
        self._decision_rules = decision_rules
        super().__init__(client, model=model, timeout=timeout, run_condition=run_condition)

    def _render_prompt(self, ctx: LinaTurnContext) -> str:
        return self._template.format(
            field_name=self.field_name,
            target_desc=self._target_desc,
            decision_rules=self._decision_rules,
            user_text=ctx.user_text or "（无具体用户输入，当前是主动发言）",
            history_text=_render_history(ctx.history),
            is_proactive_flag="1" if ctx.is_proactive else "0",
        )

    def _normalize_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {self.field_name: _coerce_bool(payload.get(self.field_name, self.defaults[self.field_name]))}


class IntAdvisor(_AdvisorBase):
    template_path = "controller/control_int.txt"

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        name: str,
        field_name: str,
        target_desc: str,
        range_desc: str,
        decision_rules: str,
        default: int,
        minimum: int,
        maximum: int,
        timeout: float = 1.8,
        run_condition: Optional[Callable[[LinaTurnContext], bool]] = None,
    ) -> None:
        self.name = name
        self.field_name = field_name
        self.defaults = {field_name: default}
        self._target_desc = target_desc
        self._range_desc = range_desc
        self._decision_rules = decision_rules
        self._default = default
        self._minimum = minimum
        self._maximum = maximum
        super().__init__(client, model=model, timeout=timeout, run_condition=run_condition)

    def _render_prompt(self, ctx: LinaTurnContext) -> str:
        return self._template.format(
            field_name=self.field_name,
            target_desc=self._target_desc,
            range_desc=self._range_desc,
            decision_rules=self._decision_rules,
            user_text=ctx.user_text or "（无具体用户输入，当前是主动发言）",
            history_text=_render_history(ctx.history),
            is_proactive_flag="1" if ctx.is_proactive else "0",
        )

    def _normalize_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            self.field_name: _clamp_int(
                payload.get(self.field_name, self._default),
                default=self._default,
                minimum=self._minimum,
                maximum=self._maximum,
            )
        }


class TextAdvisor(_AdvisorBase):
    template_path = "controller/control_text.txt"

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        name: str,
        field_name: str,
        target_desc: str,
        decision_rules: str,
        default: str = "",
        max_chars: int = 24,
        timeout: float = 1.8,
        run_condition: Optional[Callable[[LinaTurnContext], bool]] = None,
    ) -> None:
        self.name = name
        self.field_name = field_name
        self.defaults = {field_name: default}
        self._target_desc = target_desc
        self._decision_rules = decision_rules
        self._default = default
        self._max_chars = max(1, int(max_chars or 24))
        super().__init__(client, model=model, timeout=timeout, run_condition=run_condition)

    def _render_prompt(self, ctx: LinaTurnContext) -> str:
        return self._template.format(
            field_name=self.field_name,
            target_desc=self._target_desc,
            decision_rules=self._decision_rules,
            user_text=ctx.user_text or "（无具体用户输入，当前是主动发言）",
            history_text=_render_history(ctx.history),
            is_proactive_flag="1" if ctx.is_proactive else "0",
        )

    def _normalize_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get(self.field_name, self._default) or "").strip()
        return {self.field_name: text[: self._max_chars]}


def build_lina_advisors(
    client: Any,
    *,
    model: str = "gpt-5-mini",
    timeout: float = 1.8,
) -> dict[str, _AdvisorBase]:
    """Instantiate the full set of lina advisors (~14).

    Returns a dict keyed by advisor name; `LinaController` iterates over
    `.values()` and runs them concurrently. Order doesn't matter — the
    merge step is field-by-field.
    """

    return {
        # --- style / length ---
        "tone_hint": TextAdvisor(
            client,
            model=model,
            name="tone_hint",
            field_name="tone_hint",
            target_desc=(
                "本轮回复的语气标签。极短词（≤8 字）。"
                "示例：好奇 / 雀跃 / 温柔 / 谨慎 / 沉默 / 不耐 / 莞尔 / 认真 / 疑惑 / 半信半疑。"
                "lina 的情绪连续性很强，宁可保留上一轮 mood，不必硬切。"
                "不需要时输出空字符串。"
            ),
            decision_rules=(
                "- 普通闲聊倾向「自然」或留空让她自由发挥。\n"
                "- 古代语 / 遗物 / 戏剧 / 香草这类兴奋点倾向「雀跃 / 好奇」。\n"
                "- 用户低落 / 哭 / 累 倾向「温柔」。\n"
                "- 用户在试探她或冒犯隐私倾向「警觉 / 谨慎」。\n"
                "- 用户要她做现代事（搜 / 打开链接 / 跑代码）倾向「疑惑」。"
            ),
            default="",
            max_chars=8,
            timeout=timeout,
        ),
        "sentences": IntAdvisor(
            client,
            model=model,
            name="sentences",
            field_name="sentences",
            target_desc="主模型本轮的回复句数（每个「短行」算 1 句）。",
            range_desc="1-10",
            decision_rules=(
                "总原则：宁短勿长。想说的多，靠把内容拆成几小段分多次说，"
                "而不是把一段写长。这里只判断**这一小段**的句数。\n"
                "- 问候 / 短反应 / 告别：1 句。\n"
                "- 普通闲聊：1-2 句。\n"
                "- 用户低落需要安抚：2-3 句。\n"
                "- 关系回访 / 自我介绍 / 知识好奇：2-3 句。\n"
                "- 触到她兴奋点（古代语 / 遗物 / 戏剧 / 香草 / 甜点）：最多 4 句，"
                "再多就该拆段而不是堆长。\n"
                "- 信任度低（≤3）时不要超过 2 句。"
            ),
            default=2,
            minimum=1,
            maximum=10,
            timeout=timeout,
        ),
        "max_reply_chars": IntAdvisor(
            client,
            model=model,
            name="max_reply_chars",
            field_name="max_reply_chars",
            target_desc="主模型本轮这一小段回复的总字数上限（不算 mood 标记行）。",
            range_desc="20-300",
            decision_rules=(
                "总原则：宁短勿长。这是**单段**字数上限，长内容靠切段，不要堆长。\n"
                "- 短反应 / 问候：20-35。\n"
                "- 普通闲聊：35-55。\n"
                "- 安抚 / 关系回访：50-80。\n"
                "- 兴奋点话题：60-100，仍宁短勿长，更长就拆段。\n"
                "- 现代行为请求 / 能力边界拒绝：40-70。"
            ),
            default=45,
            minimum=20,
            maximum=300,
            timeout=timeout,
        ),
        # --- modules ---
        "module_user_vent": BoolAdvisor(
            client,
            model=model,
            name="module_user_vent",
            field_name="module_user_vent",
            target_desc="是否加载「安抚陪伴」模块。",
            decision_rules=(
                "- 用户低落 / 累 / 哭 / 焦虑 / 想被陪 → true。\n"
                "- 普通闲聊 / 知识问答 / 现代请求 → false。"
            ),
            timeout=timeout,
        ),
        "module_action_boundary": BoolAdvisor(
            client,
            model=model,
            name="module_action_boundary",
            field_name="module_action_boundary",
            target_desc="是否加载「1760 年知识边界 / AI 自指禁忌」模块。",
            decision_rules=(
                "- 用户要她跑代码 / 打开链接 / 搜索 / 翻译 / 询问 AI 概念 → true。\n"
                "- 提到现代概念（手机 / 电脑 / 互联网 / AI / 量子物理）→ true。\n"
                "- 普通日常 / 古代世界相关闲聊 → false。"
            ),
            timeout=timeout,
        ),
        "module_world_immersion": BoolAdvisor(
            client,
            model=model,
            name="module_world_immersion",
            field_name="module_world_immersion",
            target_desc="是否加载「lina 兴奋点 · 古代世界沉浸」模块。",
            decision_rules=(
                "- 用户提古代语 / 古文字 / 碑文 / 遗物 / 炼金 / 魔法石 → true。\n"
                "- 用户聊戏剧 / 香草 / 草药 / 甜点 → true。\n"
                "- 用户聊她的日常工作（鉴定遗物、跑实验）→ true。\n"
                "- 现代请求 / 安抚 / 短反应 → false。"
            ),
            timeout=timeout,
        ),
        "module_relationship_recall": BoolAdvisor(
            client,
            model=model,
            name="module_relationship_recall",
            field_name="module_relationship_recall",
            target_desc="是否加载「关系续聊 / 回访」模块。",
            decision_rules=(
                "- 用户问「还记得吗 / 上次 / 之前说过 / 好久不见」→ true。\n"
                "- 用户回顾过去交往 → true。\n"
                "- 全新话题 / 现代请求 / 短反应 → false。"
            ),
            timeout=timeout,
        ),
        "module_self_introspection": BoolAdvisor(
            client,
            model=model,
            name="module_self_introspection",
            field_name="module_self_introspection",
            target_desc="是否加载「自我介绍 / 自我反思」模块。",
            decision_rules=(
                "- 用户问「你是谁 / 你性格 / 你从哪里来 / 介绍一下你自己」→ true。\n"
                "- 用户在套问她的过去 / 隐私 → true（但模块会让她保持谨慎）。\n"
                "- 普通闲聊 / 短反应 → false。"
            ),
            timeout=timeout,
        ),
        # --- hooks ---
        "hook_concrete_example": BoolAdvisor(
            client,
            model=model,
            name="hook_concrete_example",
            field_name="hook_concrete_example",
            target_desc="是否在本轮回复里强制带至少一个具体细节（专名 / 场景 / 物件）。",
            decision_rules=(
                "- 用户问她最近在做什么、喜欢什么、有哪些经历 → true。\n"
                "- 古代语 / 遗物 / 戏剧 / 香草 等兴奋点 → true。\n"
                "- 安抚低落情绪 / 短反应 / 现代请求 → false。"
            ),
            timeout=timeout,
        ),
        "hook_callback": BoolAdvisor(
            client,
            model=model,
            name="hook_callback",
            field_name="hook_callback",
            target_desc="是否在本轮轻轻回勾最近几轮历史中未聊完的话头或新冒出的梗。",
            decision_rules=(
                "- 最近几轮用户抛出一个话题但没展开 → true。\n"
                "- 用户在切换新话题 / 想简短收束 → false。\n"
                "- 历史极短（< 2 轮）/ 首次对话 → false。"
            ),
            timeout=timeout,
            run_condition=lambda ctx: len(ctx.history) >= 2,
        ),
        "hook_history_recall": BoolAdvisor(
            client,
            model=model,
            name="hook_history_recall",
            field_name="hook_history_recall",
            target_desc="是否在本轮主动引用用户之前讲过的事实 / 偏好 / 承诺，让用户感觉被记住。",
            decision_rules=(
                "- 用户在关系回访（还记得 / 上次 / 之前）→ true。\n"
                "- 安抚 / 兴奋点话题，且有跨会话记忆可用 → true。\n"
                "- 没有跨会话记忆且本会话历史很短 → false。\n"
                "- 现代请求 / 纯短反应 → false。"
            ),
            timeout=timeout,
            run_condition=lambda ctx: ctx.has_cross_session_memory or len(ctx.history) >= 4,
        ),
        "use_self_facts": BoolAdvisor(
            client,
            model=model,
            name="use_self_facts",
            field_name="use_self_facts",
            target_desc=(
                "是否检索「莉娜自我事实清单」——她在过往对话里亲口说过的、关于她自己的"
                "稳定事实（养的猫、答应过的事、喜好、经历）。检索出相关几条注入，"
                "帮她对自己说过的话保持前后一致。"
            ),
            decision_rules=(
                "- 用户在问莉娜自己（你养宠物吗 / 你喜欢啥 / 你不是说过 / 你上次提的）→ true。\n"
                "- 需要莉娜谈及自己的经历、偏好、之前的承诺时 → true。\n"
                "- 纯知识问答、现代请求、与莉娜本人无关的闲聊 → false。"
            ),
            timeout=timeout,
        ),
        "allow_doubt_wrap": BoolAdvisor(
            client,
            model=model,
            name="allow_doubt_wrap",
            field_name="allow_doubt_wrap",
            target_desc=(
                "是否允许 lina 用「我也不太确定…让我想想 / 好像是？」这类含糊起头来开本轮回复。"
                "这是 lina 的关键人物特征之一——她是人，不是百科全书。"
            ),
            decision_rules=(
                "- 默认 true：日常闲聊 / 知识好奇 / 自我反思可以带含糊语气。\n"
                "- 用户明显低落 / 崩溃 / 哭，需要她坚定地温柔接住 → false。\n"
                "- 现代请求需要她坦率说不会 → 仍可 true，但模块本身会强制她说「这是某种新发明吗」。\n"
                "- 问候 / 短反应 / 告别 → false。"
            ),
            default=True,
            timeout=timeout,
        ),
        "enforce_mood_continuity": BoolAdvisor(
            client,
            model=model,
            name="enforce_mood_continuity",
            field_name="enforce_mood_continuity",
            target_desc="是否在本轮硬性要求 mood 与上一轮连贯，不允许情绪突变。",
            decision_rules=(
                "- 默认 true：lina 的人设要求情绪连续。\n"
                "- 用户做了明显冒犯 / 极度惊喜的事 → false（允许情绪较大跳动）。\n"
                "- 首次对话 → 用 true 但允许从默认平静起步。"
            ),
            default=True,
            timeout=timeout,
        ),
        # --- retrieval ---
        "query_hint": TextAdvisor(
            client,
            model=model,
            name="query_hint",
            field_name="query_hint",
            target_desc=(
                "给 BM25 静态检索器的短查询提示。优先保留 2-24 字的关键短标签。"
                "例如：性格 / 香草 / 戏剧 / 鉴定 / 童年。"
                "不需要检索时输出空字符串。"
            ),
            decision_rules=(
                "- 用户问她性格 / 喜好 → 「性格」/「喜好」类标签。\n"
                "- 用户问她经历 / 来历 → 「童年」/「修道院」/「成长」类标签。\n"
                "- 兴奋点话题 → 用本轮的关键词（古代语 / 香草 / 戏剧 / 炼金）。\n"
                "- 关系回访 → 短线索（上次说过的事 / 之前的那只猫）。\n"
                "- 短反应 / 问候 / 现代请求 → 空字符串。"
            ),
            default="",
            max_chars=24,
            timeout=timeout,
        ),
        "history_window": IntAdvisor(
            client,
            model=model,
            name="history_window",
            field_name="history_window",
            target_desc="本轮给主模型保留多少轮历史消息（最近 N 轮）。",
            range_desc="0-60",
            decision_rules=(
                "- 问候 / 短反应 / 告别：4-8。\n"
                "- 普通闲聊：12-20。\n"
                "- 安抚 / 自我反思：18-24。\n"
                "- 兴奋点 / 关系回访：24-30。\n"
                "- 现代请求：6-10。"
            ),
            default=24,
            minimum=0,
            maximum=60,
            timeout=timeout,
        ),
    }
