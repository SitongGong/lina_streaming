"""LinaController: top-level per-turn decision dispatcher.

Flow per turn:
1. Rule layer tries to short-circuit with a hand-tuned plan. If it
   matches, we return immediately — zero LLM calls, zero latency.
2. If no rule fires AND an OpenAI client is configured, we fan out to
   all advisors. They run concurrently under a single hard deadline; any
   advisor that times out contributes its default value.
3. If no client is configured OR all advisors fail, we return a
   fallback plan that mirrors lina's pre-controller behavior (so the
   degraded mode is "just like before").

The controller never raises out of `dispatch()` for advisor failures;
the worst case is a fallback plan plus a trace entry. This keeps a slow
or broken controller from breaking the chat path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from copy import deepcopy
from typing import Any, Optional

from .experts import AdvisorResult, _AdvisorBase, build_lina_advisors
from .rule_router import LinaRuleRouter
from .schema import LinaPromptPlan, LinaTurnContext


logger = logging.getLogger(__name__)

# Deadlines tuned for gpt-5-mini over the public OpenAI API. A single
# advisor call measures ~1.0-1.3s in isolation; under 14-way concurrency
# some calls queue and run slower, so we leave generous headroom. The
# per-advisor timeout must stay < the total so a straggler is cancelled
# (and filled with its default) rather than blowing the whole budget.
_CONTROLLER_TIMEOUT = 6.0  # total fan-out deadline (seconds)
_ADVISOR_TIMEOUT = 5.0  # per-advisor timeout (must be < total)
# Self-facts updates run in a BACKGROUND thread (not the latency-sensitive
# per-turn path) and regenerate the full facts JSON (up to 800 tokens), which
# grows as facts accumulate. They need a much more generous deadline than the
# per-turn advisors — especially on Claude, which is slower per output token
# than gpt-5-mini-minimal. Override via LINA_SELF_FACTS_TIMEOUT.
_SELF_FACTS_TIMEOUT = 30.0
DEFAULT_CONTROLLER_MODEL = "gpt-5-mini"


def _resolve_timeouts() -> tuple[float, float]:
    """Resolve (total, per-advisor) deadlines, honoring an env override.

    `LINA_CONTROLLER_TIMEOUT=<seconds>` sets the TOTAL fan-out deadline;
    the per-advisor timeout is derived as ~83% of it (kept below total so
    a straggler is cancelled rather than blowing the whole budget).

    The main use is debugging: set it to e.g. 600 so stepping through
    breakpoints in dispatch()/judge() doesn't trip the deadline. Unset =
    the production defaults (6.0 / 5.0).
    """
    raw = os.environ.get("LINA_CONTROLLER_TIMEOUT")
    if not raw or not raw.strip():
        return _CONTROLLER_TIMEOUT, _ADVISOR_TIMEOUT
    try:
        total = float(raw.strip())
    except ValueError:
        logger.warning("invalid LINA_CONTROLLER_TIMEOUT=%r; using defaults", raw)
        return _CONTROLLER_TIMEOUT, _ADVISOR_TIMEOUT
    total = max(0.5, total)
    advisor = max(0.2, total * 0.83)
    return total, advisor


class LinaController:
    """Per-turn decision dispatcher."""

    def __init__(
        self,
        *,
        openai_client: Optional[Any] = None,
        model_name: str = DEFAULT_CONTROLLER_MODEL,
        timeout: float = _CONTROLLER_TIMEOUT,
        advisor_timeout: float = _ADVISOR_TIMEOUT,
    ) -> None:
        self._client = openai_client
        self._model_name = model_name
        self._timeout = max(0.5, float(timeout or _CONTROLLER_TIMEOUT))
        self._advisor_timeout = max(0.2, float(advisor_timeout or _ADVISOR_TIMEOUT))
        self._rule_router = LinaRuleRouter()
        self._advisors: dict[str, _AdvisorBase] = (
            build_lina_advisors(self._client, model=self._model_name, timeout=self._advisor_timeout)
            if self._client is not None
            else {}
        )
        self._last_plan: Optional[LinaPromptPlan] = None
        self._last_trace: Optional[dict[str, Any]] = None

    @property
    def has_llm(self) -> bool:
        return self._client is not None and bool(self._advisors)

    @property
    def last_plan(self) -> Optional[LinaPromptPlan]:
        return self._last_plan

    @property
    def last_trace(self) -> Optional[dict[str, Any]]:
        return deepcopy(self._last_trace) if self._last_trace else None

    def dispatch_sync(self, ctx: LinaTurnContext) -> LinaPromptPlan:
        """Sync wrapper. Each call uses a fresh event loop so we never
        clash with a host that already has one running (Flask, anyio)."""
        return asyncio.run(self.dispatch(ctx))

    def update_self_facts_sync(
        self, current_facts: dict, sliding_turns: list[tuple[str, str]]
    ) -> dict | None:
        """Sync wrapper for update_self_facts."""
        return asyncio.run(self.update_self_facts(current_facts, sliding_turns))

    async def update_self_facts(
        self, current_facts: dict, sliding_turns: list[tuple[str, str]]
    ) -> dict | None:
        """用 gpt-5-mini 把「即将滑出窗口的几轮对话」里莉娜的自我陈述，概括/合并
        进现有自我事实清单。返回更新后的分桶 dict；无 client/出错/空 → 返回 None
        （调用方保持旧清单不变）。

        只概括「快被遗忘的那部分」，不碰窗口内原文，所以不和历史上下文重复。
        """
        if self._client is None or not sliding_turns:
            return None
        import json as _json

        from ._prompts import load_prompt
        from .experts import _parse_json_object

        template = load_prompt("controller/self_facts.txt")
        if not template:
            return None
        # 明确标注说话人，避免提炼模型把「用户说的」误记成「莉娜说的」。
        # 每个 turn 是 (user_text, assistant_text)；空串表示该侧没说话（如主动发言）。
        lines: list[str] = []
        for u, a in sliding_turns:
            if (u or "").strip():
                lines.append(f"用户说：{str(u).strip()[:160]}")
            if (a or "").strip():
                lines.append(f"莉娜说：{str(a).strip()[:160]}")
        sliding_text = "\n".join(lines) if lines else "(无)"
        prompt = template.format(
            sliding_text=sliding_text,
            current_facts=_json.dumps(current_facts or {}, ensure_ascii=False),
        )
        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=800,
                    reasoning_effort="minimal",
                    response_format={"type": "json_object"},
                ),
                # Background task → generous deadline, not the per-turn 5s budget.
                timeout=float(os.environ.get("LINA_SELF_FACTS_TIMEOUT") or _SELF_FACTS_TIMEOUT),
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_object(raw)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning(
                "update_self_facts failed: %s", f"{type(exc).__name__}: {exc}".rstrip(": ")
            )
            return None

    def pick_proactive_topic_sync(
        self, ctx: LinaTurnContext, avoid_hooks: list[str] | None = None, stage: str = "recent"
    ) -> dict[str, Any]:
        """Sync wrapper for pick_proactive_topic."""
        return asyncio.run(self.pick_proactive_topic(ctx, avoid_hooks=avoid_hooks, stage=stage))

    async def pick_proactive_topic(
        self, ctx: LinaTurnContext, avoid_hooks: list[str] | None = None, stage: str = "recent"
    ) -> dict[str, Any]:
        """Pick ONE past thread worth resurfacing for a proactive opener.

        Distilled from MapDia (#9, learned topic-retrieval) + PaRT (#4,
        user-anchored topic generation): instead of a hand-written
        query_hint, ask gpt-5-mini to read the recent history and choose
        the single most re-engaging thread, preferring user-related and
        unfinished ones. Runs ONLY on the proactive path (low frequency),
        so the extra LLM call is affordable.

        Returns {topic_hook, query_hint, user_related}. On no client /
        error / empty pick, returns empty strings so callers fall back to
        the rule's static query_hint.
        """
        empty = {"topic_hook": "", "query_hint": "", "user_related": False}
        if self._client is None:
            return empty
        from ._prompts import load_prompt
        from .experts import _parse_json_object, _render_history

        template = load_prompt("controller/proactive_topic.txt")
        if not template:
            return empty
        avoid = [h for h in (avoid_hooks or []) if h]
        avoid_text = (
            "（已经主动抛过下面这些话头，这次必须换一个，不要重复）：\n"
            + "\n".join(f"- {h}" for h in avoid)
        ) if avoid else "（暂无，自由选择）"
        # 分级策略：随主动次数升级，从"最近话题"→"更早话题"→"莉娜自己的经历"。
        stage_text = {
            "recent": (
                "本次策略【接最近话题】：挑你们**最近一两轮**里提到、但还没聊透的话头，"
                "顺着它自然往下问。"
            ),
            "earlier": (
                "本次策略【翻更早的话题】：最近的话头对方没接，这次**跳过最近几轮**，"
                "从**更早**的对话里挑一个用户提过、还算有意思的话题重新捡起来。"
            ),
            "self": (
                "本次策略【说你自己的事】：用户对共同话题似乎没兴趣了。这次**不挑用户的话题**，"
                "改成你（莉娜）主动抛一件**自己的**经历/见闻/小八卦（炼金、遗物、戏剧、香草这类），"
                "结合你的人设自由发挥，留个钩子等对方接。topic_hook 写你要讲的那件事。"
            ),
        }.get(stage, "")
        prompt = template.format(
            history_text=_render_history(ctx.history, limit=8),
            avoid_text=avoid_text,
            stage_text=stage_text,
        )
        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=512,
                    reasoning_effort="minimal",
                    response_format={"type": "json_object"},
                ),
                timeout=self._advisor_timeout,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = _parse_json_object(raw)
            return {
                "topic_hook": str(data.get("topic_hook", "") or "").strip()[:40],
                "query_hint": str(data.get("query_hint", "") or "").strip()[:24],
                "user_related": bool(data.get("user_related", False)),
            }
        except Exception as exc:
            logger.warning("pick_proactive_topic failed: %s", f"{type(exc).__name__}: {exc}".rstrip(": "))
            return empty

    async def dispatch(self, ctx: LinaTurnContext) -> LinaPromptPlan:
        # 1) Rule layer (free, deterministic).
        rule_plan = self._rule_router.route(ctx)
        if rule_plan is not None:
            self._last_plan = rule_plan
            self._last_trace = {
                "source": "rule",
                "matched_rule": rule_plan.matched_rule,
                "plan": rule_plan.to_dict(),
            }
            return rule_plan

        # 2) No LLM available → fallback.
        if not self.has_llm:
            plan = self._fallback_plan(ctx, reason="no_llm")
            self._last_plan = plan
            self._last_trace = {"source": "fallback", "reason": "no_llm", "plan": plan.to_dict()}
            return plan

        # 3) Fan-out advisors.
        started = time.monotonic()
        results = await self._run_advisors(ctx)
        plan = self._merge(ctx, results)
        elapsed_ms = (time.monotonic() - started) * 1000
        self._last_plan = plan
        self._last_trace = {
            "source": "llm",
            "latency_ms": round(elapsed_ms, 1),
            "plan": plan.to_dict(),
            "advisors": {
                name: {
                    "fields": result.fields,
                    "source": result.source,
                    "latency_ms": round(result.latency_ms, 1),
                    "error": result.error,
                }
                for name, result in results.items()
            },
        }
        return plan

    async def _run_advisors(self, ctx: LinaTurnContext) -> dict[str, AdvisorResult]:
        tasks = {
            name: asyncio.create_task(advisor.judge(ctx)) for name, advisor in self._advisors.items()
        }
        results: dict[str, AdvisorResult] = {}
        if not tasks:
            return results
        deadline_at = time.monotonic() + self._timeout
        pending = dict(tasks)
        while pending:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                break
            done, _ = await asyncio.wait(
                pending.values(),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break
            for name, task in list(pending.items()):
                if task not in done:
                    continue
                pending.pop(name)
                try:
                    results[name] = task.result()
                except Exception as exc:
                    logger.warning("advisor %s task raised: %s", name, exc)

        for name, task in pending.items():
            task.cancel()
            defaults = getattr(self._advisors[name], "defaults", {})
            results[name] = AdvisorResult(
                name=name,
                fields=dict(defaults),
                source="deadline_default",
                error=f"controller_deadline>{self._timeout:.1f}s",
            )
        return results

    def _merge(self, ctx: LinaTurnContext, results: dict[str, AdvisorResult]) -> LinaPromptPlan:
        merged: dict[str, Any] = {}
        for result in results.values():
            merged.update(dict(result.fields or {}))

        plan = LinaPromptPlan(
            # retrieval defaults stay generous when controller didn't speak to them
            use_static_personality=True,
            use_static_hobbies=True,
            use_static_others=True,
            use_history_recall=True,
            use_cross_session_memory=ctx.has_cross_session_memory,
            use_self_facts=merged.get("use_self_facts", False),
            query_hint=merged.get("query_hint", ""),
            retrieve_k=4,
            history_recall_k=3,
            history_window=merged.get("history_window", 24),
            module_user_vent=merged.get("module_user_vent", False),
            module_action_boundary=merged.get("module_action_boundary", False),
            module_world_immersion=merged.get("module_world_immersion", False),
            module_relationship_recall=merged.get("module_relationship_recall", False),
            module_self_introspection=merged.get("module_self_introspection", False),
            hook_concrete_example=merged.get("hook_concrete_example", False),
            hook_callback=merged.get("hook_callback", False),
            hook_history_recall=merged.get("hook_history_recall", False),
            allow_doubt_wrap=merged.get("allow_doubt_wrap", True),
            sentences=merged.get("sentences", 2),
            max_reply_chars=merged.get("max_reply_chars", 45),
            allow_segment=merged.get("allow_segment", False),
            max_segments=merged.get("max_segments", 3),
            tone_hint=merged.get("tone_hint", ""),
            enforce_mood_continuity=merged.get("enforce_mood_continuity", True),
            trace_source="llm",
            matched_rule="",
        )

        # Proactive path always stays short, regardless of what the advisors said.
        if ctx.is_proactive:
            return LinaPromptPlan(
                **{
                    **plan.to_dict(),
                    "sentences": min(plan.sentences, 3),
                    "max_reply_chars": min(plan.max_reply_chars, 80),
                    "trace_source": "llm",
                }
            )
        return plan

    @staticmethod
    def _fallback_plan(ctx: LinaTurnContext, *, reason: str) -> LinaPromptPlan:
        # Mirror the pre-controller defaults so the no-LLM path is
        # behaviorally identical to the old engine.
        return LinaPromptPlan(
            use_static_personality=True,
            use_static_hobbies=True,
            use_static_others=True,
            use_history_recall=True,
            use_cross_session_memory=ctx.has_cross_session_memory,
            query_hint="",
            retrieve_k=4,
            history_recall_k=3,
            history_window=30,
            sentences=2,
            max_reply_chars=45,
            tone_hint="",
            enforce_mood_continuity=True,
            allow_doubt_wrap=True,
            trace_source="fallback",
            matched_rule=reason,
        )


class _AnthropicCompatCompletions:
    """Adapt AsyncOpenAI's `chat.completions.create` for Anthropic's
    OpenAI-compatible endpoint by rewriting the GPT-5-only kwargs the
    controller passes:
      - drop `reasoning_effort` (OpenAI reasoning models only),
      - map `max_completion_tokens` → `max_tokens` (Anthropic requires the latter),
      - drop `response_format` (the compat endpoint doesn't enforce JSON mode;
        prompts already demand JSON and `_parse_json_object` is tolerant).
    Everything else passes through untouched.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def create(self, **kwargs: Any):
        kwargs.pop("reasoning_effort", None)
        if "max_completion_tokens" in kwargs:
            kwargs.setdefault("max_tokens", kwargs.pop("max_completion_tokens"))
        kwargs.pop("response_format", None)
        return await self._inner.create(**kwargs)


class _AnthropicCompatChat:
    def __init__(self, inner: Any) -> None:
        self.completions = _AnthropicCompatCompletions(inner.completions)


class _AnthropicCompatClient:
    """Minimal shim over an AsyncOpenAI client exposing only the slice the
    controller uses (`client.chat.completions.create`), adapted for Anthropic."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.chat = _AnthropicCompatChat(inner.chat)


def build_default_controller(
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_CONTROLLER_MODEL,
    base_url: Optional[str] = None,
    provider: str = "openai",
) -> LinaController:
    """Construct a controller wired to OpenAI (or any OpenAI-compatible base_url).

    `provider="anthropic"` targets Anthropic's OpenAI-compatible endpoint: the
    key is the Anthropic key (no fallback to OPENAI_API_KEY), the base_url
    defaults to Anthropic's, and the client is wrapped so the GPT-5-only kwargs
    are normalized for Claude.

    Returns a controller with no LLM client (rule-layer + fallback only)
    when no key is available — *not* an error. lina should still chat fine
    without a controller key.

    A flag `LINA_CONTROLLER=off` disables LLM advisors entirely; the
    rule layer still fires so cheap scenarios stay tight.

    `LINA_CONTROLLER_TIMEOUT=<seconds>` overrides the fan-out deadline
    (useful when debugging — set it large so breakpoints don't trip it).
    """
    total_timeout, advisor_timeout = _resolve_timeouts()
    enabled = (os.environ.get("LINA_CONTROLLER") or "on").strip().lower() != "off"
    provider = (provider or "openai").strip().lower()
    if provider == "anthropic":
        resolved_key = api_key or ""   # 不回退到 OPENAI_API_KEY
        base_url = base_url or "https://api.anthropic.com/v1/"
    else:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
    if not enabled or not resolved_key:
        return LinaController(
            openai_client=None, model_name=model,
            timeout=total_timeout, advisor_timeout=advisor_timeout,
        )
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as exc:
        logger.warning("openai SDK not installed (%s); controller falls back to rules-only", exc)
        return LinaController(
            openai_client=None, model_name=model,
            timeout=total_timeout, advisor_timeout=advisor_timeout,
        )
    client_kwargs: dict[str, Any] = {"api_key": resolved_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    try:
        client = AsyncOpenAI(**client_kwargs)
    except Exception as exc:
        logger.warning("failed to construct AsyncOpenAI client: %s", exc)
        return LinaController(
            openai_client=None, model_name=model,
            timeout=total_timeout, advisor_timeout=advisor_timeout,
        )
    if provider == "anthropic":
        client = _AnthropicCompatClient(client)
    return LinaController(
        openai_client=client, model_name=model,
        timeout=total_timeout, advisor_timeout=advisor_timeout,
    )
