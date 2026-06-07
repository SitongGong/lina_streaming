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

    def pick_proactive_topic_sync(self, ctx: LinaTurnContext) -> dict[str, Any]:
        """Sync wrapper for pick_proactive_topic."""
        return asyncio.run(self.pick_proactive_topic(ctx))

    async def pick_proactive_topic(self, ctx: LinaTurnContext) -> dict[str, Any]:
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
        prompt = template.format(history_text=_render_history(ctx.history, limit=6))
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


def build_default_controller(
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_CONTROLLER_MODEL,
    base_url: Optional[str] = None,
) -> LinaController:
    """Construct a controller wired to OpenAI (or any compatible base_url).

    Returns a controller with no LLM client (rule-layer + fallback only)
    when `api_key` is empty and `OPENAI_API_KEY` is not set — *not* an
    error. lina should still chat fine without a controller key.

    A flag `LINA_CONTROLLER=off` disables LLM advisors entirely; the
    rule layer still fires so cheap scenarios stay tight.

    `LINA_CONTROLLER_TIMEOUT=<seconds>` overrides the fan-out deadline
    (useful when debugging — set it large so breakpoints don't trip it).
    """
    total_timeout, advisor_timeout = _resolve_timeouts()
    enabled = (os.environ.get("LINA_CONTROLLER") or "on").strip().lower() != "off"
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
    return LinaController(
        openai_client=client, model_name=model,
        timeout=total_timeout, advisor_timeout=advisor_timeout,
    )
