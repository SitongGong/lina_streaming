"""合并版 controller 入口（实验，与 LinaController 并存，不改它）。

复用 LinaController 的规则层 + _merge + fallback；只把"LLM 路径的 20 路并发
fan-out"换成"4 组合并请求"（merged_experts.run_merged_advisors）。

用于和原版对比速度/一致性，不接入生产。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .controller import LinaController
from .merged_experts import run_merged_advisors
from .schema import LinaPromptPlan, LinaTurnContext


class MergedController(LinaController):
    """只重写 LLM 路径：规则没命中时走 4 组合并，而非 20 路并发。"""

    def dispatch_sync(self, ctx: LinaTurnContext) -> LinaPromptPlan:
        return asyncio.run(self.dispatch_merged(ctx))

    async def dispatch_merged(self, ctx: LinaTurnContext) -> LinaPromptPlan:
        # 1) 规则层（和原版完全一致）
        rule_plan = self._rule_router.route(ctx)
        if rule_plan is not None:
            self._last_plan = rule_plan
            return rule_plan
        # 2) 无 LLM → fallback
        if not self.has_llm:
            return self._fallback_plan(ctx, reason="no_llm")
        # 3) 4 组合并请求（替代 20 路并发）
        started = time.monotonic()
        merged, group_trace = await run_merged_advisors(
            self._client, self._model_name, self._advisor_timeout, ctx
        )
        # 复用原版 _merge：它从 results.values() 收集 fields。这里直接构造一个
        # 伪 result 让 _merge 拿到 merged。简单起见，直接复刻 _merge 的读法：
        plan = self._merge_from_dict(ctx, merged)
        self._last_plan = plan
        self._last_trace = {
            "source": "llm-merged",
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
            "groups": group_trace,
            "plan": plan.to_dict(),
        }
        return plan

    def _merge_from_dict(self, ctx: LinaTurnContext, merged: dict[str, Any]) -> LinaPromptPlan:
        """复用父类 _merge 的逻辑：父类 _merge(ctx, results) 内部是
        `for r in results.values(): merged.update(r.fields)`。这里我们已有 merged，
        包一个最小 result 对象传进去即可，零重复逻辑。"""
        class _R:
            def __init__(self, fields):
                self.fields = fields
        return self._merge(ctx, {"merged": _R(merged)})
