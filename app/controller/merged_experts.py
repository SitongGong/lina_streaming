"""合并版微顾问（实验，不动 experts.py 的 20 路并发）。

把 20 个单字段 advisor 按"判断逻辑相近"合并成 4 组，每组**一次 LLM 调用**问出
该组所有字段 → 每轮 4 个并发请求（而非 20 个）。目的：砍掉"撞上慢请求"的长尾、
减少网络往返，给 controller 提速。

设计要点：
- 字段的判定文字（target_desc/decision_rules/default/类型/范围）**直接从现有
  advisor 对象上取**，不重写规则——保证和并发版判一样的标准。
- 输出多字段 JSON，复用 _parse_json_object。
- 返回的 `merged` 字段 dict 形状与并发版一致，可直接喂给 LinaController._merge。

用法见 merged_controller.py（实验入口，与 LinaController 并存）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .experts import build_lina_advisors, _parse_json_object, _render_history
from .schema import LinaTurnContext


# 4 组：每组列出它包含的字段名（必须是 build_lina_advisors 里的 advisor 名）。
GROUPS: dict[str, list[str]] = {
    "scene": [
        "module_user_vent", "module_action_boundary", "module_world_immersion",
        "module_relationship_recall", "module_self_introspection",
    ],
    "style": ["tone_hint", "sentences", "max_reply_chars", "allow_segment"],
    "behavior": [
        "suppress_trailing_question", "lenient_typos", "user_positive",
        "allow_doubt_wrap", "enforce_mood_continuity",
    ],
    "retrieval": [
        "use_self_facts", "query_hint", "history_window",
        "hook_history_recall", "hook_callback", "hook_concrete_example",
        "need_diary",
    ],
}


def _field_meta(advisors: dict) -> dict[str, dict]:
    """从现有 advisor 对象抽每个字段的元信息（类型/默认/规则/范围），不重写规则。"""
    meta: dict[str, dict] = {}
    for name, adv in advisors.items():
        typ = type(adv).__name__  # BoolAdvisor / IntAdvisor / TextAdvisor
        meta[name] = {
            "type": "bool" if typ == "BoolAdvisor" else ("int" if typ == "IntAdvisor" else "text"),
            "default": adv.defaults.get(name),
            "target_desc": getattr(adv, "_target_desc", ""),
            "decision_rules": getattr(adv, "_decision_rules", ""),
            "range_desc": getattr(adv, "_range_desc", ""),
            "max_chars": getattr(adv, "_max_chars", 24),
            "minimum": getattr(adv, "_minimum", None),
            "maximum": getattr(adv, "_maximum", None),
        }
    return meta


def _build_group_prompt(group_fields: list[str], meta: dict, ctx: LinaTurnContext) -> str:
    """把一组字段拼成一个 prompt：逐字段给含义+规则+类型，要求一次性输出多字段 JSON。"""
    lines = [
        "你是「西比莉娜」（lina，1760 年前的炼金术学徒）回复决策中的判官。",
        "请**一次性**判断下面**多个**字段，每个字段严格按它自己的含义和规则判，互不影响。",
        "",
    ]
    schema_hint: list[str] = []
    for f in group_fields:
        m = meta[f]
        t = m["type"]
        if t == "bool":
            typ_note = "true / false"
        elif t == "int":
            rng = m.get("range_desc") or f"{m['minimum']}-{m['maximum']}"
            typ_note = f"整数（范围 {rng}）"
        else:
            typ_note = "短字符串（不需要时空串）"
        lines.append(f"字段「{f}」（{typ_note}）")
        lines.append(f"  含义：{m['target_desc']}")
        if m["decision_rules"]:
            lines.append(f"  规则：{m['decision_rules']}")
        lines.append("")
        schema_hint.append(f'"{f}": ...')
    lines += [
        "输入：",
        f"U:{ctx.user_text or '（无具体用户输入，当前是主动发言）'}",
        "H:",
        _render_history(ctx.history),
        f"P:{'1' if ctx.is_proactive else '0'}",
        "",
        "只输出严格 JSON，一个对象包含上面所有字段，形如：",
        "{" + ", ".join(schema_hint) + "}",
        "不要写解释、思考、code fence、其他键。",
    ]
    return "\n".join(lines)


def _coerce(field: str, val: Any, m: dict):
    """按字段类型把 LLM 返回值规范化（缺失/异常用 default）。"""
    t = m["type"]
    if t == "bool":
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in {"1", "true", "yes", "y", "是", "on"}:
            return True
        if s in {"0", "false", "no", "n", "否", "off", ""}:
            return False
        return bool(m["default"])
    if t == "int":
        try:
            iv = int(val)
        except (TypeError, ValueError):
            return m["default"]
        lo, hi = m.get("minimum"), m.get("maximum")
        if lo is not None:
            iv = max(lo, iv)
        if hi is not None:
            iv = min(hi, iv)
        return iv
    # text
    s = str(val or "").strip()
    return s[: m.get("max_chars", 24)]


async def _run_group(client, model: str, timeout: float, group_fields: list[str],
                     meta: dict, ctx: LinaTurnContext) -> tuple[dict, float, str]:
    """跑一组 → 返回 (该组字段dict, 耗时ms, source)。失败/超时 → 全用 default。"""
    defaults = {f: meta[f]["default"] for f in group_fields}
    prompt = _build_group_prompt(group_fields, meta, ctx)
    started = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=700,   # 多字段，给宽一点
                reasoning_effort="minimal",
                response_format={"type": "json_object"},
            ),
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _parse_json_object(raw)
        out = {f: _coerce(f, data.get(f, meta[f]["default"]), meta[f]) for f in group_fields}
        return out, (time.monotonic() - started) * 1000, "llm"
    except Exception:
        return dict(defaults), (time.monotonic() - started) * 1000, "default_error"


async def run_merged_advisors(client, model: str, timeout: float,
                              ctx: LinaTurnContext) -> tuple[dict, dict]:
    """4 组并发跑 → 返回 (merged 字段dict, 每组耗时trace)。
    merged 形状与并发版一致，可直接喂 LinaController._merge。"""
    advisors = build_lina_advisors(client, model=model, timeout=timeout)
    meta = _field_meta(advisors)
    tasks = {
        g: asyncio.create_task(_run_group(client, model, timeout, fields, meta, ctx))
        for g, fields in GROUPS.items()
    }
    merged: dict[str, Any] = {}
    trace: dict[str, Any] = {}
    for g, task in tasks.items():
        fields_out, ms, src = await task
        merged.update(fields_out)
        trace[g] = {"latency_ms": round(ms, 1), "source": src}
    return merged, trace
