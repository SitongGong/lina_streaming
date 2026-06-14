"""Translate a `LinaPromptPlan` into the dynamic system tail block.

`CharacterEngine` runs with a two-block system:
- Block 1 (cached, ephemeral): the existing core_text + BEHAVIOR_RULES +
  MOOD_FORMAT_SPEC. Unchanged across turns → prompt cache stays warm.
- Block 2 (this composer's output): per-turn module text + length /
  tone constraints + hook reminders. Re-built every turn from the
  current `LinaPromptPlan`.

If a turn produces no module text and no special constraints, the
composer returns an empty string and the engine omits the second block
entirely (so simple turns get the same payload as before).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._prompts import load_prompt
from .schema import LinaPromptPlan


_CONSTRAINTS_FILE = Path(__file__).resolve().parent.parent.parent / "prompts" / "controller" / "constraints.json"

# 兜底（constraints.json 缺失/损坏时用，保证约束块不崩）。
_FALLBACK_CONSTRAINTS = {
    "header": "【本轮回复约束】",
    "sentences": "- 句数：{sentences} 行左右",
    "max_reply_chars": "- 总字数上限：{max_reply_chars}",
    "tone_hint": "- 语气：{tone_hint}",
    "mood_continuity": "- mood 与上一轮连贯，不要突变",
    "no_doubt_wrap": "- 本轮不要用含糊语气开头",
    "no_segment": "- 本轮不要分段，一口气说完。",
    "allow_segment": "- 多个意思可分段，最多 {max_segments} 段。",
    "suppress_question_excited": "- 可以追问，但一条消息最多一个问号。",
    "suppress_question_default": "- 不要在结尾硬甩问句，最多问一个。",
    "lenient_typos": "- 按最合理的意思理解错别字，不要揪着追问。",
}


_CONSTRAINTS_REL = "controller/constraints.json"


def _load_constraints() -> dict[str, str]:
    """读约束句文字，逐条经 override（网页改即时生效）；缺失退回兜底。"""
    from ._prompts import load_json_value
    out = {}
    for key, fb in _FALLBACK_CONSTRAINTS.items():
        out[key] = load_json_value(_CONSTRAINTS_REL, key, fallback=fb)
    return out


_MODULE_PATHS: dict[str, str] = {
    "user_vent": "modules/user_vent.txt",
    "action_boundary": "modules/action_boundary.txt",
    "world_immersion": "modules/world_immersion.txt",
    "relationship_recall": "modules/relationship_recall.txt",
    "self_introspection": "modules/self_introspection.txt",
    "welcome_back": "modules/welcome_back.txt",
    "continuation": "modules/continuation.txt",
    "hook_concrete_example": "modules/hook_concrete_example.txt",
    "hook_callback": "modules/hook_callback.txt",
    "hook_history_recall": "modules/hook_history_recall.txt",
}


@dataclass(frozen=True)
class LinaPromptBundle:
    """Composer output.

    `tail_text` is the full dynamic system tail (possibly empty).
    `module_texts` and `trace` are kept for the inspector / debug panel.
    """

    tail_text: str = ""
    module_texts: dict[str, str] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


class LinaPromptComposer:
    """Stateless composer; safe to share across sessions / threads."""

    def compose(self, plan: LinaPromptPlan) -> LinaPromptBundle:
        blocks: list[str] = []
        module_texts: dict[str, str] = {}

        # Module bodies in their plan-defined order.
        for module_name in plan.module_names:
            path = _MODULE_PATHS.get(module_name)
            if not path:
                continue
            text = load_prompt(path).strip()
            if not text:
                continue
            module_texts[module_name] = text
            blocks.append(text)

        constraint_block = self._build_constraint_block(plan)      # 构建本轮回复的约束，比如容纳错别字，是否分段说，是否限制连珠炮的提问
        if constraint_block:
            blocks.append(constraint_block)

        fewshot_block = self._build_fewshot_block(plan)     # 总结一些few-shot示例，辅助模型回答
        if fewshot_block:
            blocks.append(fewshot_block)

        instruction_block = self._build_instruction_block(plan)
        if instruction_block:
            blocks.append(instruction_block)

        tail_text = "\n\n".join(b.strip() for b in blocks if str(b or "").strip())
        trace = {
            "module_names": plan.module_names,
            "module_chars": sum(len(t) for t in module_texts.values()),
            "constraint_chars": len(constraint_block),
            "instruction_chars": len(instruction_block),
            "fewshot_tags": list(plan.fewshot_tags),
            "fewshot_chars": len(fewshot_block),
            "tail_total_chars": len(tail_text),
            "trace_source": plan.trace_source,
            "matched_rule": plan.matched_rule,
        }
        return LinaPromptBundle(tail_text=tail_text, module_texts=module_texts, trace=trace)

    @staticmethod
    def _build_constraint_block(plan: LinaPromptPlan) -> str:
        # 约束句文字外置到 prompts/controller/constraints.json，改文字不动代码。
        c = _load_constraints()
        lines = [
            c["header"],
            c["sentences"].format(sentences=plan.sentences),
            c["max_reply_chars"].format(max_reply_chars=plan.max_reply_chars),
        ]
        if plan.tone_hint:
            lines.append(c["tone_hint"].format(tone_hint=plan.tone_hint))
        if plan.enforce_mood_continuity:
            lines.append(c["mood_continuity"])
        if not plan.allow_doubt_wrap:
            lines.append(c["no_doubt_wrap"])
        # 切分预算：不准拆就直接说完；准拆则给上限，主模型在框内自定。
        if not plan.allow_segment:
            lines.append(c["no_segment"])
        else:
            lines.append(c["allow_segment"].format(max_segments=plan.max_segments))
        # 行为微调（controller 按场景注入，主模型 prompt 不动）。
        if plan.suppress_trailing_question:
            # 兴奋点保留好奇但限一个问号；其余场景不要硬甩问句。
            lines.append(
                c["suppress_question_excited"] if plan.module_world_immersion
                else c["suppress_question_default"]
            )
        if plan.lenient_typos:
            lines.append(c["lenient_typos"])
        return "\n".join(s for s in lines if s)

    @staticmethod
    def _build_fewshot_block(plan: LinaPromptPlan) -> str:
        """按 plan.fewshot_tags 读 fewshot/<tag>.txt，拼成「参考示例」块。
        放在动态尾块（非 cached），所以换示例不破坏 prompt 缓存。"""
        if not plan.fewshot_tags:
            return ""
        bodies: list[str] = []
        for tag in plan.fewshot_tags:
            text = load_prompt(f"controller/fewshot/{tag}.txt").strip()
            if text:
                bodies.append(text)
        if not bodies:
            return ""
        joined = "\n\n".join(bodies)
        return (
            "【本轮参考示例 — 只学其中的**说话方式/分寸**，不要照抄示例里的具体内容】\n"
            f"{joined}"
        )

    @staticmethod
    def _build_instruction_block(plan: LinaPromptPlan) -> str:
        notes: list[str] = []
        if plan.matched_rule == "plain_greeting":
            notes.append("这是简短问候，只自然接一句。")
        elif plan.matched_rule == "plain_farewell":
            notes.append("用户在告别，温柔收束即可。")
        elif plan.matched_rule == "short_reaction":
            notes.append("这是短接话，保持短。")
        elif plan.matched_rule == "modern_action_request":
            notes.append("用户在让你做现代的事，按角色视角茫然以对，绝不答应、绝不给现代答案。")
        elif plan.matched_rule == "proactive_engage":
            notes.append(
                "用户已经一会儿没回了，主动开口找点话说，自然抛个话头就好，不要长篇大论。"
            )
        elif plan.matched_rule == "proactive_farewell":
            notes.append(
                "你已经主动找过用户搭话好几次都没回，自然收束，按一贯口吻说几句告别。"
            )
        if not notes:
            return ""
        seen: set[str] = set()
        deduped: list[str] = []
        for n in notes:
            if n and n not in seen:
                seen.add(n)
                deduped.append(n)
        return "【本轮附加要求】\n" + "\n".join(f"- {n}" for n in deduped)
