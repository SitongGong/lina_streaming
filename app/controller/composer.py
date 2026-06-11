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

from dataclasses import dataclass, field
from typing import Any

from ._prompts import load_prompt
from .schema import LinaPromptPlan


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
        lines = [
            "【本轮回复约束】",
            f"- 句数（每行 = 1 口气）：{plan.sentences} 行左右",
            f"- 总字数上限：{plan.max_reply_chars}",
        ]
        if plan.tone_hint:
            lines.append(f"- 语气：{plan.tone_hint}")
        if plan.enforce_mood_continuity:
            lines.append("- mood 与上一轮连贯，不要突变（除非用户做了明显冒犯或惊喜的事）")
        if not plan.allow_doubt_wrap:
            lines.append("- 本轮不要用「我也不太确定 / 让我想想」开头；直接温柔或坦率地接")
        # 切分预算：覆盖「分段说话机制」里那个"你自己判断要不要拆"。
        # 由 controller 给框——不准拆就直接说完；准拆则给上限，主模型在框内自定。
        if not plan.allow_segment:
            lines.append("- 本轮**不要分段**（不输出 [segments:…]），一口气把话说完即可。")
        else:
            lines.append(
                f"- 本轮**如果有多个意思**可以分段说：第一段先发，其余写进 [segments:…]，"
                f"最多 {plan.max_segments} 段；只有一个意思就别硬凑。"
            )
        # 行为微调（controller 按场景注入，主模型 prompt 不动）。
        if plan.suppress_trailing_question:
            if plan.module_world_immersion:
                # 兴奋点：保留她的好奇，但限"一次一个问题"，别第一段就连珠炮。
                lines.append(
                    "- 你对这个话题很感兴趣，可以追问——但**一条消息里最多一个问号**，"
                    "别第一段就连甩两三个问句像审问。先给一句你自己的真反应/感想，再问那一个最想问的。"
                )
            else:
                lines.append(
                    "- 本轮**不要在结尾硬甩问句**：可以陈述、附和、分享自己类似的经历来接住话，"
                    "不必每条都用问号收尾。真有想问的，**最多问一个**，别连珠炮追问、别为了显得在互动而凑问句。"
                )
        if plan.lenient_typos:
            lines.append(
                "- 用户可能有错别字/漏字/拼音/同音字：**按最通顺合理的意思去理解**，自然接住，"
                "就当对方本来要说的是那个意思。**不要揪着明显的笔误反复追问、纠错、或当成没听过的怪词**。"
                "只有整句确实无法推断时才轻描淡写确认一次。"
            )
        return "\n".join(lines)

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
