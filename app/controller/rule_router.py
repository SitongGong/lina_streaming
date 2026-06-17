"""Deterministic rule layer.

For stable, repeatable scenarios we don't need an LLM — a regex is faster,
free, and more predictable. Any scenario whose handling is *not* fully
nailed down should be left to the fan-out advisors instead of being
half-baked here. When in doubt: return None and let the controller fan
out.

Returned plans always set `trace_source='rule'` so logs make it obvious
which path produced the decision.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import LinaPromptPlan, LinaTurnContext


# 规则层关键词外置到 prompts/controller/rules.json，改关键词不必动代码。
_RULES_FILE = Path(__file__).resolve().parent.parent.parent / "prompts" / "controller" / "rules.json"

# 内置兜底（rules.json 缺失/损坏时用，保证规则层不崩；正常以文件为准）。
_FALLBACK_RULES: dict[str, list[str]] = {
    "greeting": [r"^(你好|哈喽|hello|hi|嗨|早上好|晚上好|在吗)$"],
    "farewell": [r"^(拜拜|回头见|先走了|我先睡了|下次聊|改天聊)$"],
    "short_reaction": [r"^(嗯+|哦+|啊+|哈+|真的假的|离谱)$"],
    "modern_action": [r"(帮我|给我).{0,12}(搜索|查询|打开|下载|运行|翻译)", r"(写段代码|打开链接)"],
    "user_vent": [r"(累死|烦死|崩溃|心累|压力大|焦虑|好烦|太累)"],
    "relationship_recall": [r"(还记得|记得我|之前(说|讲|聊|提)过|你答应过|好久不见)"],
    "world_immersion": [r"(古代语|楔形|碑文|羊皮卷|炼金|魔法石|遗物|符文|戏剧|话剧|香草|草药|薄荷)"],
    "self_introspection": [r"(你是谁|介绍一下你自己|你性格|你害怕什么|你从哪里来|你小时候)"],
}


_RULES_REL = "controller/rules.json"


def _load_rule_patterns() -> dict[str, tuple[re.Pattern[str], ...]]:
    """从 rules.json 读各场景的正则并编译，**逐场景经 override**（网页改某场景的
    关键词即时生效；override 值为一个 JSON 数组字符串）。文件/字段坏 → 内置兜底。"""
    from ._prompts import load_json_value, get_prompt_overrides, make_json_key
    raw: dict
    try:
        raw = json.loads(_RULES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = _FALLBACK_RULES
    overrides = get_prompt_overrides()
    out: dict[str, tuple[re.Pattern[str], ...]] = {}
    for key, patterns in raw.items():
        if key.startswith("_") or not isinstance(patterns, list):
            continue  # 跳过 _comment 等元信息
        ovk = make_json_key(_RULES_REL, key)
        if ovk in overrides:  # 网页改了这一场景的关键词（存为 JSON 数组字符串）
            try:
                patterns = json.loads(overrides[ovk])
            except ValueError:
                pass
        compiled = []
        for p in (patterns if isinstance(patterns, list) else []):
            try:
                compiled.append(re.compile(str(p), re.I))
            except re.error:
                continue  # 单条坏正则跳过
        out[key] = tuple(compiled)
    return out


_RULE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {}
_GREETING_PATTERNS: tuple = ()
_FAREWELL_PATTERNS: tuple = ()
_SHORT_REACTION_PATTERNS: tuple = ()
_MODERN_ACTION_PATTERNS: tuple = ()
_USER_VENT_PATTERNS: tuple = ()
_RELATIONSHIP_PATTERNS: tuple = ()
_WORLD_IMMERSION_PATTERNS: tuple = ()
_SELF_INTROSPECTION_PATTERNS: tuple = ()


def reload_rule_patterns() -> None:
    """重新从 rules.json + override 层加载并编译正则。网页改了规则后调一次即生效。"""
    global _RULE_PATTERNS, _GREETING_PATTERNS, _FAREWELL_PATTERNS, _SHORT_REACTION_PATTERNS
    global _MODERN_ACTION_PATTERNS, _USER_VENT_PATTERNS, _RELATIONSHIP_PATTERNS
    global _WORLD_IMMERSION_PATTERNS, _SELF_INTROSPECTION_PATTERNS
    _RULE_PATTERNS = _load_rule_patterns()
    _GREETING_PATTERNS = _RULE_PATTERNS.get("greeting", ())
    _FAREWELL_PATTERNS = _RULE_PATTERNS.get("farewell", ())
    _SHORT_REACTION_PATTERNS = _RULE_PATTERNS.get("short_reaction", ())
    _MODERN_ACTION_PATTERNS = _RULE_PATTERNS.get("modern_action", ())
    _USER_VENT_PATTERNS = _RULE_PATTERNS.get("user_vent", ())
    _RELATIONSHIP_PATTERNS = _RULE_PATTERNS.get("relationship_recall", ())
    _WORLD_IMMERSION_PATTERNS = _RULE_PATTERNS.get("world_immersion", ())
    _SELF_INTROSPECTION_PATTERNS = _RULE_PATTERNS.get("self_introspection", ())


reload_rule_patterns()  # 启动时加载一次


# 超过这个间隔（秒）用户才回来，算「久别重逢」，走 welcome_back。
# 30 分钟：比"喝杯水回来"长，比"第二天"短，落在"离开了一会儿"这个区间。
_WELCOME_BACK_GAP_SECONDS = 30 * 60


def _match_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(text) for p in patterns)


def _join_hint(*parts: str, limit: int = 24) -> str:
    seen: set[str] = set()
    picked: list[str] = []
    for raw in parts:
        s = str(raw or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        picked.append(s)
    return " ".join(picked)[:limit]


class LinaRuleRouter:
    """Deterministic regex routing. Returns None if no rule fires."""

    # 规则层场景 → 该注入的场景专属 few-shot 示例库。
    _SCENE_FEWSHOT = {
        "user_vent": "comfort",                 # 安抚：先接情绪别说教
        "modern_action_request": "modern_boundary",  # 现代请求：茫然以对别出戏
        # 正向回应（报喜）没有专属规则场景（多走 LLM 或 relationship_recall），
        # 在 LLM 路径里按情绪/语义带出 positive_response（见 controller._merge）。
        "relationship_recall": "positive_response",  # 回访常含报喜/致谢，给正向示例兜底
    }

    def route(self, ctx: LinaTurnContext) -> LinaPromptPlan | None:
        plan = self._route(ctx)
        if plan is None:
            return None
        return self._apply_behavior_defaults(ctx, plan)

    @staticmethod
    def _apply_behavior_defaults(ctx: LinaTurnContext, plan: LinaPromptPlan) -> LinaPromptPlan:
        """规则层产出的 plan 不走微顾问，需要在这里补上两个行为微调开关：
        - suppress_trailing_question：默认抑制「句尾强行甩问号」，但 world_immersion
          这种「追问才是魅力」的兴奋点场景不抑制。
        - lenient_typos：自由文本场景（用户随手打字）善意理解错别字。
        告别/续说/空输入这类极短或无用户文本的场景，两者都没意义，保持关闭。
        """
        # 告别/续说/空输入：极短、无意义，两个开关都跳过。
        skip = plan.matched_rule in {
            "proactive_farewell", "continuation", "empty_input",
        }
        if skip:
            return plan
        # 句尾抑制**所有场景都开**（含兴奋点、含主动发言）——区别交给 composer：兴奋点仍可
        # 追问，但限"一次一个问题、别第一段就连珠炮"，而不是整条都不许问。
        # 主动发言（proactive_engage）尤其要压：它本就该自然起个话头，不该上来甩问号。
        suppress = True
        # 用户自由发言的场景才需要容错；主动发言没有用户输入、纯问候/短反应字少，关掉省事。
        lenient = plan.matched_rule in {
            "user_vent", "relationship_recall", "self_introspection",
            "world_immersion", "welcome_back", "modern_action_request",
        }
        # few-shot 注入：场景专属示例放最前（优先级高，截断时先保留），
        # 通用示例（别连问/容错）搭 suppress/lenient 的便车放后面。
        # _SCENE_FEWSHOT 把规则层场景映射到对应示例库。
        tags = list(plan.fewshot_tags)
        scene_tag = LinaRuleRouter._SCENE_FEWSHOT.get(plan.matched_rule)
        if scene_tag and scene_tag not in tags:
            tags.insert(0, scene_tag)
        if suppress and "no_trailing_question" not in tags:
            tags.append("no_trailing_question")
        if lenient and "typo_tolerance" not in tags:
            tags.append("typo_tolerance")
        # world.md（世界观）：只在涉及她所在世界/身份/古代设定的场景检索，
        # 其余场景关掉省 token（问候、安抚、现代请求都用不到世界观细节）。
        use_world = plan.matched_rule in {
            "world_immersion", "self_introspection", "welcome_back",
        }
        # sample_conversations.md（说话范例）：教她"怎么说话"，对大多数真聊天有用；
        # 现代请求（要茫然以对）和已经极短的场景不需要。
        use_sample = plan.matched_rule not in {"modern_action_request"}
        return LinaPromptPlan(
            **{
                **plan.to_dict(),
                "suppress_trailing_question": suppress,
                "lenient_typos": lenient,
                "fewshot_tags": tuple(tags),
                "use_world": use_world,
                "use_sample_conversations": use_sample,
            }
        )

    def _route(self, ctx: LinaTurnContext) -> LinaPromptPlan | None:
        # 主动告别：在 proactive_farewell 路径里走。短、温柔、不挖深历史。
        if ctx.is_farewell:
            return LinaPromptPlan(
                use_static_personality=False,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=False,
                use_cross_session_memory=False,
                retrieve_k=0,
                history_recall_k=0,
                history_window=8,
                sentences=2,
                max_reply_chars=60,
                tone_hint="温柔",
                allow_doubt_wrap=False,
                allow_segment=False,   # 告别要短，不拆段
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="proactive_farewell",
            )

        # 主动发言：复用近期话头，挂历史钩子，不要长篇。
        if ctx.is_proactive:
            return LinaPromptPlan(
                use_static_personality=True,
                use_static_hobbies=True,
                use_static_others=False,
                use_history_recall=True,
                use_cross_session_memory=ctx.has_cross_session_memory,
                query_hint=_join_hint("近期话题", "未聊完的事"),
                retrieve_k=2,
                history_recall_k=3,
                history_window=12,
                hook_history_recall=True,
                hook_callback=len(ctx.history) >= 2,
                sentences=2,
                max_reply_chars=60,
                tone_hint="自然",
                allow_segment=False,   # 主动开口是一小句，不拆段
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="proactive_engage",
            )

        # 续说：把上一条回复没说完的小段接着说。极短、顺着上一段语气、
        # 不重新检索（素材在上一轮已经定了）。
        if ctx.is_continuation:
            return LinaPromptPlan(
                use_static_personality=False,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=False,
                use_cross_session_memory=False,
                retrieve_k=0,
                history_recall_k=0,
                history_window=8,
                module_continuation=True,
                sentences=2,
                max_reply_chars=50,
                tone_hint="自然",
                allow_segment=False,   # 续说本身就是一小段，不再二次拆
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="continuation",
            )

        text = ctx.user_text

        # 久别重逢：用户离开一段时间后带着新消息回来。带时间感 + 勾历史，
        # 立刻给一个"被记住、被惦记"的理由。放在内容匹配之前，
        # 但只在确有较大间隔时触发，普通连续对话不受影响。
        if text and ctx.gap_seconds >= _WELCOME_BACK_GAP_SECONDS and len(ctx.history) >= 1:
            return LinaPromptPlan(
                use_static_personality=True,
                use_static_hobbies=True,
                use_static_others=False,
                use_history_recall=True,
                use_cross_session_memory=ctx.has_cross_session_memory,
                query_hint=_join_hint("上次的话题", "未聊完的事", text[:12]),
                retrieve_k=2,
                history_recall_k=5,
                history_window=20,
                module_welcome_back=True,
                hook_history_recall=True,
                hook_callback=len(ctx.history) >= 2,
                sentences=2,
                max_reply_chars=60,
                tone_hint="熟悉",
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="welcome_back",
            )

        if not text:
            return LinaPromptPlan(
                use_static_personality=False,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=False,
                use_cross_session_memory=False,
                retrieve_k=0,
                history_recall_k=0,
                history_window=4,
                sentences=1,
                max_reply_chars=30,
                allow_segment=False,
                trace_source="rule",
                matched_rule="empty_input",
            )

        if _match_any(text, _MODERN_ACTION_PATTERNS):       # 和现代生活相关的话题需要拒绝
            return LinaPromptPlan(
                use_static_personality=False,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=False,
                use_cross_session_memory=False,
                retrieve_k=0,
                history_recall_k=0,
                history_window=4,
                module_action_boundary=True,
                sentences=2,
                max_reply_chars=55,
                tone_hint="疑惑",
                allow_doubt_wrap=True,
                allow_segment=False,   # 能力边界回应要干脆，不拆段
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="modern_action_request",
            )

        if _match_any(text, _USER_VENT_PATTERNS):      # 用户抱怨，安慰用户
            return LinaPromptPlan(
                use_static_personality=True,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=True,
                use_cross_session_memory=ctx.has_cross_session_memory,
                query_hint=_join_hint("最近压力", "近期情绪"),
                retrieve_k=2,
                history_recall_k=4,
                history_window=18,
                module_user_vent=True,
                hook_history_recall=ctx.has_cross_session_memory,
                allow_doubt_wrap=False,
                sentences=3,
                max_reply_chars=70,
                tone_hint="温柔",
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="user_vent",
            )

        if _match_any(text, _RELATIONSHIP_PATTERNS):      # 回忆之前发生过的事情
            return LinaPromptPlan(
                use_static_personality=True,
                use_static_hobbies=True,
                use_static_others=True,
                use_history_recall=True,
                use_cross_session_memory=True,
                use_self_facts=True,   # 关系回访常涉及"你之前说过…"，查自我事实
                query_hint=_join_hint(text[:16]),
                retrieve_k=3,
                history_recall_k=5,
                history_window=30,
                module_relationship_recall=True,
                hook_history_recall=True,
                hook_callback=len(ctx.history) >= 2,
                sentences=3,
                max_reply_chars=70,
                tone_hint="熟悉",
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="relationship_recall",
            )

        if _match_any(text, _WORLD_IMMERSION_PATTERNS):
            # 兴奋点：开长、开钩子、用 hobbies/others 检索她真实的喜好细节。
            return LinaPromptPlan(
                use_static_personality=True,
                use_static_hobbies=True,
                use_static_others=True,
                use_history_recall=True,
                use_cross_session_memory=ctx.has_cross_session_memory,
                use_self_facts=True,   # 兴奋点常涉及她自己研究/经历过的东西
                query_hint=_join_hint(text[:20]),
                retrieve_k=5,
                history_recall_k=3,
                history_window=24,
                module_world_immersion=True,
                hook_concrete_example=True,
                hook_callback=len(ctx.history) >= 2,
                allow_doubt_wrap=True,
                sentences=3,
                max_reply_chars=80,
                tone_hint="雀跃",
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="world_immersion",
            )

        if _match_any(text, _SELF_INTROSPECTION_PATTERNS):      # 介绍自己相关的话题
            return LinaPromptPlan(
                use_static_personality=True,
                use_static_hobbies=True,
                use_static_others=False,
                use_history_recall=True,
                use_cross_session_memory=ctx.has_cross_session_memory,
                use_self_facts=True,   # 问莉娜自己 → 查她亲口说过的自我事实
                query_hint=_join_hint(text[:20], "性格", "经历"),
                retrieve_k=5,
                history_recall_k=2,
                history_window=18,
                module_self_introspection=True,
                hook_concrete_example=True,
                allow_doubt_wrap=True,
                sentences=3,
                max_reply_chars=80,
                tone_hint="认真",
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="self_introspection",
            )

        if _match_any(text, _GREETING_PATTERNS) or _match_any(text, _FAREWELL_PATTERNS):
            is_farewell = _match_any(text, _FAREWELL_PATTERNS)       # 用户主动告别
            return LinaPromptPlan(
                use_static_personality=False,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=False,
                use_cross_session_memory=False,
                retrieve_k=0,
                history_recall_k=0,
                history_window=6,
                sentences=1,
                max_reply_chars=30,
                tone_hint="温柔" if is_farewell else "自然",
                allow_doubt_wrap=False,
                allow_segment=False,   # 问候/告别一句话，不拆段
                enforce_mood_continuity=True,
                user_farewell=is_farewell,   # 规则层告别命中 → 标记用户在告别（停主动）
                trace_source="rule",
                matched_rule="plain_farewell" if is_farewell else "plain_greeting",
            )

        if _match_any(text, _SHORT_REACTION_PATTERNS):      # 一般的语气词回复
            return LinaPromptPlan(
                use_static_personality=False,
                use_static_hobbies=False,
                use_static_others=False,
                use_history_recall=False,
                use_cross_session_memory=False,
                retrieve_k=0,
                history_recall_k=0,
                history_window=6,
                sentences=1,
                max_reply_chars=24,
                tone_hint="轻松",
                allow_doubt_wrap=False,
                allow_segment=False,   # 短反应一句话，不拆段
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="short_reaction",
            )

        return None
