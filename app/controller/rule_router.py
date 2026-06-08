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

import re

from .schema import LinaPromptPlan, LinaTurnContext


_GREETING_PATTERNS = (
    re.compile(r"^(你好|你好呀|哈喽|hello|hi|嗨|早上好|下午好|晚上好|在吗)$", re.I),
    re.compile(r"^(早安|早呀|晚安|午安)$", re.I),
)

_FAREWELL_PATTERNS = (
    re.compile(r"^(拜拜|拜|回头见|先走了|我先撤了|我先睡了|下次聊|改天聊)$", re.I),
)

_SHORT_REACTION_PATTERNS = (
    re.compile(r"^(嗯+|哦+|啊+|诶+|欸+|哈+|草+|6+|牛啊|牛哇|真的假的|离谱|笑死|绷不住了)$", re.I),
)

# 现代/超出 1760 年知识边界 的请求 → 强制 action_boundary 模块
_MODERN_ACTION_PATTERNS = (
    re.compile(
        r"(帮我|给我|替我).{0,12}(搜|搜索|查|查询|打开|点开|访问|下载|播放|运行|执行|生成代码|发链接|翻译)",
        re.I,
    ),
    re.compile(
        r"(你能不能|可以不可以|能不能).{0,16}(搜|查|打开|访问|下载|播放|运行|执行|写代码|控制|操作设备)",
        re.I,
    ),
    re.compile(r"(打开链接|点开链接|访问网址|帮我查|写个脚本|写段代码|控制电脑|控制设备)", re.I),
    re.compile(r"(互联网|手机|电脑|app|应用程序|api|ai|神经网络|机器学习|chatgpt|claude|gpt)", re.I),
)

# 用户在 emo / 发泄
_USER_VENT_PATTERNS = (
    re.compile(
        r"(累死|烦死|崩溃|emo|难受|哭了|心累|压力大|顶不住|撑不住|哭死|要命|好惨|焦虑|郁闷|窒息|好烦|真烦|太累|好累)",
        re.I,
    ),
    re.compile(r"(唉+|哎+).{0,4}(累|烦|怎么办|活不下去|没意思|心累)", re.I),
    re.compile(r"(感觉|觉得).{0,4}(活不下去|撑不下去|没意思|没动力|好痛苦|很丧)", re.I),
)

# 关系回访 / 「还记得吗」
_RELATIONSHIP_PATTERNS = (
    re.compile(r"(还记得|记得我|认得我|认出我|上次|之前(说|讲|聊)过|你答应过|好久不见|回来了|老粉)", re.I),
)

# lina 的兴奋点 —— 古代语 / 遗物 / 戏剧 / 香草 / 甜点 / 炼金 / 魔法石
_WORLD_IMMERSION_PATTERNS = (
    re.compile(r"(古代语|古代文字|古文字|古文|楔形|象形|碑文|铭文|手稿|羊皮卷|竹简)", re.I),
    re.compile(r"(炼金|魔法石|魔法|遗物|古物|文物|符文|圣物|废墟)", re.I),
    re.compile(r"(戏剧|话剧|歌剧|戏本|台词|莎士比亚|希腊悲剧)", re.I),
    re.compile(r"(香草|草药|薄荷|薰衣草|迷迭香|百里香|鼠尾草|甘草|药草)", re.I),
    re.compile(r"(甜点|糕点|布丁|蛋糕|司康|果酱|蜂蜜)", re.I),
)

# 问她自己 —— 性格 / 经历 / 来历 / 世界观位置
_SELF_INTROSPECTION_PATTERNS = (
    re.compile(r"(你是谁|你是个什么样的人|介绍一下你自己|讲讲你自己|聊聊你自己|你性格|你脾气|你喜欢什么|你害怕什么)", re.I),
    re.compile(r"(你从哪里来|你哪里来的|你出身|你的过去|你小时候|你以前|你怎么成为|你怎么变成)", re.I),
    re.compile(r"(你的世界|你那个世界|你所在的世界|你那里的)", re.I),
)


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

    def route(self, ctx: LinaTurnContext) -> LinaPromptPlan | None:
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
                enforce_mood_continuity=True,
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
                enforce_mood_continuity=True,
                trace_source="rule",
                matched_rule="short_reaction",
            )

        return None
