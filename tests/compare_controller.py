#!/usr/bin/env python3
"""对比演示：同一句话，有 controller vs 没 controller，两个回复并排打印。

让人直观看到 controller 的作用：场景识别、长度/语气调节、模块注入、
错字容错、抑制连环提问、报喜共情等。

用法：
    export ANTHROPIC_API_KEY 和 OPENAI_API_KEY（或走 .env）
    python tests/compare_controller.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


# 精选最能体现 controller 作用的对比案例。history 用 (user, assistant) 对。
CASES = [
    # —— 现代边界 ——
    {"label": "现代请求·查天气", "expect": "茫然以对，不真去查",
     "user": "帮我查一下明天北京的天气"},
    {"label": "现代请求·跑代码", "expect": "当成没听过的怪术语，不答应",
     "user": "帮我写段Python代码爬个网站"},
    {"label": "AI自指", "expect": "把'AI'当陌生词，不承认是程序",
     "user": "你是不是一个AI啊"},
    # —— 错别字容错 ——
    {"label": "错别字·焦炉", "expect": "领会'焦虑'，不揪错字",
     "user": "我最近老是很焦炉，睡不好"},
    {"label": "错别字·钢请", "expect": "领会'钢琴'，自然接",
     "user": "我最近在学钢请，手指都疼了"},
    {"label": "错别字·戏据", "expect": "领会'戏剧'，顺着聊",
     "user": "昨天那场戏据真的很震撼"},
    # —— 报喜 / 情绪 ——
    {"label": "用户报喜·团子", "expect": "替他高兴，不浇冷水",
     "user": "团子好一些了！", "history": [("团子最近过敏好点没？", "（主动问过）")]},
    {"label": "用户报喜·升职", "expect": "真心恭喜，不泼冷水",
     "user": "我升职了！"},
    {"label": "用户低落", "expect": "先安抚，别连环追问/说教",
     "user": "我觉得自己什么都做不好，好累"},
    {"label": "用户发泄", "expect": "接住情绪，不急着讲道理",
     "user": "今天被老板当众骂了，烦死了"},
    # —— 长度 / 提问控制 ——
    {"label": "简单问候", "expect": "一句话，别长篇",
     "user": "在吗"},
    {"label": "短反应", "expect": "短接一句，别小题大做",
     "user": "真的假的"},
    {"label": "开放分享（易连环提问）", "expect": "先反应再最多问一个，别连甩问号",
     "user": "我昨天去看了场话剧"},
    # —— 兴奋点 ——
    {"label": "兴奋点·遗物", "expect": "热情展开、带细节",
     "user": "你最近鉴定了什么有意思的遗物吗"},
    {"label": "兴奋点·香草", "expect": "兴致高、可多问",
     "user": "薄荷和迷迭香能一起种吗"},
    # —— 歧义 / 陷阱（压力）——
    {"label": "陷阱·古代符文(像现代请求)", "expect": "识别是兴奋点而非现代请求",
     "user": "你能帮我查查这个古代符文什么意思吗"},
    {"label": "歧义·问候+发泄", "expect": "优先安抚而非寒暄",
     "user": "你好啊，我最近好烦"},
    {"label": "歧义·告别+回访", "expect": "优先回应实质追问",
     "user": "拜拜啦，对了你之前说的那个戏剧叫什么"},
    # —— 多轮上下文 ——
    {"label": "隐式延续(靠上下文)", "expect": "顺着上轮，不揪'是的'",
     "user": "是的", "history": [("你是说那块铁片是传动用的？", "对")]},
    {"label": "关系回访", "expect": "记得并回应，不答非所问",
     "user": "你还记得我跟你说过什么吗",
     "history": [("我养了只猫叫团子", "诶猫啊"), ("它老咬我笔", "哈哈")]},
]


def main() -> int:
    _load_dotenv()
    from app.character import CharacterEngine, DEFAULT_MODEL
    from app.conversation import Conversation
    from app.config import resolve_api_key, resolve_controller_settings
    from app.controller import build_default_controller

    key = resolve_api_key()
    if not key:
        print("没有 Anthropic key，无法跑主模型")
        return 1

    cfg = resolve_controller_settings()
    controller = build_default_controller(
        api_key=cfg["api_key"], model=cfg["model"],
        base_url=cfg["base_url"], provider=cfg["provider"],
    )
    print(f"controller: provider={cfg['provider']} model={cfg['model']} has_llm={controller.has_llm}")

    # 两个引擎：一个挂 controller，一个不挂（裸主模型）。
    eng_with = CharacterEngine(api_key=key, static_dir=str(ROOT / "static"),
                               model=DEFAULT_MODEL, controller=controller)
    eng_without = CharacterEngine(api_key=key, static_dir=str(ROOT / "static"),
                                  model=DEFAULT_MODEL, controller=None)

    def seed(history):
        c = Conversation(session_id="cmp")
        for u, a in (history or []):
            c.add("user", u); c.add("assistant", a)
        return c

    rows = []  # (label, user, expect, plan_tag, with_text, without_text)
    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] {case['label']} ...", flush=True)
        try:
            r1 = eng_with.chat(seed(case.get("history")), case["user"])
            plan = r1.plan or {}
            tag = f"rule={plan.get('matched_rule') or 'LLM'} 句≤{plan.get('sentences')} 字≤{plan.get('max_reply_chars')} 抑问={'T' if plan.get('suppress_trailing_question') else 'F'} 容错={'T' if plan.get('lenient_typos') else 'F'}"
            with_text = r1.text.replace("\n", " ")
        except Exception as e:
            tag = "ERR"; with_text = f"出错: {e}"
        try:
            r2 = eng_without.chat(seed(case.get("history")), case["user"])
            without_text = r2.text.replace("\n", " ")
        except Exception as e:
            without_text = f"出错: {e}"
        rows.append((case["label"], case["user"], case.get("expect", ""), tag, with_text, without_text))

    # 写 Markdown 汇总表
    out = ROOT / "docs" / "controller_demo.md"
    out.parent.mkdir(exist_ok=True)
    lines = [
        "# Controller 作用对比演示（有 vs 无 controller）",
        "",
        f"> 同一主模型（{cfg['model']}），同一批输入。**有 controller** 会按场景调节"
        "（长度/语气/是否追问/错字容错/场景模块）；**无 controller** 用一套固定参数应对所有情况。",
        "",
        "| # | 场景 | 用户输入 | 期望 | Controller 决策 | ✅ 有 controller | ❌ 无 controller |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (label, user, expect, tag, w, wo) in enumerate(rows, 1):
        def esc(s): return str(s).replace("|", "\\|")
        lines.append(f"| {i} | {esc(label)} | {esc(user)} | {esc(expect)} | `{esc(tag)}` | {esc(w)} | {esc(wo)} |")
    lines += [
        "",
        "## 怎么看",
        "- **Controller 决策** 列：controller 为这一轮判出的场景 + 参数（句数上限/字数上限/是否抑制提问/是否容错）。无 controller 时这些全是固定默认值。",
        "- 重点对比**错别字**、**用户报喜**、**现代请求**、**歧义/陷阱**几行——差异最明显。",
        "- 无 controller：要么揪错字、要么浇冷水、要么该短的啰嗦、要么把现代请求当真去办。",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 汇总表已写入: {out}")
    print(f"   共 {len(rows)} 个对比案例")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
