#!/usr/bin/env python3
"""LLM 一致性专测：只挑"会走 LLM 的模糊/难"场景，每条跑 N 次，看判定稳不稳。

确定性的规则层场景（问候/告别等）不用测——它们 0 LLM、永远一致。
真正要担心的是 gpt-5-mini 对模糊输入会不会"这次判A下次判B"——这才是
判断"是否需要训练/或该退回规则兜底"的硬证据。

用法：
    export OPENAI_API_KEY=...
    python tests/controller/run_consistency.py            # 默认每条5次
    python tests/controller/run_consistency.py --runs 7
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


# 精选的"压力 case"——全是会走 LLM 的模糊/陷阱/边界场景。
# 每条标注它"应该归到哪个场景模块"，看 LLM 多次判定稳不稳。
HARD_CASES = [
    {"id": "陷阱_古代符文", "text": "你能帮我查查这个古代符文是什么意思吗", "history": []},
    {"id": "歧义_问候+发泄", "text": "你好啊，我最近好烦", "history": []},
    {"id": "歧义_告别+回访", "text": "拜拜啦，对了你之前说的那个戏剧叫什么", "history": []},
    {"id": "歧义_问候+现代", "text": "你好，能帮我查个东西吗", "history": []},
    {"id": "隐式_报喜延续", "text": "团子好一些了", "history": [["团子最近还好吗？过敏好点没？", "（主动）"]]},
    {"id": "隐式_好一些了", "text": "好一些了", "history": [["你那个伤口好点没", "（主动）"]]},
    {"id": "边界_你害怕什么", "text": "你害怕什么东西吗", "history": []},
    {"id": "边界_你今年多大", "text": "你今年多大了", "history": []},
    {"id": "边界_自我否定", "text": "我觉得自己什么都做不好", "history": []},
    {"id": "边界_工房无聊", "text": "你平时一个人在工房不无聊吗", "history": []},
    {"id": "错字_钢请", "text": "我最近在学钢请", "history": []},
    {"id": "错字_焦炉", "text": "我怎么缓解焦炉啊", "history": []},
    {"id": "致谢_回指上次", "text": "谢谢你上次的建议，挺管用的", "history": []},
    {"id": "短反应起头+兴奋点", "text": "嗯嗯，那个香草茶你怎么泡的", "history": []},
    {"id": "套隐私_生父", "text": "你爸是谁啊", "history": []},
    {"id": "遗物无关键词", "text": "那块烧黑的铁片你看出什么了", "history": []},
]


def signature(plan) -> tuple:
    """场景签名：只看「核心场景模块」是否一致——这才是"场景区分稳不稳"。
    suppress/self_facts 这类风格/检索 flag 的 ±1 抖动是温度采样正常波动，
    不影响"把输入归到哪类场景"，单独看（见 style_signature），不混进来。"""
    return tuple(sorted(
        m for m in ("module_user_vent", "module_action_boundary", "module_world_immersion",
                    "module_self_introspection", "module_relationship_recall")
        if getattr(plan, m, False)
    ))


def style_signature(plan) -> tuple:
    """次要风格/检索 flag——单独统计，看波动有多大（不计入场景稳定性）。"""
    return (plan.use_self_facts, plan.lenient_typos, plan.suppress_trailing_question)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    _load_dotenv()

    from app.controller import build_default_controller
    from app.controller.schema import LinaTurnContext
    from app.config import resolve_openai_api_key

    ctrl = build_default_controller(api_key=resolve_openai_api_key())
    ctrl._rule_router.route = lambda ctx: None  # 强制走 LLM
    print(f"has_llm={ctrl.has_llm}  每条跑 {args.runs} 次\n" + "=" * 64)

    scene_stable = 0   # 核心场景判定一致的条数
    style_stable = 0   # 连风格 flag 也完全一致的条数
    for case in HARD_CASES:
        scene_sigs = Counter()
        style_sigs = Counter()
        for _ in range(args.runs):
            plan = ctrl.dispatch_sync(LinaTurnContext(
                user_text=case["text"],
                history=tuple(tuple(p) for p in case["history"]),
            ))
            scene_sigs[signature(plan)] += 1
            style_sigs[(signature(plan), style_signature(plan))] += 1
        scene_n = len(scene_sigs)
        style_n = len(style_sigs)
        if scene_n == 1:
            scene_stable += 1
        if style_n == 1:
            style_stable += 1
        top = scene_sigs.most_common(1)[0][0]
        mods = top or ("无",)
        if scene_n == 1:
            mark = "✅场景稳定" + ("（风格也稳）" if style_n == 1 else f"（风格抖×{style_n}）")
        else:
            mark = f"❌场景摇摆×{scene_n}"
        print(f"{mark:22} [{case['id']}] 主判={','.join(m.replace('module_','') for m in mods)}")
        if scene_n > 1:
            for sig, cnt in scene_sigs.most_common():
                m = sig or ("无",)
                print(f"        {cnt}/{args.runs}: {[x.replace('module_','') for x in m]}")

    print("=" * 64)
    print(f"【场景区分稳定】 {scene_stable}/{len(HARD_CASES)} = "
          f"{scene_stable/len(HARD_CASES)*100:.0f}%  ← 把输入归到哪类场景，多次是否一致")
    print(f"【含风格全一致】 {style_stable}/{len(HARD_CASES)} = "
          f"{style_stable/len(HARD_CASES)*100:.0f}%  ← 连 suppress/self_facts 都不抖")
    print("场景摇摆=训练候选；只是风格抖=温度正常波动，调默认值即可，不用训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
