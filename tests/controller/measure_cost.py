#!/usr/bin/env python3
"""统计 controller 的成本：每个 advisor 的 prompt 长度（字符/估算token）+ 实测耗时。

回答："训练能省多少" —— 训练后 prompt 可大幅缩短、甚至砍掉规则层，
这里先量化现状：总 prompt 体量、每轮总 token、单轮端到端耗时。

用法：
    export OPENAI_API_KEY=...
    python tests/controller/measure_cost.py            # 只统计 prompt 体量（不调 API，快）
    python tests/controller/measure_cost.py --timing   # 额外实测耗时（调 API）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
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


def est_tokens(s: str) -> int:
    """粗估 token：中文约 1 字符≈1 token，英文/符号约 4 字符≈1 token。
    取保守混合：len/1.5。够用于横向比较。"""
    return int(len(s) / 1.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true", help="实测端到端耗时（调 API）")
    ap.add_argument("--runs", type=int, default=3, help="耗时取几次平均")
    args = ap.parse_args()
    _load_dotenv()

    from app.controller.experts import build_lina_advisors
    from app.controller.schema import LinaTurnContext

    # 一个代表性 ctx：中等长度用户输入 + 几轮历史（贴近真实每轮）
    ctx = LinaTurnContext(
        user_text="你最近鉴定了什么有意思的遗物吗，我对古代的东西特别好奇",
        history=(
            ("我养了只猫叫团子", "诶猫啊，它平时黏不黏你"),
            ("挺黏的，老踩我书", "哈哈，书页都给踩皱了吧"),
            ("对，还咬笔", "嗯……古人也用这种笔吗"),
        ),
    )

    advisors = build_lina_advisors(client=None, model="gpt-5-mini")
    print(f"advisor 数 = {len(advisors)}（每轮并发这么多个 gpt-5-mini 请求）")
    print("=" * 70)

    # ---- prompt 体量 ----
    print(f"\n### 各 advisor 的 prompt 长度")
    print(f"{'advisor':28} {'字符':>6} {'估token':>8}")
    total_chars = 0
    total_tok = 0
    for name, adv in advisors.items():
        try:
            prompt = adv._render_prompt(ctx)
        except Exception as e:
            prompt = f"<render失败: {e}>"
        c = len(prompt); t = est_tokens(prompt)
        total_chars += c; total_tok += t
        print(f"{name:28} {c:6} {t:8}")
    print("-" * 44)
    print(f"{'每轮 input 总计（19个相加）':28} {total_chars:6} {total_tok:8}")
    print(f"\n注：每个 advisor 是独立请求，所以每轮 controller 的 input token")
    print(f"   = 上面总和 ≈ {total_tok} token（还要加上每个的 output + reasoning）")

    # ---- 模板本身有多大（训练后能省掉的部分）----
    from app.controller._prompts import load_prompt
    flag_tpl = load_prompt("controller/control_flag.txt")
    print("\n### 模板拆解（训练能省的就是这些判定说明）")
    print(f"  control_flag.txt 模板本身 = {len(flag_tpl)} 字符 ≈ {est_tokens(flag_tpl)} token")
    print(f"  每个 advisor 还各自带 target_desc + decision_rules（判定规则文字）")
    # 量一个典型 advisor 的 decision_rules 长度
    for name in ("module_world_immersion", "lenient_typos", "suppress_trailing_question"):
        adv = advisors.get(name)
        if adv:
            dr = getattr(adv, "_decision_rules", "")
            td = getattr(adv, "_target_desc", "")
            print(f"  [{name}] target_desc={len(td)}字 + decision_rules={len(dr)}字"
                  f"（≈{est_tokens(td+dr)} token，训练后可压缩/省去）")

    # ---- 实测耗时 ----
    if args.timing:
        from app.controller import build_default_controller
        from app.config import resolve_openai_api_key
        ctrl = build_default_controller(api_key=resolve_openai_api_key())
        if not ctrl.has_llm:
            print("\n[跳过耗时] 没有 LLM client")
            return 0
        ctrl._rule_router.route = lambda c: None  # 强制走 LLM，测真实 fan-out 耗时
        print(f"\n### 端到端耗时（强制走 LLM fan-out，{args.runs} 次取平均）")
        times = []
        for i in range(args.runs):
            t0 = time.monotonic()
            ctrl.dispatch_sync(ctx)
            dt = time.monotonic() - t0
            times.append(dt)
            print(f"  第{i+1}次: {dt:.2f}s")
        avg = sum(times) / len(times)
        print(f"  平均: {avg:.2f}s / 轮（19 advisor 并发，受最慢的 + 网络重试影响）")
        print(f"\n  对比：规则层命中时耗时 ≈ 0s（0 次 LLM 调用）")
        print(f"  → 走 LLM 的轮次每次多 ~{avg:.1f}s 延迟 + ~{total_tok} input token")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
