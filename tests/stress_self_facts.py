#!/usr/bin/env python3
"""压测自我清单：模拟多轮对话（专门诱导 Lina 编一次性遗物剧情），
走真实的 chat() + update_self_facts() 累积路径，最后检查清单有没有被
一次性剧情污染（领导反馈的 bug）。controller 走 Claude（不烧 gpt-5-mini）。

用法：
    python tests/stress_self_facts.py --turns 200 --user _stress_test
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    s = raw.strip()
    if s and not s.startswith("#") and "=" in s:
        k, v = s.split("=", 1); k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

# 一次性剧情/具体遗物的特征词——清单里若出现，就是污染（不该记的）。
POLLUTION = ["盒子", "铜币", "铜牌", "铜片", "铁片", "陶片", "石板", "木牌",
             "撬开", "翻出", "送来", "刻着", "刻有", "发现一"]

# 轮流发的用户消息，故意诱导她讲具体遗物 / 编剧情。
PROMPTS = [
    "你最近鉴定了什么遗物吗",
    "那东西上面有花纹吗",
    "打开看看里面有什么",
    "好神奇，还有别的吗",
    "这个是从哪挖出来的",
    "你研究古代语有什么新发现",
    "讲讲你最近遇到的怪事",
    "你工房里最奇怪的东西是什么",
    "那块东西后来怎么处理了",
    "你不累吗，天天研究这些",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--user", default="_stress_test")
    args = ap.parse_args()

    from app.character import CharacterEngine, DEFAULT_MODEL
    from app.conversation import Conversation
    from app.config import resolve_api_key, resolve_controller_settings, STATIC_DIR
    from app.controller import build_default_controller
    from app.self_facts import SelfFactsStore

    cfg = resolve_controller_settings()
    assert cfg["provider"] == "anthropic", f"controller 不是 Claude: {cfg['provider']}"
    ctrl = build_default_controller(api_key=cfg["api_key"], model=cfg["model"],
                                    base_url=cfg["base_url"], provider=cfg["provider"])
    eng = CharacterEngine(api_key=resolve_api_key(), static_dir=str(STATIC_DIR),
                          model=DEFAULT_MODEL, controller=ctrl)
    store = SelfFactsStore(ROOT / "users" / "self_facts")
    # 干净起点
    facts: dict = {}
    conv = Conversation(session_id="stress")

    print(f"controller={cfg['model']}  跑 {args.turns} 轮 …")
    t0 = time.monotonic()
    for i in range(args.turns):
        msg = PROMPTS[i % len(PROMPTS)]
        try:
            r = eng.chat(conv, msg, self_facts=facts)
        except Exception as e:
            print(f"  轮 {i}: chat 出错 {e}"); continue
        # 复刻 web：有轮次滑出窗口就更新清单
        if r.slid_out_turns:
            updated = ctrl.update_self_facts_sync(dict(facts), list(r.slid_out_turns))
            if updated is not None:
                facts = updated
        if (i + 1) % 20 == 0:
            total = sum(len(v) for v in facts.values())
            print(f"  轮 {i+1}: 清单共 {total} 条  耗时 {time.monotonic()-t0:.0f}s", flush=True)

    store.save(args.user, facts)
    print("\n" + "=" * 60)
    print("最终清单：")
    polluted = []
    for bucket, items in facts.items():
        for it in items:
            mark = ""
            if any(p in str(it) for p in POLLUTION):
                mark = "  ⚠️污染(一次性剧情)"; polluted.append(it)
            print(f"  [{bucket}] {it}{mark}")
    print("=" * 60)
    total = sum(len(v) for v in facts.values())
    print(f"共 {total} 条；其中疑似污染 {len(polluted)} 条")
    print("✅ 修复有效：清单只剩稳定事实" if not polluted
          else f"❌ 仍有 {len(polluted)} 条一次性剧情混入")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
