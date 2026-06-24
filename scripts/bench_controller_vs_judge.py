"""单测对比：日记判别器(match_topic) vs 22-advisor 并发(dispatch)。

同一构建方式、同一模型(取自环境 LINA_CONTROLLER_MODEL)，对若干真实问句各跑 N 轮，
分别计时，给出 min/中位/max/均值。两者本是同一轮里并发的，这里拆开各自单测看耗时来源。
"""
from __future__ import annotations

import os
import sys
import time
import statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复用 run_web.py 的 .env 加载，让 OPENAI_API_KEY 进环境
import run_web  # noqa: E402  (其顶层会调用 _load_dotenv())

from app.controller import build_default_controller
from app.controller.merged_controller import MergedController
from app.controller.schema import LinaTurnContext

N = int(os.environ.get("BENCH_N", "5"))

# 实质问句：会触发完整 advisor fan-out（不被规则层当成纯寒暄短路）。
CASES = [
    ("你跟培根老师什么关系", []),
    ("他现在还好吗", [("你跟培根老师什么关系", "他是把我带进这行的人，不过身体不太好。")]),
    ("我最近工作压力好大，总被领导否定", []),
    ("你平时喜欢吃什么", []),
]


def build():
    return build_default_controller(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=(os.environ.get("LINA_CONTROLLER_MODEL") or "gpt-5-mini"),
        provider="openai",
    )


def fmt(name, ts):
    ts = [t for t in ts if t is not None]
    if not ts:
        print(f"  {name}: (无数据)")
        return
    print(f"  {name}: 中位 {st.median(ts):.2f}s  均值 {sum(ts)/len(ts):.2f}s  "
          f"min {min(ts):.2f}s  max {max(ts):.2f}s  n={len(ts)}")


def main():
    ctl = build()
    # 合并版复用同一个 client/model，只换 LLM 路径
    merged = MergedController(openai_client=ctl._client, model_name=ctl._model_name)
    model = ctl._model_name
    print(f"controller 模型 = {model}；每个 case 各跑 {N} 次\n")
    print(f"原版 advisor 数量 = {len(ctl._advisors)}")

    judge_all, fanout_all, merged_all = [], [], []
    src22 = src4 = None
    for msg, hist in CASES:
        print(f"\n=== 「{msg}」 ===")
        ctx = LinaTurnContext(user_text=msg, history=tuple(hist), session_id="bench")

        # 先 warmup 一次（建连接，不计入）
        try:
            ctl.match_topic_sync(msg, hist, None)
            ctl.dispatch_sync(ctx)
            merged.dispatch_sync(ctx)
        except Exception:
            pass
        # 记录走没走 LLM 路径（确认没被规则层短路）
        try:
            src22 = (ctl.last_trace or {}).get("source")
            src4 = (merged.last_trace or {}).get("source")
        except Exception:
            pass

        judge_ts, fan_ts, mer_ts = [], [], []
        for _ in range(N):
            t0 = time.perf_counter()
            try:
                ctl.match_topic_sync(msg, hist, None)
                judge_ts.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"    judge err: {e}")

            t0 = time.perf_counter()
            try:
                ctl.dispatch_sync(ctx)
                fan_ts.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"    fanout err: {e}")

            t0 = time.perf_counter()
            try:
                merged.dispatch_sync(ctx)
                mer_ts.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"    merged err: {e}")
        fmt("日记判别器 match_topic   ", judge_ts)
        fmt("22-advisor 并发 dispatch ", fan_ts)
        fmt("4-merged 合并 dispatch   ", mer_ts)
        judge_all += judge_ts
        fanout_all += fan_ts
        merged_all += mer_ts

    print(f"\n（路径确认：22-advisor source={src22!r}, 4-merged source={src4!r}；"
          f"应为 llm-* 才算真打了 LLM）")
    print("\n========== 汇总 ==========")
    fmt("日记判别器 match_topic   ", judge_all)
    fmt("22-advisor 并发 dispatch ", fanout_all)
    fmt("4-merged 合并 dispatch   ", merged_all)


if __name__ == "__main__":
    raise SystemExit(main())
