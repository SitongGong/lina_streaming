"""长期记忆压测：验证 EverMem 云对话记忆在几百轮后还能不能记起早期事实。

测什么
------
长期记忆的核心价值 = 早期讲过的事，灌大量对话后仍能**换个说法**正确召回。
脚本流程：
  1. 开头埋 N 条「用户事实」（养猫团子 / 考研 / 老家大连 …）。
  2. 灌大量噪声轮次（无关闲聊），把早期事实挤出任何上下文窗口。
  3. 期间分批 flush，触发 EverMem 提取成 profile/episode。
  4. 末尾用**换说法**的 query 逐条追问，统计召回率（不靠原词匹配 = 语义检索的价值）。
  5. 顺带测 EverOS 人设检索（agent 轨道）几个问题。

直接打记忆层（MemoryClient / PersonaMemory），**不走主模型** —— 我们验证的是
记忆系统的写入/检索能力，几百轮主模型调用又慢又烧 token，没必要。

用法：
    EVERMEM_API_KEY=... EVEROS_PERSONA=1 EVEROS_LOCAL_URL=http://127.0.0.1:8090 \
        python tests/stress_long_term_memory.py --turns 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.memory_client import MemoryClient  # noqa: E402
from app.persona_memory import PersonaMemory  # noqa: E402

# 埋点：(用户原话, 莉娜回应, 后面换说法的追问, 命中判定关键词列表)
# 命中判定：召回文本里出现任一关键词即算记起（语义检索应能跨「原词→追问词」召回）。
FACTS = [
    ("我养了只布偶猫，叫团子。", "团子，好可爱的名字。",
     "你还记得我家的宠物吗", ["团子", "猫"]),
    ("我在准备考研，目标是计算机专业。", "考研加油，我陪着你。",
     "我最近在为什么考试忙", ["考研", "计算机"]),
    ("我老家在大连，海边长大的。", "大连的海一定很美。",
     "我是哪里人来着", ["大连", "海"]),
    ("我女朋友叫小柔，我们在一起三年了。", "三年了，很长情呢。",
     "我有没有跟你提过我的感情状况", ["小柔", "女朋友", "三年"]),
    ("我做后端开发，主要用 Go 语言。", "Go 写后端很合适。",
     "我是干什么工作的", ["后端", "Go", "开发"]),
    ("我对咖啡很挑剔，只喝手冲单品。", "手冲单品，讲究。",
     "你知道我平时喝什么吗", ["咖啡", "手冲", "单品"]),
    ("我有点社恐，不太喜欢人多的场合。", "那我们慢慢聊就好。",
     "我性格上有什么需要你注意的", ["社恐", "人多", "场合"]),
    ("我最近在追一部叫《群星》的科幻剧。", "科幻剧，听起来有意思。",
     "我最近在看什么剧", ["群星", "科幻"]),
]

# 噪声对话池（无关闲聊，灌进去把早期事实挤出窗口）。
NOISE = [
    ("今天天气怎么样", "外面看着挺晴的。"),
    ("你觉得呢", "我觉得都行，看你。"),
    ("嗯嗯", "嗯。"),
    ("有点累", "那歇会儿吧。"),
    ("随便聊聊", "好啊，想聊什么。"),
    ("没什么特别的", "平淡也挺好。"),
    ("你说呢", "我也说不好。"),
    ("哈哈", "笑什么呢。"),
    ("无聊", "找点事做做。"),
    ("在吗", "在的。"),
]


def _safe_flush(mc: MemoryClient, uid: str, sess: str, retries: int = 2) -> None:
    """flush 容错：攒的消息多时提取慢、易 ReadTimeout，重试几次，仍失败就跳过
    （不影响已写入的数据，下次 flush 或检索时仍在）。"""
    for attempt in range(retries + 1):
        try:
            mc.flush(uid, session_id=sess)
            return
        except Exception as e:
            print(f"    flush 超时/失败（第{attempt+1}次）：{type(e).__name__}，{'重试' if attempt < retries else '跳过'}")
            time.sleep(3)


def run(uid: str, turns: int) -> None:
    mc = MemoryClient()
    if not mc.enabled:
        print("✗ MemoryClient 未启用（缺 EVERMEM_API_KEY）"); return
    # 提取累积消息可能很慢，flush 给足时间（覆盖 MemoryClient 默认的 30s）。
    mc._write_timeout = 120.0
    print(f"用户 user_id = {uid} | 计划 {turns} 轮")
    sess = f"stress_{uid}"

    # 1) 埋点
    print("\n[1] 埋入早期事实...")
    for u, a, *_ in FACTS:
        mc.add_turn(uid, u, a, session_id=sess)
    _safe_flush(mc, uid, sess)
    print(f"    埋入 {len(FACTS)} 条，已 flush")

    # 2) 灌噪声
    print(f"\n[2] 灌 {turns} 轮噪声对话（每 50 轮 flush 一次）...")
    t0 = time.monotonic()
    for i in range(turns):
        u, a = NOISE[i % len(NOISE)]
        mc.add_turn(uid, f"{u}（第{i+1}轮）", a, session_id=sess)
        if (i + 1) % 25 == 0:
            _safe_flush(mc, uid, sess)
            print(f"    {i+1}/{turns} 轮，已 flush（耗时 {time.monotonic()-t0:.0f}s）")
    _safe_flush(mc, uid, sess)
    print("    末次 flush，等待提取沉淀 8s...")
    time.sleep(8)

    # 3) 换说法追问，统计召回
    print("\n[3] 换说法追问早期事实（召回率）...")
    hit = 0
    for u, a, query, keywords in FACTS:
        txt = mc.search_text(uid, query, top_k=8)
        ok = any(k in txt for k in keywords)
        hit += ok
        mark = "✓" if ok else "✗"
        snippet = txt.replace("\n", " ")[:70] if txt else "(空)"
        print(f"  {mark} 问「{query}」期望{keywords} → {snippet}")
    print(f"\n  === 对话记忆召回率：{hit}/{len(FACTS)} = {hit/len(FACTS)*100:.0f}% ===")

    # 4) 人设检索（EverOS agent 轨道）
    print("\n[4] 人设检索（EverOS agent 轨道）...")
    pm = PersonaMemory()
    if not pm.enabled:
        print("    （PersonaMemory 未启用，跳过）")
        return
    for q in ["莉娜住的世界科技水平如何", "西比莉娜是什么性格", "莉娜有什么兴趣爱好"]:
        chunks = pm.retrieve(q, k=3)
        heads = [f"{c.source}·{c.heading}" for c in chunks]
        print(f"  问「{q}」→ {heads or '(空)'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=300, help="噪声轮数")
    ap.add_argument("--uid", default="", help="user_id（默认按时间戳生成，避免脏数据）")
    args = ap.parse_args()
    uid = args.uid or f"stress_ltm_{int(time.time())}"
    run(uid, args.turns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
