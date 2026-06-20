"""100 轮对话评测，验证三件事（穿插设计）：
1. 问莉娜的经历/偏好 → 能否命中话题卡+日记（看 [diary] 日志 命中话题）
2. 用户讲自己的事 → 莉娜共情 + EverMem 跨轮记住（后面追问用户早期事实）
3. 对某话题代词追问/接续 → 能否识别接续、紧接上文（看 接续老话题 / 前后一致）

每轮打印：用户句、莉娜回复、[diary决策标记]。跑完人看连续性/共情/检索准不准。
"""
from __future__ import annotations
import json, sys, time, re
import httpx

BASE = "http://127.0.0.1:8080"
CID = "eval100"
SID = "eval100sess"

# (标签, 用户说的话) —— 标签仅供阅读分组
TURNS = [
    # ── 开场 + 埋用户事实（目标2的长程记忆锚点）──
    ("寒暄", "嗨，在吗"),
    ("用户事实", "我叫阿哲，最近在准备研究生考试"),
    ("用户事实", "我还养了只橘猫，叫芝麻"),
    # ── 目标1：问莉娜经历/偏好（命中A/B话题卡）──
    ("问莉娜", "你那边今天天气怎么样"),
    ("接续", "那你喜欢这种天气吗"),                      # 目标3：代词/接续
    ("问莉娜", "你平时一个人在工房都做些什么"),
    ("接续", "那不会觉得无聊吗"),                        # 接续独处话题
    ("问莉娜", "你吃东西口味重还是清淡"),
    ("接续", "有没有什么是你绝对不吃的"),                 # 接续口味话题
    # ── 目标2：用户倾诉，看共情（不抢话）──
    ("用户倾诉", "唉，今天复习了一整天，好累"),
    ("用户倾诉", "感觉怎么背都背不进去，有点焦虑"),
    ("问莉娜共鸣", "你有没有过那种怎么努力都达不到的时候"),  # 目标1+2：问经历共鸣
    # ── 目标3：隔几轮后绕回早先话题 ──
    ("闲聊", "对了你养花吗"),
    ("问莉娜", "你做饭吗，有什么拿手菜"),
    ("接续", "那味道怎么样，好吃吗"),                     # 接续做饭话题
    # ── 目标2：验证长程记忆（早先说的考研/猫）──
    ("查记忆", "你还记得我在准备什么吗"),                 # 该记得考研
    ("查记忆", "我家猫叫什么来着，你记得不"),             # 该记得芝麻
    # ── 目标3：对莉娜某段经历深入追问 ──
    ("问莉娜", "你最近有遇到什么烦心事吗"),
    ("接续", "后来呢，解决了吗"),                        # 接续
    ("接续", "那你当时是什么心情"),                      # 继续接续
    # ── 收尾 ──
    ("寒暄", "今天聊得挺开心的"),
    ("告别", "那我去复习了，改天再聊"),
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(TURNS)
    c = httpx.Client(timeout=60)
    c.post(f"{BASE}/api/register", json={"username": "eval100u", "password": "test123456"},
           headers={"X-Client-Id": CID})
    c.post(f"{BASE}/api/auth", json={"api_key": "0"}, headers={"X-Client-Id": CID})
    # 循环 TURNS 直到 n 轮
    i = 0
    while i < n:
        tag, msg = TURNS[i % len(TURNS)]
        try:
            r = c.post(f"{BASE}/api/chat", json={"message": msg, "session_id": SID},
                       headers={"X-Client-Id": CID})
            d = r.json()
            reply = d.get("reply", "").replace("\n", " ")
            plan = d.get("plan", {})
            print(f"[{i+1:03d}|{tag}] 用户: {msg}", flush=True)
            print(f"        莉娜: {reply[:110]}", flush=True)
        except Exception as e:
            print(f"[{i+1:03d}] 失败: {e}", flush=True)
        i += 1
        time.sleep(0.3)
    print("\n评测完成", flush=True)


if __name__ == "__main__":
    main()
