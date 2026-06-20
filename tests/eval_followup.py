"""专测：基于莉娜【自身某段具体经历】连续追问，看她能否锚定那段日记往下答、不跑偏不编。

设计：先用开放问题引出莉娜一段经历，然后**针对她答里提到的细节**层层追问
（用代词/省略，模拟真人追问），看连续几轮是否都在同一段经历上、细节一致。
"""
from __future__ import annotations
import sys, time
import httpx

BASE = "http://127.0.0.1:8080"
import time as _t
CID = "evalfu_%d" % int(_t.time())
SID = "evalfu_sess"

TURNS = [
    "你最近有没有出过远门，或者去首都那种大城市",   # 引出旅途/进城经历
    "进城的时候顺利吗",                              # 追问：盘查
    "那盘查官为难你了吗",                            # 追问细节
    "他具体说你什么了",                              # 追问更细：配方纸/符号
    "那你怎么应付过去的",                            # 追问：背培根头衔
    "进去之后住的地方怎么样",                        # 追问：培根寓所
    "那天晚上你是什么感觉",                          # 追问：情绪/身体
    "现在回想起来呢",                                # 追问：事后
]


def main():
    c = httpx.Client(timeout=60)
    c.post(f"{BASE}/api/register", json={"username": CID, "password": "test123456"},
           headers={"X-Client-Id": CID})
    c.post(f"{BASE}/api/auth", json={"api_key": "0"}, headers={"X-Client-Id": CID})
    for i, msg in enumerate(TURNS, 1):
        try:
            r = c.post(f"{BASE}/api/chat", json={"message": msg, "session_id": SID},
                       headers={"X-Client-Id": CID})
            reply = r.json().get("reply", "").replace("\n", " ")
            print(f"[{i}] 用户: {msg}", flush=True)
            print(f"    莉娜: {reply}", flush=True)
        except Exception as e:
            print(f"[{i}] 失败: {e}", flush=True)
        time.sleep(0.3)
    print("\n完成", flush=True)


if __name__ == "__main__":
    main()
