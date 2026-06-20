"""重放 /root/lina/sess-3ce3c405.json 的真实用户对话到 lina_evermemos(8080)，
看长对话后的反应（长期记忆 / 人设 / 日记调用 / 是否降智/张冠李戴）。

只喂真实用户发言（过滤掉「（系统提示…」那些运行时注入的消息）。
"""
from __future__ import annotations
import json, sys, time
import httpx

SESS = "/root/lina/sess-3ce3c405.json"
BASE = "http://127.0.0.1:8080"
CID = "replay_sibyllina"
SID = "replay1"

def main():
    d = json.load(open(SESS))
    users = [m["content"] for m in d["messages"]
             if m.get("role") == "user" and not m["content"].startswith("（系统提示")]
    print(f"真实用户发言 {len(users)} 条，开始重放\n", flush=True)

    c = httpx.Client(timeout=60)
    # 注册+连 key
    c.post(f"{BASE}/api/register", json={"username": "replay_sib", "password": "test123456"},
           headers={"X-Client-Id": CID})
    c.post(f"{BASE}/api/auth", json={"api_key": "0"}, headers={"X-Client-Id": CID})

    for i, u in enumerate(users, 1):
        try:
            r = c.post(f"{BASE}/api/chat",
                       json={"message": u, "session_id": SID},
                       headers={"X-Client-Id": CID})
            d2 = r.json()
            reply = d2.get("reply", "")
            plan = d2.get("plan", {})
            nd = plan.get("need_diary")
            uf = plan.get("user_farewell")
            print(f"[{i:02d}] 用户: {u[:40]}", flush=True)
            print(f"     莉娜: {reply[:80]}  [diary={nd} farewell={uf}]", flush=True)
        except Exception as e:
            print(f"[{i:02d}] 失败: {e}", flush=True)
        time.sleep(0.3)
    print("\n重放完成", flush=True)

if __name__ == "__main__":
    main()
