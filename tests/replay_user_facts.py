#!/usr/bin/env python3
"""用真实对话记录重放，测「用户事实清单」的提炼+检索。

把已有的 (user, assistant) 对当历史灌进一个 Conversation，按 history_window
模拟「滑出窗口」，对滑出的几轮调 update_user_facts 累积清单（和 web 真实路径
一致），最后打印清单 + 几个检索查询，看记没记住用户讲的事。

不重新生成回复（用记录里的原回复当历史），只测记忆提炼/检索这环。
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for raw in (ROOT / ".env").read_text().splitlines():
    s = raw.strip()
    if s and not s.startswith("#") and "=" in s:
        k, v = s.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.config import resolve_controller_settings
from app.controller import build_default_controller
from app.user_facts import UserFactsStore

HISTORY_WINDOW = 30  # 和默认一致

def main() -> int:
    pairs = json.loads(Path("/tmp/replay_pairs.json").read_text(encoding="utf-8"))
    cfg = resolve_controller_settings()
    print(f"controller={cfg['model']}  重放 {len(pairs)} 轮")
    c = build_default_controller(api_key=cfg["api_key"], model=cfg["model"],
                                 base_url=cfg["base_url"], provider=cfg["provider"])
    facts: dict = {}
    # 逐轮推进：当对话长度超过窗口，最旧的那一轮"滑出"→提炼
    for i in range(len(pairs)):
        depth = i + 1
        if depth > HISTORY_WINDOW:
            slid = [pairs[depth - HISTORY_WINDOW - 1]]  # 刚滑出的那一轮
            updated = c.update_user_facts_sync(dict(facts), slid)
            if updated is not None:
                facts = updated
        if depth % 20 == 0:
            print(f"  轮 {depth}: 清单 {sum(len(v) for v in facts.values())} 条", flush=True)

    print("\n" + "=" * 60)
    print("【最终用户事实清单】")
    for b, items in facts.items():
        for it in items:
            print(f"  [{b}] {it}")
    print("=" * 60)

    print("\n【检索测试】莉娜被问到时能不能想起：")
    for q in ["你还记得我叫什么名字吗", "我是做什么工作的", "我在研究什么", "我有没有跟你说过我的感情/网聊"]:
        hit = UserFactsStore.search(facts, q, k=5)
        print(f"  问『{q}』→")
        print("    " + (hit.replace("\n", "\n    ") if hit else "（没检索到）"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
