#!/usr/bin/env python3
"""端到端：建好用户清单后，新会话里聊 20+ 轮、自然引用之前讲过的事，
看莉娜每轮能不能凭记忆正常应答。打印完整对话供人工核验。

清单来自之前 sess-3ce3c405 真实对话提炼的事实。controller 走 Claude。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for raw in (ROOT / ".env").read_text().splitlines():
    s = raw.strip()
    if s and not s.startswith("#") and "=" in s:
        k, v = s.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.config import resolve_controller_settings, resolve_api_key, STATIC_DIR
from app.controller import build_default_controller
from app.character import CharacterEngine, DEFAULT_MODEL
from app.conversation import Conversation

# 之前真实对话提炼出的用户清单
USER_FACTS = {
    "身份": ["名字叫 Sibyllina，妈妈起的", "在做科研工作"],
    "经历": ["在研究主动流式视频场景下的 AI 长期记忆"],
    "喜好": ["喜欢接触性格各异的女生", "爱讲笑话逗人玩"],
    "厌恶": ["反感直系领导苛刻的工作环境", "反感父母逆来顺受的处世方式"],
    "关系": ["与父亲最近闹矛盾"],
    "近况": ["同时与 100 多个女生聊天被发现过", "有女生说他是渣男", "最近压力很大，主要是人际关系", "需要维系和老板的关系"],
    "约定": ["想帮人类实现 AGI", "莉娜说他有烦恼可以随时找她倾诉"],
}

# 20+ 轮，自然地把旧事一点点带出来（不是生硬考问）
TURNS = [
    "嘿，好久没找你聊了",
    "你还记得我叫什么吗",
    "哈哈对，咱俩名字还挺有缘的",
    "我最近还是忙科研，头都大了",
    "你还记得我具体研究啥方向不",
    "对，就是让模型能记住长期的东西，跟你现在这个有点像",
    "最近压力是真大，主要不是技术，是人际那摊子事",
    "你猜我为啥人际关系搞得这么累",
    "唉，我跟我爸最近也别扭",
    "他那套逆来顺受的活法我真受不了",
    "工作上老板也烦，天天苛刻要求",
    "对了你还记得我那点感情破事吗",
    "嗐，就那个被说渣男的事呗",
    "我也不是渣，就是喜欢认识不同的女生嘛",
    "你说我是不是有点贪心",
    "算了不说这个了，给我讲个笑话呗",
    "我也爱讲笑话，逗人开心",
    "说回正事，我那个 AGI 的理想你还记得吗",
    "对，想让全世界都投身 AI 浪潮",
    "今天聊得挺开心，谢谢你还记得我这些事",
    "那我先去搬砖了，回头再找你倾诉",
]


def main() -> int:
    cfg = resolve_controller_settings()
    ctrl = build_default_controller(api_key=cfg["api_key"], model=cfg["model"],
                                    base_url=cfg["base_url"], provider=cfg["provider"])
    eng = CharacterEngine(api_key=resolve_api_key(), static_dir=str(STATIC_DIR),
                          model=DEFAULT_MODEL, controller=ctrl)
    conv = Conversation(session_id="e2e")
    print(f"controller={cfg['model']}  共 {len(TURNS)} 轮\n" + "=" * 64)
    for i, msg in enumerate(TURNS, 1):
        try:
            r = eng.chat(conv, msg, user_facts=USER_FACTS)
            reply = r.text.replace("\n", " ")
        except Exception as e:
            reply = f"[出错: {e}]"
        print(f"\n[{i:02d}] 用户：{msg}")
        print(f"     莉娜：{reply}")
    print("\n" + "=" * 64)
    print("人工核验点：名字/研究方向/感情(渣男)/父亲矛盾/老板/AGI理想/讲笑话 — 是否被正确记起、不编造")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
