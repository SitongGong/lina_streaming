#!/usr/bin/env python3
"""对比：原 20 路并发 advisor vs 新 4 组合并。
50 条复杂输入，各跑两版，比速度（avg/P50/P95）+ 关键字段一致率。
强制走 LLM（屏蔽规则层），纯测 fan-out 那段。controller=Claude。
"""
from __future__ import annotations
import os, sys, time, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for raw in (ROOT / ".env").read_text().splitlines():
    s = raw.strip()
    if s and not s.startswith("#") and "=" in s:
        k, v = s.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.config import resolve_controller_settings
from app.controller.controller import build_default_controller
from app.controller.merged_controller import MergedController
from app.controller.schema import LinaTurnContext as C

# 50 条偏复杂/歧义/多意图的输入（最能体现合并是否降准）
INPUTS = [
    "我最近压力好大，我那只猫团子也病了，唉",
    "你能帮我查查这个古代符文是什么意思吗",
    "你好啊，对了我之前跟你说的换工作的事我定了",
    "我升职了！不过手底下的人难带，头疼",
    "你还记得我研究啥方向不，就那个AI记忆的",
    "讲讲你自己吧，还有你们那个世界长啥样",
    "我怎么缓解焦炉啊，最近总睡不着",
    "拜拜啦，对了你上次说的那出戏剧叫啥来着",
    "炼金术到底靠不靠谱，能炼出金子吗",
    "我同时和好几个女生聊天被发现了，被骂渣男",
    "你帮我用手机导航去最近的咖啡馆呗",
    "我妈又催婚了，跟我爸吵了一架",
    "薄荷和迷迭香种一起会不会打架",
    "你是不是个AI啊，怎么啥都懂",
    "今天面试搞砸了，感觉自己一无是处",
    "你最爱哪部悲剧，我最近迷上希腊神话了",
    "我跟你说过我在准备考研对吧，进度有点慢",
    "随便聊聊呗，你今天心情咋样",
    "那块烧黑的铁片你到底看出什么名堂没",
    "我老家成都的，你知道成都吗",
    "我觉得自己什么都做不好，好累",
    "你能写段代码帮我自动回消息吗",
    "上次那只咬笔的猫，现在还咬吗",
    "你帮我搜下附近有什么好吃的",
    "我研究让模型有长期记忆，跟你现在挺像的",
    "唉，人际关系真比技术难搞多了",
    "你怕什么东西吗，我特别怕打雷",
    "我最近顺利多了，谢谢你之前的建议",
    "古希腊那些羊皮卷上的字你认得几个",
    "你知道ChatGPT吗，最近特别火",
    "我跟我爸那套逆来顺受的活法真合不来",
    "我就叫Sibyllina，我妈给起的，巧不巧",
    "你帮我把这段话翻译成英文",
    "最近老板苛刻得很，天天加班",
    "你研究遗物的时候有没有遇到过怪事",
    "我想帮人类实现AGI，让全世界都搞AI",
    "心好累，不想说话，就想找你倾诉下",
    "你说香草茶怎么泡才好喝",
    "我和那个同事的破事终于了结了",
    "你还记得我是谁吗，咱俩还同名呢",
    "戏剧里你最喜欢哪段台词",
    "我准备周末连约三个会，精力旺盛得很",
    "你能控制我电脑帮我关个机吗",
    "我研究的是主动流式视频场景下的记忆",
    "唉感觉做什么都没意思，提不起劲",
    "你工房里最奇怪的遗物是啥",
    "我喜欢认识各种性格的女生，算贪心吗",
    "拜拜，今天聊得挺开心",
    "你帮我留意下治猫毛过敏的法子呗",
    "嗯嗯，那个魔法石真能炼出来吗",
]

KEY_FIELDS = ["module_user_vent", "module_world_immersion", "module_action_boundary",
              "module_relationship_recall", "module_self_introspection",
              "use_self_facts", "suppress_trailing_question", "lenient_typos"]


def sig(p):
    return tuple(getattr(p, f) for f in KEY_FIELDS)


def main() -> int:
    cfg = resolve_controller_settings()
    orig = build_default_controller(api_key=cfg["api_key"], model=cfg["model"],
                                    base_url=cfg["base_url"], provider=cfg["provider"])
    merged = MergedController(openai_client=orig._client, model_name=cfg["model"],
                              advisor_timeout=orig._advisor_timeout)
    orig._rule_router.route = lambda c: None
    merged._rule_router.route = lambda c: None

    t_orig, t_merged, agree = [], [], 0
    for i, text in enumerate(INPUTS, 1):
        ctx = C(user_text=text)
        s = time.monotonic(); po = orig.dispatch_sync(ctx); t_orig.append(time.monotonic() - s)
        s = time.monotonic(); pm = merged.dispatch_sync(ctx); t_merged.append(time.monotonic() - s)
        same = sig(po) == sig(pm)
        if same:
            agree += 1
        print(f"[{i:02d}] orig={t_orig[-1]:4.1f}s merged={t_merged[-1]:4.1f}s {'一致' if same else '差异:'+text[:18]}", flush=True)

    def stats(xs):
        xs = sorted(xs)
        return (statistics.mean(xs), xs[len(xs)//2], xs[int(len(xs)*0.95)])
    ao, mo, po = stats(t_orig)
    am, mm, pm_ = stats(t_merged)
    print("\n" + "=" * 60)
    print(f"原 20 并发 ：avg {ao:.2f}s  P50 {mo:.2f}s  P95 {po:.2f}s")
    print(f"新 4 合并 ：avg {am:.2f}s  P50 {mm:.2f}s  P95 {pm_:.2f}s")
    print(f"提速：avg {(1-am/ao)*100:+.0f}%  P95 {(1-pm_/po)*100:+.0f}%")
    print(f"关键字段一致率：{agree}/{len(INPUTS)} = {agree/len(INPUTS)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
