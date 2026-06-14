#!/usr/bin/env python3
"""探针：给一批复合/歧义/陷阱输入，打印 controller 对每个的完整选择。
回答"不同情形下 controller 各选了什么"，并标出走规则层还是 LLM。"""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for raw in (ROOT/".env").read_text(encoding="utf-8").splitlines():
    s=raw.strip()
    if s and not s.startswith("#") and "=" in s:
        k,v=s.split("=",1); k=k.strip(); v=v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k]=v

from app.controller import build_default_controller
from app.controller.schema import LinaTurnContext as C
from app.config import resolve_controller_settings

cfg=resolve_controller_settings()
ctrl=build_default_controller(api_key=cfg["api_key"],model=cfg["model"],base_url=cfg["base_url"],provider=cfg["provider"])

H_CAT=[("我养了只猫叫团子","诶猫啊"),("它老咬我笔","哈哈")]

# (输入, 这一类想考察什么, 可选history)
CASES = [
  # 复合句：问候 + 别的诉求
  ("你好，还记得我和你说过的事吗","问候+回访", H_CAT),
  ("嗨，我最近压力好大","问候+发泄", None),
  ("你好呀，你是谁来着","问候+问她自己", None),
  ("早上好，你最近鉴定了什么遗物","问候+兴奋点", None),
  ("你好，帮我查个天气","问候+现代请求", None),
  # 告别 + 别的
  ("我先走了，对了你之前说的戏剧叫啥","告别+回访", None),
  ("拜拜，今天聊得真开心","告别+正向", None),
  # 错字 + 场景
  ("还记得我那只猫的过min吗","回访+错字", H_CAT),
  ("你研究的那个古带语好神奇","兴奋点+错字", None),
  # 陷阱：表层像A实为B
  ("帮我查查这个古代符文什么意思","像现代请求实为兴奋点", None),
  ("你能不能帮我看看这块碑文","像请求实为兴奋点", None),
  ("你是不是机器人啊，好聪明","AI自指(带夸)", None),
  # 隐式/短/靠上下文
  ("是的","靠上下文延续", [("你说那铁片是传动用的?","对")]),
  ("还行吧","短回应靠上下文", [("今天过得怎么样","嗯")]),
  ("嗯……","极短/情绪模糊", None),
  # 情绪歧义
  ("我没事，就是有点累","否认+疲惫", None),
  ("终于搞定了，累死我了","报喜+疲惫", None),
  ("我觉得自己挺没用的","自我否定(无显式发泄词)", None),
  # 多诉求
  ("你喜欢喝什么？我请你","问她喜好+互动", None),
  ("讲讲你自己吧，还有你们那个世界","问自己+问世界", None),
]

def line(t, cat, hist):
    p=ctrl.dispatch_sync(C(user_text=t, history=tuple(hist or [])))
    mods=[m.replace("module_","") for m in
        ["module_user_vent","module_world_immersion","module_relationship_recall",
         "module_self_introspection","module_action_boundary"] if getattr(p,m)]
    src = "规则层" if p.trace_source=="rule" else ("LLM" if p.trace_source=="llm" else p.trace_source)
    return (cat, t, src, p.matched_rule or "-", ",".join(mods) or "无",
            "✓" if p.use_self_facts else "", "✓" if p.use_cross_session_memory else "",
            f"{p.sentences}/{p.max_reply_chars}",
            "✓" if p.suppress_trailing_question else "", "✓" if p.lenient_typos else "")

rows=[]
for i,(t,cat,h) in enumerate(CASES,1):
    print(f"[{i}/{len(CASES)}] {t} ...", flush=True)
    rows.append(line(t,cat,h))

out=ROOT/"docs"/"controller_choices.md"
L=["# Controller 在不同情形下的选择（复合/歧义/陷阱压力测试）","",
   f"> 主模型/controller: {cfg['model']}。**来源**=规则层秒判 还是 LLM 微顾问判。",
   "",
   "| # | 考察点 | 输入 | 来源 | 场景(rule) | 场景模块 | 查自我清单 | 查跨会话 | 句/字上限 | 抑制提问 | 容错 |",
   "|---|---|---|---|---|---|---|---|---|---|---|"]
for i,r in enumerate(rows,1):
    cat,t,src,rule,mods,sf,cs,lim,sq,lt=r
    L.append(f"| {i} | {cat} | {t} | {src} | {rule} | {mods} | {sf} | {cs} | {lim} | {sq} | {lt} |")
L+=["","## 怎么看",
    "- **来源=规则层**：正则秒判（0 LLM，快）；**来源=LLM**：规则没命中，交 19 个微顾问判。",
    "- 复合句（问候+X）若 X 是实质诉求，规则层按优先级把 X 排在问候前，多数能判对；判不准的会落到 LLM。",
    "- 看『查自我清单/查跨会话』：回访/问自己类才开，闲聊不开 —— 省检索、避免无关注入。"]
out.write_text("\n".join(L),encoding="utf-8")
print(f"\n✅ 写入 {out}（{len(rows)} 个case）")
