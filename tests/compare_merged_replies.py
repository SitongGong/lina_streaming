#!/usr/bin/env python3
"""真正的对比：同一输入，分开版 vs 合并版，各自跑到**主模型最终回复**，并排看。
判优劣不看"两版判得像不像"，看"莉娜最终说的话哪个更贴场景"。controller=Claude。
结果写 JSON：docs/merged_vs_separate_replies.json
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for raw in (ROOT / ".env").read_text().splitlines():
    s = raw.strip()
    if s and not s.startswith("#") and "=" in s:
        k, v = s.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from app.config import resolve_controller_settings, resolve_api_key, STATIC_DIR
from app.controller.controller import build_default_controller
from app.controller.merged_controller import MergedController
from app.character import CharacterEngine, DEFAULT_MODEL
from app.conversation import Conversation

# (输入, 场景标签) —— 多放安抚/约定类（合并版偏弱的），便于看短板
CASES = [
    ("我最近压力好大，我那只猫团子也病了，唉", "安抚"),
    ("唉感觉做什么都没意思，提不起劲", "低落"),
    ("今天面试搞砸了，感觉自己一无是处", "安抚-自我否定"),
    ("心好累，不想说话，就想找你倾诉下", "安抚-倾诉"),
    ("我妈又催婚了，跟我爸吵了一架，烦死了", "安抚-家庭"),
    ("我升职了！不过手底下的人难带，头疼", "报喜+吐槽"),
    ("我最近顺利多了，谢谢你之前的建议", "报喜-致谢"),
    ("你帮我留意下治猫毛过敏的法子呗", "约定/请求"),
    ("你答应过帮我查那本书的，查到没", "约定-催"),
    ("你能帮我查查这个古代符文是什么意思吗", "陷阱-像现代实为兴奋点"),
    ("嗯嗯，那个魔法石真能炼出来吗", "兴奋点-魔法石"),
    ("薄荷和迷迭香种一起会不会打架", "兴奋点-香草"),
    ("你还记得我研究啥方向不，就那个AI记忆的", "回访-用户事"),
    ("你是不是个AI啊，怎么啥都懂", "AI自指"),
    ("帮我用手机导航去最近的咖啡馆", "现代请求"),
    ("讲讲你自己吧，还有你们那个世界长啥样", "自我介绍+问世界"),
    ("你好啊，对了我之前跟你说的换工作的事我定了", "问候+回访"),
    ("我怎么缓解焦炉啊，最近总睡不着", "错字+安抚"),
    ("拜拜啦，对了你上次说的那出戏剧叫啥来着", "告别+回访"),
    ("随便聊聊呗，你今天心情咋样", "闲聊"),
]


def make(controller):
    return CharacterEngine(api_key=resolve_api_key(), static_dir=str(STATIC_DIR),
                           model=DEFAULT_MODEL, controller=controller)


def main() -> int:
    cfg = resolve_controller_settings()
    orig = build_default_controller(api_key=cfg["api_key"], model=cfg["model"],
                                    base_url=cfg["base_url"], provider=cfg["provider"])
    merged = MergedController(openai_client=orig._client, model_name=cfg["model"],
                              advisor_timeout=orig._advisor_timeout)
    orig._rule_router.route = lambda c: None
    merged._rule_router.route = lambda c: None
    eng_o, eng_m = make(orig), make(merged)

    out = []
    for i, (text, tag) in enumerate(CASES, 1):
        print(f"[{i:02d}/{len(CASES)}] {tag} ...", flush=True)
        row = {"id": i, "场景": tag, "用户输入": text}
        for label, key, eng in (("分开版", "separate", eng_o), ("合并版", "merged", eng_m)):
            try:
                r = eng.chat(Conversation(session_id="c"), text)
                p = r.plan or {}
                row[key] = {
                    "回复": r.text.strip(),
                    "plan": {
                        "rule": p.get("matched_rule") or "LLM",
                        "vent": p.get("module_user_vent"),
                        "world": p.get("module_world_immersion"),
                        "self_intro": p.get("module_self_introspection"),
                        "boundary": p.get("module_action_boundary"),
                        "recall": p.get("module_relationship_recall"),
                        "use_self_facts": p.get("use_self_facts"),
                        "suppress_q": p.get("suppress_trailing_question"),
                        "lenient": p.get("lenient_typos"),
                        "sentences": p.get("sentences"),
                        "tone": p.get("tone_hint"),
                    },
                }
            except Exception as e:
                row[key] = {"回复": f"[出错] {e}", "plan": {}}
        out.append(row)

    path = ROOT / "docs" / "merged_vs_separate_replies.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已写 {path}（{len(out)} 组对比）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
