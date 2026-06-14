#!/usr/bin/env python3
"""清洗 self_facts：删掉「一次性剧情/具体遗物」类的伪经历，保留长期稳定事实。

默认 dry-run（只打印会删什么，不落盘）。加 --apply 才真正写回（先备份）。
判定：含「一次性遭遇」特征词的条目删除；明确的长期设定保留。
"""
from __future__ import annotations
import json, re, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SF_DIR = ROOT / "users" / "self_facts"

# 一次性剧情/具体遗物的特征：动作词 + 具体物件。命中即视为「伪经历」删除。
DROP_PATTERNS = [
    r"撬开", r"翻出", r"翻过来", r"收到一?块", r"送来", r"鉴定了?一?(块|枚|个|件)",
    r"盒子", r"铜币", r"铜牌", r"铜片", r"铁片", r"陶片", r"石板", r"木牌", r"铁皮",
    r"刻(着|有).{0,6}(字|符号|缩写|阿波罗|桃源)", r"发现一?枚",
]
DROP_RE = re.compile("|".join(DROP_PATTERNS))

# 即使含上面词，但属于长期设定的，保留（白名单）。
KEEP_RE = re.compile(r"常年|常去|窗外.*香草|一架|总是|平时")


def clean_facts(facts: dict) -> tuple[dict, list]:
    removed = []
    out = {}
    for bucket, items in (facts or {}).items():
        kept = []
        for it in (items or []):
            s = str(it)
            if DROP_RE.search(s) and not KEEP_RE.search(s):
                removed.append(f"{bucket}: {s}")
            else:
                kept.append(it)
        if kept:
            out[bucket] = kept
    return out, removed


def main() -> int:
    apply = "--apply" in sys.argv
    if not SF_DIR.exists():
        print("没有 self_facts 目录"); return 0
    total_removed = 0
    for f in sorted(SF_DIR.glob("*.json")):
        try:
            facts = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        cleaned, removed = clean_facts(facts)
        if not removed:
            continue
        total_removed += len(removed)
        print(f"\n### {f.name}  将删 {len(removed)} 条：")
        for r in removed:
            print(f"   ✗ {r}")
        if apply:
            shutil.copy(f, f.with_suffix(".json.bak"))  # 备份
            f.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 50)
    if apply:
        print(f"✅ 已清洗，共删 {total_removed} 条（原文件已备份为 .json.bak）")
    else:
        print(f"DRY-RUN：共会删 {total_removed} 条。确认无误后加 --apply 落盘。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
