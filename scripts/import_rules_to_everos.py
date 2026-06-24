"""把行为规则 / 情绪格式按小节拆块导入本地 EverOS（agent 轨道 lina_rules）。

和人设(import_persona_to_everos.py)走完全相同的检索机制：md 按小节拆 chunk →
写成 agent_skill → cascade 用 embedding 建索引 → 运行时按当前用户话语义检索 top-k
条相关规则注入，而不是把整份 behavior_rules / mood_format_spec 全量塞进 prompt。

拆分对象：
  - prompts/main/behavior_rules.txt   —— 按 `### x.` 小节拆（每条规则一块）
  - prompts/main/mood_format_spec.txt —— 情绪表按「- 情绪词」拆（每个情绪一块）+ 规则头

用法：
  python scripts/import_rules_to_everos.py \
      --root /root/lina_everos_data --agent-id lina_rules
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_ID = "default"
PROJECT_ID = "default"
MAIN_DIR = Path(__file__).resolve().parent.parent / "prompts" / "main"


def _safe_name(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^\w一-鿿]+", "_", s)
    return s.strip("_")[:60] or "x"


def _skill_md(agent_id: str, name: str, description: str, content: str) -> str:
    desc = description.replace("\n", " ").strip()[:80]
    return (
        "---\n"
        "type: agent_skill\n"
        f"agent_id: {agent_id}\n"
        f"name: {name}\n"
        f"description: {desc}\n"
        "confidence: 1.0\n"
        "maturity_score: 1.0\n"
        "---\n\n"
        f"{content}\n"
    )


def _split_behavior(text: str) -> list[tuple[str, str]]:
    """按 `### ` 小节拆。返回 [(heading, chunk_text), ...]。"""
    blocks = re.split(r"(?m)^(?=### )", text)
    out = []
    for b in blocks:
        b = b.strip()
        if len(b) < 30:
            continue
        head = b.split("\n", 1)[0].lstrip("# ").strip()[:40]
        out.append((head, b))
    return out


def _split_mood(text: str) -> list[tuple[str, str]]:
    """情绪格式：规则头(到「情绪表」前)整块 + 情绪表里每个「- 情绪词」一块。"""
    out = []
    # 规则头（格式说明 + 三条铁律）整块保留——这是机制性的，每条情绪检索时也用得上。
    head_m = re.split(r"\*\*情绪表\*\*", text, maxsplit=1)
    if head_m and head_m[0].strip():
        out.append(("情绪标记格式与铁律", head_m[0].strip()))
    table = head_m[1] if len(head_m) > 1 else ""
    # 情绪表按二级「- 词」条目拆（每个情绪词 + 其说明一块）。
    for m in re.split(r"(?m)^(?=\s*-\s+\S)", table):
        m = m.strip()
        if len(m) < 20:
            continue
        head = re.sub(r"^[-\s]+", "", m.split("\n", 1)[0]).strip()[:30]
        out.append((f"情绪·{head}", m))
    return out


def import_rules(root: Path, agent_id: str) -> int:
    skills_root = root / APP_ID / PROJECT_ID / "agents" / agent_id / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    written = 0
    sources = [
        ("behavior", MAIN_DIR / "behavior_rules.txt", _split_behavior),
        ("mood", MAIN_DIR / "mood_format_spec.txt", _split_mood),
    ]
    for stem, fpath, splitter in sources:
        if not fpath.exists():
            print(f"  跳过（不存在）：{fpath}")
            continue
        chunks = splitter(fpath.read_text(encoding="utf-8"))
        for i, (heading, ctext) in enumerate(chunks):
            name = _safe_name(f"{stem}_{heading}_{i}")
            description = f"{stem} · {heading}"
            skill_dir = skills_root / f"skill_{name}"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                _skill_md(agent_id, name, description, ctext), encoding="utf-8"
            )
            written += 1
        print(f"  {fpath.name}: {len(chunks)} 块 → skill")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/lina_everos_data")
    ap.add_argument("--agent-id", default="lina_rules")
    args = ap.parse_args()
    root = Path(args.root)
    print(f"导入规则 → {root}/{APP_ID}/{PROJECT_ID}/agents/{args.agent_id}/skills/")
    n = import_rules(root, args.agent_id)
    print(f"\n完成：写入 {n} 个 agent_skill。cascade watcher 会自动索引（约几秒）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
