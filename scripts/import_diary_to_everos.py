"""把莉娜的日记导入本地 EverOS 的 agent 轨道（agent_id=lina_diary）。

每条"记忆"（`##### 记忆｜M2026-01-01-01｜标题`）→ 一个 agent_skill：
- name        = diary_M2026-01-01-01（ID 里的 - 在 skill 名里保留，安全）
- description  = 记忆标题
- content      = 日期 + 事实骨架/正文 + 情绪 + 地点/人物 +「话题: <话题ID列表>」
                 （话题ID 写进正文做可检索标签——按话题ID检索日记靠它命中）

检索链路（设计见 docs/日记谈资接入EverOS设计.md）：
  当前对话 → 检索话题卡库(lina_topic) → 话题ID → 按话题ID检索日记库(lina_diary)

写 md → 本地 EverOS cascade 用 embedding 建索引，**零 LLM**（已验证）。

用法：
    python scripts/import_diary_to_everos.py --root /root/lina_everos_data \
        --agent-id lina_diary --diary ./diary/日记
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

APP_ID = "default"
PROJECT_ID = "default"

# 一条记忆的标题行：##### 记忆｜M2026-01-01-01｜整理年初的废旧纸张与账目
MEM_RE = re.compile(r"(?m)^#{4,5}\s*记忆｜(?P<id>M\d{4}-\d{2}-\d{2}-\d{2})｜(?P<title>.+?)\s*$")
# 话题卡引用：`A02-04|最近小变化|...`
TOPIC_RE = re.compile(r"`([A-Z]\d{2}-\d{2})\|")
# 想保留进 content 的字段（检索价值高的）。
KEEP_FIELDS = ("事实骨架", "正文", "当时情绪", "记忆类型", "地点", "涉及人物", "时间段")


def _field(block: str, name: str) -> str:
    """从一条记忆的文本块里抽 `- **字段**：值`（值可能跨行到下一个 `- **`）。"""
    m = re.search(rf"- \*\*{name}\*\*：\s*(.*?)(?=\n\s*- \*\*|\n#{{1,5}} |\Z)", block, re.S)
    return m.group(1).strip() if m else ""


def parse_memories(md_text: str, date: str) -> list[dict]:
    """把一篇日记拆成记忆条目列表。"""
    out: list[dict] = []
    # 按记忆标题切块（MULTILINE 让 ^ 匹配每行行首）
    parts = re.split(r"(?m)(?=^#{4,5}\s*记忆｜)", md_text)
    if len(parts) <= 1:
        parts = [md_text]
    for part in parts:
        m = MEM_RE.search(part)
        if not m:
            continue
        mid, title = m.group("id"), m.group("title")
        topics = sorted(set(TOPIC_RE.findall(part)))
        fields = {f: _field(part, f) for f in KEEP_FIELDS}
        fields = {k: v for k, v in fields.items() if v}
        out.append({"id": mid, "title": title, "date": date, "topics": topics, "fields": fields})
    return out


def _skill_md(agent_id: str, mem: dict) -> str:
    name = f"diary_{mem['id']}"
    desc = mem["title"].replace("\n", " ").strip()[:80]
    # 正文：标题 + 各字段 + 话题标签（话题ID 供「按话题检索日记」命中）
    lines = [f"# {mem['date']} {mem['title']}", ""]
    for k in ("事实骨架", "正文", "当时情绪", "地点", "涉及人物", "记忆类型", "时间段"):
        if mem["fields"].get(k):
            lines.append(f"{k}：{mem['fields'][k]}")
    if mem["topics"]:
        lines.append("")
        lines.append(f"话题: {' '.join(mem['topics'])}")
    body = "\n".join(lines)
    return (
        "---\n"
        "type: agent_skill\n"
        f"agent_id: {agent_id}\n"
        f"name: {name}\n"
        f"description: {desc}\n"
        "confidence: 1.0\n"
        "maturity_score: 1.0\n"
        "---\n\n"
        f"{body}\n"
    )


def import_diary(root: Path, agent_id: str, diary_dir: Path) -> tuple[int, int]:
    skills_root = root / APP_ID / PROJECT_ID / "agents" / agent_id / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    files = sorted(diary_dir.rglob("*.md"))
    n_mem = 0
    for fpath in files:
        # 日期取文件名 2026-01-01
        date = fpath.stem
        mems = parse_memories(fpath.read_text(encoding="utf-8"), date)
        for mem in mems:
            skill_dir = skills_root / f"skill_diary_{mem['id']}"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(_skill_md(agent_id, mem), encoding="utf-8")
            n_mem += 1
    return len(files), n_mem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/lina_everos_data")
    ap.add_argument("--agent-id", default="lina_diary")
    ap.add_argument("--diary", default="./diary/日记")
    args = ap.parse_args()
    root, diary_dir = Path(args.root), Path(args.diary)
    print(f"导入日记 → {root}/{APP_ID}/{PROJECT_ID}/agents/{args.agent_id}/skills/")
    n_files, n_mem = import_diary(root, args.agent_id, diary_dir)
    print(f"\n完成：{n_files} 篇日记 → {n_mem} 条记忆 skill。cascade 自动索引（稍等）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
