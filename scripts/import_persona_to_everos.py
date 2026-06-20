"""把莉娜的「可检索人设 md」导入本地 EverOS 的 agent 轨道（agent_skill）。

背景与设计
----------
莉娜的人设分两类（见 app/rag.py）：
- **核心人设** person_setup.md —— 永远全量进 prompt，**不导入**（人格底线，不能漂）。
- **可检索人设** world / personality / hobbies / others —— 原来靠 BM25 检索注入，
  现改为走本地 EverOS：写成 agent_skill md → cascade 用 embedding 建索引 → agent
  轨道语义检索。**全程不需要大 LLM**（cascade 只用 embedding，已实测）。

每个 md 按章节（chunk_markdown，与原 BM25 同款拆法）拆成多个 chunk，每个 chunk
写一个 agent_skill：
    <root>/<app>/<project>/agents/<agent_id>/skills/skill_<name>/SKILL.md

写完文件后 cascade watcher 会自动索引，无需调任何 API。

日记将来同理：把每天日记写成 agent_skill（或 episode）md 进同一 agent 轨道即可。

用法：
    python scripts/import_persona_to_everos.py \
        --root /root/lina_everos_data --agent-id lina --static ./static
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 复用 app 里现成的章节拆分（与原 BM25 检索完全一致的拆法）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag import chunk_markdown  # noqa: E402

# 要导入的可检索人设文件（person_setup.md 是 CORE，不在此列）。
PERSONA_FILES = ("world.md", "personality.md", "hobbies.md", "others.md")

# EverOS agent_skill 的固定目录形状。
APP_ID = "default"
PROJECT_ID = "default"


def _safe_name(raw: str) -> str:
    """把「文件名+章节」压成 filesystem/ID 安全的 snake_case skill 名。"""
    s = re.sub(r"[^\w一-鿿]+", "_", raw.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "section"


def _skill_md(agent_id: str, name: str, description: str, content: str) -> str:
    """构造一个 agent_skill SKILL.md 文本（frontmatter + 正文）。

    frontmatter 必填：type/agent_id/name/description/confidence/maturity_score
    （缺 agent_id 或 name，cascade 会拒绝索引）。人设是「确定的设定」，
    confidence/maturity 都给 1.0。
    """
    # description 不能换行，截断到一行。
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


def import_persona(root: Path, agent_id: str, static_dir: Path) -> int:
    skills_root = root / APP_ID / PROJECT_ID / "agents" / agent_id / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for fname in PERSONA_FILES:
        fpath = static_dir / fname
        if not fpath.exists():
            print(f"  跳过（不存在）：{fpath}")
            continue
        content = fpath.read_text(encoding="utf-8")
        chunks = chunk_markdown(content, source=fname)
        stem = fname.rsplit(".", 1)[0]  # world / personality / ...
        for i, ch in enumerate(chunks):
            # skill 名 = 文件名 + 章节标题（去重靠序号），保证唯一且可读。
            name = _safe_name(f"{stem}_{ch.heading}_{i}")
            description = f"{stem} · {ch.heading}"
            skill_dir = skills_root / f"skill_{name}"
            skill_dir.mkdir(parents=True, exist_ok=True)
            # 正文只放章节内容；来源/章节由 description 携带，检索端从 description
            # 还原成 Chunk.source/heading，render 时再加前缀（避免前缀重复）。
            (skill_dir / "SKILL.md").write_text(
                _skill_md(agent_id, name, description, ch.text), encoding="utf-8"
            )
            written += 1
        print(f"  {fname}: {len(chunks)} 个章节 → skill")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/lina_everos_data", help="EVEROS_MEMORY__ROOT")
    ap.add_argument("--agent-id", default="lina")
    ap.add_argument("--static", default="./static", help="人设 md 所在目录")
    args = ap.parse_args()

    root = Path(args.root)
    static_dir = Path(args.static)
    print(f"导入人设 → {root}/{APP_ID}/{PROJECT_ID}/agents/{args.agent_id}/skills/")
    n = import_persona(root, args.agent_id, static_dir)
    print(f"\n完成：写入 {n} 个 agent_skill。cascade watcher 会自动索引（约几秒）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
