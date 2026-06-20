"""把【写完整的】话题卡导入本地 EverOS（agent_id=lina_topic）。

重构（2026-06-18）：从「按日记反推三段ID」改为「直接解析话题卡文件」。
- **只导带「日记索引」的完整卡**（约 213 个）；空壳卡（没写完）不导入、不检索。
- 解析话题卡全字段：适用场景/话题功能/信任门槛/触发方式/开场句/可追问/莉娜观点/
  共情目标/共情方式/暴露边界/可转场/日记索引(M2026-..)。
- content 里写全这些信息（检索命中后整张卡注入做谈资素材），并把【信任门槛】和
  【日记索引ID】写进 content（同时也放 frontmatter 冗余），供：
    · 主动发言按信任门槛过滤
    · 命中话题后按日记索引ID精确取日记（不再 vector 猜）

检索链路见 DiaryMemory：检索 lina_topic 命中卡 → 读卡里的日记索引ID → 精确取日记。

用法：
    python scripts/import_topics_to_everos.py --root /root/lina_everos_data \
        --agent-id lina_topic --cards "./diary/谈资索引"
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

APP_ID = "default"
PROJECT_ID = "default"

# 话题卡里要抽的字段（顺序即注入顺序）
FIELDS = [
    "适用场景", "话题功能", "触发方式", "世界观锚点",
    "开场句", "可追问", "莉娜观点", "共情目标", "莉娜可共鸣点",
    "用户可共鸣点", "共情方式", "暴露边界", "可转场",
]
_DIARY_ID_RE = re.compile(r"(M\d{4}-\d{2}-\d{2}-\d{2})")
_TRUST_RE = re.compile(r"信任\s*>?=?\s*(\d+)")
_TOPIC_ID_RE = re.compile(r"([A-Z]\d{2}-\d{2})")


def _field(text: str, name: str) -> str:
    """抽 `- **字段**：值`（值可跨行到下一个 `- **` 或 `####`）。"""
    m = re.search(rf"- \*\*{name}\*\*[：:]\s*(.*?)(?=\n\s*- \*\*|\n#{{1,5}} |\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def parse_card(fpath: str) -> dict | None:
    """解析一张话题卡。无「日记索引」=未写完，返回 None（不导入）。"""
    text = Path(fpath).read_text(encoding="utf-8")
    if "日记索引" not in text:
        return None
    # 话题ID 从路径取
    tid_m = _TOPIC_ID_RE.search(fpath)
    if not tid_m:
        return None
    tid = tid_m.group(1)
    # 标题（话题卡｜XXX）
    title_m = re.search(r"话题卡[｜|]\s*(.+)", text)
    title = title_m.group(1).strip() if title_m else fpath.split("/")[-1].replace(".md", "")
    # 信任门槛
    trust_m = _TRUST_RE.search(_field(text, "信任门槛") or text)
    trust = int(trust_m.group(1)) if trust_m else 1
    # 日记索引ID
    diary_idx_block = _field(text, "日记索引")
    diary_ids = _DIARY_ID_RE.findall(diary_idx_block)
    # 其它字段
    fields = {k: _field(text, k) for k in FIELDS}
    fields = {k: v for k, v in fields.items() if v}
    return {
        "tid": tid, "title": title, "trust": trust,
        "diary_ids": diary_ids, "fields": fields,
    }


def _safe_slug(s: str) -> str:
    s = re.sub(r"[^\w一-鿿]+", "_", s.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "x"


def _skill_md(agent_id: str, card: dict) -> str:
    # 同一话题ID下有多个角度卡（如 B02-04 有 7 个），name 必须带角度，否则互相覆盖。
    name = f"topic_{card['tid'].replace('-', '_')}_{_safe_slug(card['title'])}"
    desc = f"{card['tid']} {card['title']}"[:80]
    lines = [f"# {card['tid']} {card['title']}", ""]
    # 检索/过滤用的结构化信息（也写进正文，保证检索一定拿得到）
    lines.append(f"信任门槛: {card['trust']}")
    if card["diary_ids"]:
        lines.append(f"日记索引: {' '.join(card['diary_ids'])}")
    lines.append("")
    # 谈资素材（命中后注入，莉娜自由化用）
    for k in FIELDS:
        if card["fields"].get(k):
            lines.append(f"{k}：{card['fields'][k]}")
    body = "\n".join(lines)
    return (
        "---\n"
        "type: agent_skill\n"
        f"agent_id: {agent_id}\n"
        f"name: {name}\n"
        f"description: {desc}\n"
        f"trust_threshold: {card['trust']}\n"
        "confidence: 1.0\n"
        "maturity_score: 1.0\n"
        "---\n\n"
        f"{body}\n"
    )


def import_topics(root: Path, agent_id: str, cards_dir: str) -> tuple[int, int]:
    skills_root = root / APP_ID / PROJECT_ID / "agents" / agent_id / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{cards_dir}/**/*.md", recursive=True)
    n_card, n_skipped = 0, 0
    for f in files:
        if f.endswith("索引.md"):
            continue
        card = parse_card(f)
        if card is None:
            n_skipped += 1
            continue
        name = f"topic_{card['tid'].replace('-', '_')}_{_safe_slug(card['title'])}"
        skill_dir = skills_root / f"skill_{name}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(_skill_md(agent_id, card), encoding="utf-8")
        n_card += 1
    return n_card, n_skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/lina_everos_data")
    ap.add_argument("--agent-id", default="lina_topic")
    ap.add_argument("--cards", default="./diary/谈资索引")
    args = ap.parse_args()
    root = Path(args.root)
    print(f"导入完整话题卡 → {root}/{APP_ID}/{PROJECT_ID}/agents/{args.agent_id}/skills/")
    n, skipped = import_topics(root, args.agent_id, args.cards)
    print(f"\n完成：导入 {n} 个完整话题卡，跳过 {skipped} 个空壳卡。cascade 自动索引。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
