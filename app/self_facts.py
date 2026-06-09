"""莉娜「自我事实清单」存储 —— per-user，跨会话共享。

记的是莉娜在对话里**临时说出来、人设文件里没写**的关于自己的稳定事实
（养了猫煤球 / 答应帮你查某本书 / 讨厌应酬 …），用来在历史滑出上下文窗口
后仍保持前后一致。与静态人设（personality.md 等）互补，不重复。

设计：
- 按类别分桶，每桶有条数上限 → 清单永远不会无限膨胀。
- 一个 user_id 一份 JSON（users/self_facts/<user_id>.json），跨会话共享。
- 更新由 controller LLM 概括产出（合并/覆盖/淘汰），这里只负责存取 + 裁剪。
- 纯标准库，进程内加锁。
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

# 允许的类别（桶）。LLM 概括时被要求只用这些键。
BUCKETS = ("身份", "喜好", "厌恶", "经历", "承诺", "态度", "近况")
# 每个桶最多留几条；超出时由 LLM 在概括时淘汰（这里再兜底裁剪）。
MAX_PER_BUCKET = 6
# 整份清单总条数上限（兜底）。
MAX_TOTAL = 30


def _safe_user_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]", "_", (raw or "").strip())[:64]


class SelfFactsStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, user_id: str) -> Path:
        return self.root / f"{_safe_user_id(user_id)}.json"

    def load(self, user_id: str) -> dict[str, list[str]]:
        """返回该用户的自我事实清单（分桶 dict）。不存在则空 dict。"""
        if not user_id:
            return {}
        p = self._path(user_id)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return self._clamp(data) if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save(self, user_id: str, facts: dict) -> None:
        if not user_id:
            return
        with self._lock:
            self._path(user_id).write_text(
                json.dumps(self._clamp(facts), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _clamp(facts: dict) -> dict[str, list[str]]:
        """兜底裁剪：只保留合法桶、每桶去重并截到上限、总数截到 MAX_TOTAL。
        （LLM 概括时也会控制，这里是第二道保险。）"""
        out: dict[str, list[str]] = {}
        total = 0
        for bucket in BUCKETS:
            items = facts.get(bucket)
            if not isinstance(items, list):
                continue
            seen: set[str] = set()
            kept: list[str] = []
            for it in items:
                s = str(it or "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                kept.append(s)
                if len(kept) >= MAX_PER_BUCKET:
                    break
            if kept:
                # 总数上限：超了就停止再加新桶内容
                room = MAX_TOTAL - total
                if room <= 0:
                    break
                kept = kept[:room]
                out[bucket] = kept
                total += len(kept)
        return out

    @staticmethod
    def _flatten(facts: dict[str, list[str]]) -> list[tuple[str, str]]:
        """展平成 [(bucket, item)]，保持 BUCKETS 顺序。"""
        out: list[tuple[str, str]] = []
        for bucket in BUCKETS:
            for it in (facts.get(bucket) or []):
                out.append((bucket, it))
        return out

    @staticmethod
    def render(facts: dict[str, list[str]]) -> str:
        """把整份清单渲染成文本块（回填/调试用）；空清单返回空串。"""
        lines = [f"- [{b}] {it}" for b, it in SelfFactsStore._flatten(facts)]
        return "\n".join(lines)

    @staticmethod
    def search(facts: dict[str, list[str]], query: str, k: int = 5) -> str:
        """按 query 用 BM25 检索清单里最相关的几条，渲染成文本块。

        条目少（≤30），即建即查。query 为空或清单为空 → 返回空串。命中不足
        k 条时全给。和「历史召回」同款 BM25（字符 n-gram），无额外依赖。"""
        items = SelfFactsStore._flatten(facts)
        if not items or not (query or "").strip():
            return ""
        from .rag import BM25, tokenize

        docs = [tokenize(f"{b} {it}") for b, it in items]
        bm = BM25()
        bm.fit(docs)
        q_tokens = tokenize(query)
        hits = bm.top_k(q_tokens, k=max(k, len(items)))  # 先全拿分，再按阈值筛
        if not hits:
            return ""
        # 阈值过滤，挡掉无关 query 因字符 n-gram 弱重叠（"的/了/今天"）硬塞的命中：
        # ① 绝对下限——分数太低直接丢；② 相对——只留 ≥ 最高分 70% 的；③ 截到 k 条。
        # 注意：上层 controller 已经判过「该不该查自我事实」，这里只是再加一道防线。
        top = hits[0][1]
        MIN_ABS = 0.5
        kept = [(i, s) for i, s in hits if s >= max(MIN_ABS, top * 0.7)][:k]
        if not kept:
            return ""
        lines = [f"- [{items[i][0]}] {items[i][1]}" for i, _ in kept]
        return "\n".join(lines)
