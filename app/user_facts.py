"""「用户事实清单」存储 —— per-user，跨会话共享。

记的是**用户在对话里讲过的、关于用户自己的稳定事实**（养了猫团子 / 在做什么
工作 / 喜欢喝抹茶 / 老家在哪 / 最近在忙的事 …），用来在对话历史滑出上下文
窗口后，莉娜仍能记住用户是谁、聊过什么 —— 解决「聊了上百轮后把用户的事
忘了 / 记混 / 编造」的问题。

与「莉娜自我事实清单」(self_facts) 对称：
- self_facts 记**莉娜自己**的事；user_facts 记**用户**的事。
- 同款机制：分桶、条数上限、后台 LLM 概括、BM25 按话题检索注入。
- 各存各的目录（users/user_facts/<user_id>.json）。

纯标准库，进程内加锁。
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

# 用户事实的类别（桶）。LLM 概括时被要求只用这些键。
BUCKETS = ("身份", "喜好", "厌恶", "关系", "经历", "近况", "约定")
# 每个桶最多留几条。
MAX_PER_BUCKET = 8
# 整份清单总条数上限（兜底）。用户的事比莉娜自己的多，给大一点。
MAX_TOTAL = 40


def _safe_user_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]", "_", (raw or "").strip())[:64]


class UserFactsStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, user_id: str) -> Path:
        return self.root / f"{_safe_user_id(user_id)}.json"

    def load(self, user_id: str) -> dict[str, list[str]]:
        """返回该用户的用户事实清单（分桶 dict）。不存在则空 dict。"""
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
        """兜底裁剪：只保留合法桶、每桶去重并截到上限、总数截到 MAX_TOTAL。"""
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
                room = MAX_TOTAL - total
                if room <= 0:
                    break
                kept = kept[:room]
                out[bucket] = kept
                total += len(kept)
        return out

    @staticmethod
    def _flatten(facts: dict[str, list[str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for bucket in BUCKETS:
            for it in (facts.get(bucket) or []):
                out.append((bucket, it))
        return out

    @staticmethod
    def render(facts: dict[str, list[str]]) -> str:
        """整份清单渲染成文本块（回填/调试用）；空清单返回空串。"""
        lines = [f"- [{b}] {it}" for b, it in UserFactsStore._flatten(facts)]
        return "\n".join(lines)

    @staticmethod
    def search(facts: dict[str, list[str]], query: str, k: int = 6) -> str:
        """按 query 用 BM25 检索清单里最相关的几条，渲染成文本块。
        query 为空或清单为空 → 空串；命中不足 k 条全给。"""
        items = UserFactsStore._flatten(facts)
        if not items or not (query or "").strip():
            return ""
        from .rag import BM25, tokenize

        docs = [tokenize(f"{b} {it}") for b, it in items]
        bm = BM25()
        bm.fit(docs)
        hits = bm.top_k(tokenize(query), k=max(k, len(items)))
        if not hits:
            return ""
        top = hits[0][1]
        MIN_ABS = 0.5
        kept = [(i, s) for i, s in hits if s >= max(MIN_ABS, top * 0.7)][:k]
        if not kept:
            return ""
        return "\n".join(f"- [{items[i][0]}] {items[i][1]}" for i, _ in kept)
