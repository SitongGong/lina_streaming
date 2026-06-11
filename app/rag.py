"""Lightweight RAG over the character's static documents.

Design choices:
- Only `person_setup.md` (who she is / origin) is always included in the
  system prompt — it's the identity bedrock; dropping a slice of it would
  break character, and it's small and won't grow unbounded.
- Everything else (world.md, sample_conversations.md, personality.md,
  hobbies.md, others.md) is chunked by markdown sections and retrieved via
  BM25, gated per-source by the controller plan. These files are expected to
  grow over time, so全量塞进 prompt 不划算——controller 按场景挑相关片段。
- Tokenization uses character bigrams, which works for Chinese without
  pulling in a segmenter (jieba) as a dependency.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# 只有身份根基永远全量进 prompt（不检索）。world / sample_conversations 以后会
# 越来越长，移到检索集，由 controller 按场景挑片段。
CORE_FILES = ("person_setup.md",)
RAG_FILES = (
    "world.md",
    "sample_conversations.md",
    "personality.md",
    "hobbies.md",
    "others.md",
)

# 时间衰减半衰期（秒）。一条记忆每过这么久，其检索权重减半。
# 7 天：本周内的记忆基本保权重，更老的逐步让位给近期记忆。
# 依据 LD-Agent（#8）的发现：纯语义检索精度不够，叠加时间衰减能显著
# 提升"勾到该接的那条"的命中率。
_RECENCY_HALFLIFE_SECONDS = 7 * 24 * 3600


def _recency_factor(ts: float, now: float, halflife: float = _RECENCY_HALFLIFE_SECONDS) -> float:
    """把一条记忆的时间戳换算成 (0, 1] 的衰减权重：越近越接近 1。

    用半衰期指数衰减：age 每过一个 halflife，权重减半。ts 缺失/异常时
    返回 1.0（不衰减，退化为纯 BM25）。"""
    try:
        age = max(0.0, float(now) - float(ts or 0.0))
    except (TypeError, ValueError):
        return 1.0
    if ts in (None, 0, 0.0):
        return 1.0
    return 0.5 ** (age / max(1.0, halflife))


@dataclass
class Chunk:
    text: str
    source: str
    heading: str

    def render(self) -> str:
        return f"【{self.source} · {self.heading}】\n{self.text}"


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs using level-1/2/3 headers."""
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "（开头）"
    current_body: list[str] = []
    header_re = re.compile(r"^#{1,3}\s+(.+)")
    for line in lines:
        m = header_re.match(line)
        if m:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, current_body))
    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


def _chunk_body(body: str, max_chars: int = 600) -> list[str]:
    """Further split long sections by numbered/bulleted items, paragraph-aware."""
    if len(body) <= max_chars:
        return [body]
    # Split on numbered list items at start of line ("1.", "2.")
    parts = re.split(r"\n(?=\s*\d+\.\s)", body)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = part
        else:
            buf = f"{buf}\n{part}".strip() if buf else part
    if buf:
        chunks.append(buf.strip())
    return chunks


def chunk_markdown(content: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _split_sections(content):
        for piece in _chunk_body(body):
            chunks.append(Chunk(text=piece, source=source, heading=heading))
    return chunks


def tokenize(text: str) -> list[str]:
    """Character n-gram tokens + ASCII word tokens.

    Emits both unigrams and bigrams over CJK + alphanumeric runs, plus
    ASCII word-level tokens. Unigrams give partial credit when a key
    character (e.g. 猫) appears in different bigram contexts; BM25's IDF
    naturally downweights ubiquitous characters like 的/了/我.
    """
    ascii_words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
    compact = re.sub(r"[^\w一-鿿]+", "", text)
    compact = re.sub(r"\s+", "", compact)
    if not compact:
        return ascii_words
    unigrams = [f"1:{c}" for c in compact]
    bigrams = [f"2:{compact[i:i+2]}" for i in range(len(compact) - 1)]
    return unigrams + bigrams + ascii_words


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.tf: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def fit(self, tokenized_docs: list[list[str]]) -> None:
        self.docs = tokenized_docs
        self.tf = [Counter(d) for d in tokenized_docs]
        n = max(1, len(tokenized_docs))
        self.avgdl = sum(len(d) for d in tokenized_docs) / n
        df: Counter[str] = Counter()
        for d in tokenized_docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            term: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for term, f in df.items()
        }

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if not self.docs:
            return 0.0
        doc = self.docs[doc_idx]
        if not doc:
            return 0.0
        dl = len(doc)
        tf = self.tf[doc_idx]
        s = 0.0
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * num / denom
        return s

    def top_k(self, query_tokens: list[str], k: int = 4) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]


class CharacterRAG:
    """Loads static character files, exposes core text + retrieval over the rest.

    `file_overrides` lets callers substitute the content of one or more
    static files at runtime without touching disk (used by the in-app
    prompt editor — overrides are kept off-git).
    """

    def __init__(
        self,
        static_dir: str | Path,
        file_overrides: dict[str, str] | None = None,
    ):
        self.static_dir = Path(static_dir)
        if not self.static_dir.exists():
            raise FileNotFoundError(f"Static directory not found: {self.static_dir}")
        self.file_overrides = dict(file_overrides or {})
        self._core_text: str = ""
        self._chunks: list[Chunk] = []
        self._bm25 = BM25()
        self._load()

    def _read(self, fname: str) -> str:
        if fname in self.file_overrides:
            return self.file_overrides[fname]
        fpath = self.static_dir / fname
        if fpath.exists():
            return fpath.read_text(encoding="utf-8")
        return ""

    def _load(self) -> None:
        # Core files concatenated, with file labels for traceability.
        core_parts: list[str] = []
        for fname in CORE_FILES:
            content = self._read(fname).strip()
            if content:
                core_parts.append(f"### 来源文件：{fname}\n{content}")
        self._core_text = "\n\n".join(core_parts)

        for fname in RAG_FILES:
            content = self._read(fname)
            if content:
                self._chunks.extend(chunk_markdown(content, fname))

        tokenized = [tokenize(c.text + " " + c.heading) for c in self._chunks]
        self._bm25.fit(tokenized)

    @property
    def core_text(self) -> str:
        return self._core_text

    @property
    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def retrieve(self, query: str, k: int = 4) -> list[Chunk]:
        if not query.strip() or not self._chunks:
            return []
        tokens = tokenize(query)
        hits = self._bm25.top_k(tokens, k=k)
        return [self._chunks[i] for i, _ in hits]

    def retrieve_with_scores(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not query.strip() or not self._chunks:
            return []
        tokens = tokenize(query)
        hits = self._bm25.top_k(tokens, k=k)
        return [(self._chunks[i], s) for i, s in hits]


def _top_k_weighted(
    bm: "BM25",
    query_tokens: list[str],
    chunks: list[Chunk],
    weights: list[float],
    k: int,
) -> list[Chunk]:
    """Score every doc with BM25, multiply by its per-doc weight (recency),
    and return the top-k chunks with positive score. Keeps the same
    "drop zero-score" semantics as BM25.top_k."""
    scored: list[tuple[int, float]] = []
    for i in range(len(chunks)):
        s = bm.score(query_tokens, i) * (weights[i] if i < len(weights) else 1.0)
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in scored[:k]]


def retrieve_history_chunks(
    messages,
    query: str,
    k: int = 3,
    exclude_recent_count: int = 30,
    recency: bool = True,
) -> list[Chunk]:
    """Build a per-session BM25 index over older (user, assistant) pairs and
    return the top-k most relevant ones for `query`.

    `messages` is the FULL list of past Message objects (or .role/.content
    dataclass instances). The most recent `exclude_recent_count` messages are
    skipped — those are already in the API's working window and don't need
    retrieval.

    When `recency` is True, each pair's BM25 score is multiplied by a time
    decay factor (from the user turn's timestamp), so a recent topic beats
    an old one that merely shares words. Pairs with no usable timestamp are
    left un-decayed.
    """
    if not query.strip() or not messages:
        return []

    older_count = len(messages) - exclude_recent_count
    if older_count <= 0:
        return []
    older = messages[:older_count]

    # Pair user turns with the immediately-following assistant turn.
    pairs: list[tuple] = []
    i = 0
    while i < len(older):
        m = older[i]
        if m.role == "user":
            if i + 1 < len(older) and older[i + 1].role == "assistant":
                pairs.append((m, older[i + 1]))
                i += 2
            else:
                pairs.append((m, None))
                i += 1
        else:
            i += 1

    if not pairs:
        return []

    now = time.time()
    docs: list[list[str]] = []
    chunks: list[Chunk] = []
    weights: list[float] = []
    for idx, (u, a) in enumerate(pairs):
        u_text = (u.content or "").strip()
        a_text = (a.content if a else "").strip()
        docs.append(tokenize(u_text + " " + a_text))
        body = f"用户：{u_text}"
        if a_text:
            body += f"\n西比莉娜：{a_text}"
        chunks.append(Chunk(text=body, source="对话历史", heading=f"第 {idx + 1} 轮"))
        weights.append(_recency_factor(getattr(u, "ts", 0.0), now) if recency else 1.0)

    bm = BM25()
    bm.fit(docs)
    return _top_k_weighted(bm, tokenize(query), chunks, weights, k)


def retrieve_user_memory_chunks(
    conversations,
    query: str,
    k: int = 3,
    current_session_id: str | None = None,
    skip_recent_in_current: int = 30,
    recency: bool = True,
) -> list[Chunk]:
    """跨会话的用户长期记忆检索。

    与 `retrieve_history_chunks` 同构，但范围是**同一用户的所有会话**，
    用于让莉娜记住用户在任意历史会话里讲过的事。不同用户的会话物理隔离
    （各自独立的存储目录），所以调用方只要传入该用户自己的会话列表，
    天然就不会检索到别的用户。

    参数：
      - conversations: 该用户的 Conversation 对象列表（通常是 store 里全部）。
      - query: 当前用户发言，用作检索 query。
      - current_session_id: 当前会话 id。该会话的最近
        `skip_recent_in_current` 条消息已经在上下文窗口里，跳过以免重复；
        但它更早的轮次仍会被纳入长期记忆。
    """
    if not query.strip() or not conversations:
        return []

    now = time.time()
    docs: list[list[str]] = []
    chunks: list[Chunk] = []
    weights: list[float] = []
    for conv in conversations:
        messages = list(getattr(conv, "messages", []) or [])
        sid = getattr(conv, "session_id", "") or ""
        title = (getattr(conv, "title", "") or "").strip() or sid or "（无标题）"
        # 当前会话：只取较早轮次（最近窗口已在上下文里）；其它会话：全取。
        if current_session_id is not None and sid == current_session_id:
            older_count = len(messages) - skip_recent_in_current
            if older_count <= 0:
                continue
            messages = messages[:older_count]

        # 把 user 轮和紧随其后的 assistant 轮配对。
        i = 0
        round_no = 0
        while i < len(messages):
            m = messages[i]
            if m.role == "user":
                a = messages[i + 1] if (i + 1 < len(messages) and messages[i + 1].role == "assistant") else None
                round_no += 1
                u_text = (m.content or "").strip()
                a_text = (a.content if a else "").strip()
                body = f"用户：{u_text}"
                if a_text:
                    body += f"\n西比莉娜：{a_text}"
                docs.append(tokenize(u_text + " " + a_text))
                chunks.append(
                    Chunk(text=body, source="跨会话记忆", heading=f"{title} · 第 {round_no} 轮")
                )
                weights.append(_recency_factor(getattr(m, "ts", 0.0), now) if recency else 1.0)
                i += 2 if a else 1
            else:
                i += 1

    if not chunks:
        return []

    bm = BM25()
    bm.fit(docs)
    return _top_k_weighted(bm, tokenize(query), chunks, weights, k)
