"""Conversation history persistence — one JSON file per session."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    ts: float = field(default_factory=time.time)
    # Free-form metadata. For assistant messages this carries the parsed mood
    # tag (mood / intensity / trust). Stored verbatim alongside the message.
    meta: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            content=d["content"],
            ts=d.get("ts", time.time()),
            meta=d.get("meta"),
        )


def _safe_session_id(raw: str | None) -> str:
    if not raw:
        return "sess-" + uuid.uuid4().hex[:8]
    # constrain to filesystem-friendly chars
    cleaned = re.sub(r"[^A-Za-z0-9_\-.]", "_", raw.strip())
    return cleaned[:64] or ("sess-" + uuid.uuid4().hex[:8])


@dataclass
class Conversation:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    title: str = ""
    # Prompt source for this session.
    #   "shared"  → use prompt_version_id (None = current global)
    #   "private" → use this session's own prompt_overrides dict
    prompt_mode: str = "shared"
    # Used when prompt_mode == "shared". Points to a saved version's id;
    # None = use the editable globals.
    prompt_version_id: str | None = None
    # Used when prompt_mode == "private". A dict of component → content
    # that overrides defaults for this session only. Independent from the
    # global overrides and from any saved version.
    prompt_overrides: dict | None = None
    # One-shot manual override for the next turn's starting mood/trust.
    # Engine merges these on top of last_assistant_meta then clears the
    # field. {mood?: str, intensity?: int 1-10, trust?: int 1-10}
    forced_state: dict | None = None
    # Owner of this session — a random per-browser id (localStorage). Used to
    # scope the sidebar so testers on a shared public URL only see their own
    # sessions. None = legacy/untagged (won't appear in any client's list).
    client_id: str | None = None
    # Remaining segments of a reply that was split into chunks. The first
    # chunk is sent immediately as the reply; the rest are short outline
    # points parked here and delivered one at a time via continue_segment()
    # when the user stays silent. Cleared the moment the user speaks again
    # (a new chat() turn) — a fresh message makes the old plan stale.
    pending_segments: list[str] | None = None
    # 本会话「已聊话题池」：莉娜的日记/谈资话题一旦被检索过就缓存在这，供后续
    # 追问/隔几轮绕回时复用，不重新检索——解决「追问就编 / 前后不一致」。
    #   {topic_id: {"title": str, "diary_text": str(命中卡+精确日记的注入文本),
    #               "turns": [聊过的轮序号]}}
    # 只缓存「日记内容」（静态、不会变），对话进展不缓存（靠近期窗口+用户记忆）。
    # 会话级、随会话持久化；单会话场景，不跨会话。
    topic_pool: dict | None = None

    def add(self, role: str, content: str, meta: dict | None = None) -> Message:
        msg = Message(role=role, content=content, meta=meta)
        self.messages.append(msg)
        if not self.title and role == "user":
            self.title = content[:30]
        return msg

    def last_assistant_meta(self) -> dict | None:
        for m in reversed(self.messages):
            if m.role == "assistant" and m.meta:
                return m.meta
        return None

    def as_api_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_dict(self) -> dict:
        d: dict = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "title": self.title,
            "prompt_mode": self.prompt_mode,
            "prompt_version_id": self.prompt_version_id,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.prompt_overrides:
            d["prompt_overrides"] = self.prompt_overrides
        if self.forced_state:
            d["forced_state"] = self.forced_state
        if self.client_id:
            d["client_id"] = self.client_id
        if self.pending_segments:
            d["pending_segments"] = self.pending_segments
        if self.topic_pool:
            d["topic_pool"] = self.topic_pool
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(
            session_id=d["session_id"],
            created_at=d.get("created_at", time.time()),
            title=d.get("title", ""),
            prompt_mode=d.get("prompt_mode", "shared"),
            prompt_version_id=d.get("prompt_version_id"),
            prompt_overrides=d.get("prompt_overrides"),
            forced_state=d.get("forced_state"),
            client_id=d.get("client_id"),
            pending_segments=d.get("pending_segments"),
            topic_pool=d.get("topic_pool"),
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
        )


class ConversationStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe_session_id(session_id)}.json"

    def new_session(self, requested_id: str | None = None) -> Conversation:
        sid = _safe_session_id(requested_id)
        # Ensure uniqueness if a session id collides with an existing file.
        if self._path(sid).exists() and not requested_id:
            sid = "sess-" + uuid.uuid4().hex[:8]
        conv = Conversation(session_id=sid)
        self.save(conv)
        return conv

    def load(self, session_id: str) -> Conversation:
        p = self._path(session_id)
        if not p.exists():
            return self.new_session(session_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        return Conversation.from_dict(data)

    def save(self, conv: Conversation) -> None:
        self._path(conv.session_id).write_text(
            json.dumps(conv.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, session_id: str) -> bool:
        p = self._path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict]:
        items: list[dict] = []
        for p in self.root.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                items.append(
                    {
                        "session_id": data.get("session_id", p.stem),
                        "title": data.get("title", "") or "（无标题）",
                        "created_at": data.get("created_at", p.stat().st_mtime),
                        "message_count": len(data.get("messages", [])),
                        "prompt_mode": data.get("prompt_mode", "shared"),
                        "prompt_version_id": data.get("prompt_version_id"),
                        "client_id": data.get("client_id"),
                    }
                )
            except Exception:
                continue
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items

    def iter_sessions(self) -> Iterator[Conversation]:
        for p in self.root.glob("*.json"):
            try:
                yield Conversation.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
