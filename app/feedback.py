"""Post-evaluation questionnaire storage.

After a tester has chatted with Lina they fill in a short questionnaire that
rates her on nine capability dimensions (good / bad + a reason) plus a free-text
"other" field. One questionnaire per session — keyed by session_id, so a resubmit
overwrites the previous answers for that session.

Design mirrors `conversation.ConversationStore`: one JSON file per record under a
flat directory, no database.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .conversation import _safe_session_id


# The nine scored dimensions, in display order. Single source of truth — the
# web layer exposes this to the frontend via /api/feedback/schema so the
# labels never get duplicated.
DIMENSIONS: list[tuple[str, str]] = [
    ("L1", "知识广度"),
    ("L2", "逻辑能力"),
    ("L3", "指令遵循能力"),
    ("L4", "长程对话记忆"),
    ("L5", "多轮对话能力"),
    ("L6", "情商"),
    ("L7", "有害请求拒答"),
    ("L8", "语言流畅度"),
    ("L9", "角色一致性"),
]
DIMENSION_KEYS = {k for k, _ in DIMENSIONS}

RATINGS = {"good", "bad"}

REASON_MAX = 1000
OTHER_MAX = 4000


@dataclass
class FeedbackRecord:
    session_id: str
    client_id: str | None = None
    # Human-readable label of the evaluated session, copied in at submit time
    # so a questionnaire can be traced back to its conversation without a join.
    session_title: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # {dim_key: {"rating": "good"|"bad", "reason": str}}
    dimensions: dict = field(default_factory=dict)
    other: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "client_id": self.client_id,
            "session_title": self.session_title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dimensions": self.dimensions,
            "other": self.other,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackRecord":
        return cls(
            session_id=d["session_id"],
            client_id=d.get("client_id"),
            session_title=d.get("session_title", "") or "",
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", d.get("created_at", time.time())),
            dimensions=d.get("dimensions", {}) or {},
            other=d.get("other", "") or "",
        )


def validate_submission(dimensions, other) -> tuple[dict, str]:
    """Whitelist + clamp a user-submitted questionnaire.

    Returns (cleaned_dimensions, cleaned_other). Raises ValueError with a
    Chinese message on anything malformed. Rule: a "bad" rating MUST carry a
    non-empty reason; a "good" rating's reason is optional.
    """
    if not isinstance(dimensions, dict):
        raise ValueError("dimensions 必须是对象。")
    cleaned: dict = {}
    for key, label in DIMENSIONS:
        entry = dimensions.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"维度「{label}」缺少评价。")
        rating = entry.get("rating")
        if rating not in RATINGS:
            raise ValueError(f"维度「{label}」的评价必须是 好 或 不好。")
        reason = entry.get("reason") or ""
        if not isinstance(reason, str):
            raise ValueError(f"维度「{label}」的原因必须是文本。")
        reason = reason.strip()[:REASON_MAX]
        if rating == "bad" and not reason:
            raise ValueError(f"维度「{label}」评为「不好」时必须填写原因。")
        cleaned[key] = {"rating": rating, "reason": reason}
    extra = [k for k in dimensions if k not in DIMENSION_KEYS]
    if extra:
        raise ValueError(f"未知维度：{', '.join(extra)}")
    other_text = (other or "")
    if not isinstance(other_text, str):
        raise ValueError("other 必须是文本。")
    other_text = other_text.strip()[:OTHER_MAX]
    return cleaned, other_text


class FeedbackStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe_session_id(session_id)}.json"

    def load(self, session_id: str) -> FeedbackRecord | None:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            return FeedbackRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    def save(self, record: FeedbackRecord) -> None:
        self._path(record.session_id).write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def submit(
        self,
        session_id: str,
        dimensions: dict,
        other: str,
        client_id: str | None,
        session_title: str = "",
    ) -> FeedbackRecord:
        """Validate + persist. On resubmit, keep the original created_at and
        bump updated_at."""
        cleaned_dims, cleaned_other = validate_submission(dimensions, other)
        existing = self.load(session_id)
        now = time.time()
        record = FeedbackRecord(
            session_id=session_id,
            client_id=client_id,
            session_title=(session_title or (existing.session_title if existing else "")),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            dimensions=cleaned_dims,
            other=cleaned_other,
        )
        self.save(record)
        return record

    def list_records(self, client_id: str | None = None) -> list[FeedbackRecord]:
        """All records, newest first. When client_id is given, only that
        client's records (sidebar-style scoping); when None, everything."""
        records: list[FeedbackRecord] = []
        for p in self.root.glob("*.json"):
            try:
                rec = FeedbackRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
            if client_id is not None and rec.client_id != client_id:
                continue
            records.append(rec)
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records

    def summary(self, client_id: str | None = None) -> dict:
        """Aggregate good/bad counts per dimension + collect reasons and the
        free-text 'other' notes."""
        records = self.list_records(client_id)
        dims: list[dict] = []
        for key, label in DIMENSIONS:
            good = bad = 0
            reasons: list[dict] = []
            for rec in records:
                entry = rec.dimensions.get(key)
                if not isinstance(entry, dict):
                    continue
                rating = entry.get("rating")
                if rating == "good":
                    good += 1
                elif rating == "bad":
                    bad += 1
                else:
                    continue
                reason = (entry.get("reason") or "").strip()
                if reason:
                    reasons.append(
                        {
                            "rating": rating,
                            "reason": reason,
                            "session_id": rec.session_id,
                            "session_title": rec.session_title,
                            "created_at": rec.updated_at,
                        }
                    )
            dims.append(
                {
                    "key": key,
                    "label": label,
                    "good": good,
                    "bad": bad,
                    "total": good + bad,
                    "reasons": reasons,
                }
            )
        others = [
            {
                "other": rec.other.strip(),
                "session_id": rec.session_id,
                "session_title": rec.session_title,
                "created_at": rec.updated_at,
            }
            for rec in records
            if (rec.other or "").strip()
        ]
        return {
            "submission_count": len(records),
            "dimensions": dims,
            "others": others,
        }
