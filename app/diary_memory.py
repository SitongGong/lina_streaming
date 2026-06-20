"""莉娜「日记 + 谈资」检索 —— 本地 EverOS agent 轨道。

设计（2026-06-18 重构，按用户要求）：以**话题卡为权威**，话题→日记走卡里的
**精确日记索引**，不再 vector 模糊猜。

流程：
  当前对话上下文
    → ① 检索话题卡库(lina_topic, vector) → 命中【完整话题卡】（只有写完的 213 张才在库里）
    → ② 从命中的卡里读出「日记索引」(M2026-..) → 按ID**精确取**对应日记(lina_diary)
    → ③ 把【话题卡全信息(开场句/观点/共情方式/暴露边界…) + 精确日记】一起注入

要点：
- **话题卡是权威**：信任门槛、适用条件、开场句、暴露边界都来自卡，莉娜照着用。
- **未写完的话题卡不在库里**（导入时已过滤）→ 检索不到 → 上层回退人设（不硬编）。
- **主动发言**按卡的 trust_threshold 过滤（信任<门槛的话题不选）。

开关：EVEROS_DIARY=1。本地 EverOS 8090 需在跑。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_URL = "http://127.0.0.1:8090"
METHOD = "vector"

_DIARY_ID_RE = re.compile(r"M\d{4}-\d{2}-\d{2}-\d{2}")
_TRUST_RE = re.compile(r"信任门槛:\s*(\d+)")
# 注入时要去掉的、纯结构控制行（信任门槛/日记索引），不给主模型看
_STRIP_LINES_RE = re.compile(r"^(信任门槛|日记索引|话题):.*$", re.M)


class DiaryMemory:
    def __init__(
        self,
        base_url: str = "",
        *,
        topic_agent_id: str = "lina_topic",
        diary_agent_id: str = "lina_diary",
        timeout: float = 6.0,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("EVEROS_LOCAL_URL") or DEFAULT_LOCAL_URL
        ).rstrip("/")
        self._topic_agent = topic_agent_id
        self._diary_agent = diary_agent_id
        self._timeout = timeout
        self._enabled_flag = os.environ.get("EVEROS_DIARY", "").strip() in ("1", "true", "True")

    @property
    def enabled(self) -> bool:
        return self._enabled_flag and bool(self._base_url)

    def _search(self, agent_id: str, query: str, k: int, method: str = METHOD) -> list[dict[str, Any]]:
        body = {"agent_id": agent_id, "query": query, "method": method, "top_k": k}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/v1/memory/search", json=body)
            if resp.status_code != 200:
                logger.warning("DiaryMemory search %s: %s", resp.status_code, resp.text[:160])
                return []
            return (resp.json() or {}).get("data", {}).get("agent_skills") or []
        except Exception as exc:
            logger.warning("DiaryMemory search failed: %s", exc)
            return []

    # ---- 第①级：命中话题卡 ----
    def find_topic_cards(self, context: str, k: int = 3, max_trust: int | None = None) -> list[dict[str, Any]]:
        """检索话题卡，返回 [{title, trust, diary_ids, content}]。

        max_trust（当前信任值）给定时，按卡的 trust_threshold 过滤——主动发言用，
        信任不够的高暴露话题不选。普通对话回应不传 max_trust（用户已主动问了）。
        """
        if not self.enabled or not context.strip():
            return []
        cards: list[dict[str, Any]] = []
        for s in self._search(self._topic_agent, context, k):
            content = str(s.get("content") or "")
            desc = str(s.get("description") or "").strip()
            trust_m = _TRUST_RE.search(content)
            trust = int(trust_m.group(1)) if trust_m else 1
            if max_trust is not None and trust > max_trust:
                continue  # 信任门槛过滤
            # 话题池的稳定 key 用 skill name（topic_A01_01_xxx，唯一），区分同ID多角度
            cards.append({
                "topic_id": str(s.get("name") or "").strip() or desc,
                "title": desc,
                "trust": trust,
                "diary_ids": _DIARY_ID_RE.findall(content),
                "content": content,
            })
        return cards

    # ---- 第②级：按日记索引ID精确取日记 ----
    def get_diaries_by_ids(self, diary_ids: list[str]) -> dict[str, str]:
        """按记忆ID精确取日记正文。返回 {id: 正文}。

        用 keyword 搜 ID 再按 name 精确过滤（name == diary_<ID>），避免子串误命中。
        """
        out: dict[str, str] = {}
        for did in diary_ids:
            if did in out:
                continue
            for s in self._search(self._diary_agent, did, k=5, method="keyword"):
                if str(s.get("name") or "") == f"diary_{did}":
                    out[did] = str(s.get("content") or "").strip()
                    break
        return out

    @staticmethod
    def _clean(text: str) -> str:
        """去掉结构控制行（信任门槛/日记索引/话题标签），只留给主模型看的素材。"""
        return _STRIP_LINES_RE.sub("", text).strip()

    def render_card(self, card: dict, diaries: dict[str, str]) -> str:
        """把一张话题卡（含其精确日记）渲染成注入文本块。"""
        seg = [f"【话题：{card['title']}】", self._clean(card["content"])]
        for did in card.get("diary_ids", []):
            if did in diaries:
                seg.append(f"〔相关日记〕{self._clean(diaries[did])}")
        return "\n".join(seg)

    def search_diaries_direct(self, context: str, k: int = 2) -> list[dict[str, str]]:
        """补充路径：直接检索日记库（lina_diary）。**vector + BM25(keyword) 双路融合**。

        话题卡只覆盖偏好/共感类，问到叙事性具体经历或**专名/人物**（培根、某次进城）
        话题卡匹配不到。日记直检让这类能召回；其中 BM25 对**专名/具体词**尤其准
        （"培根"出现在哪些日记里精确命中），vector 管语义近似——两路融合互补。
        返回 [{name, content}]，按「两路都命中 > 单路命中」优先、各取前 k。
        """
        # 两路各多召回一些（k+3）再融合取前 k，避免好结果在单路排第3却被融合挤掉
        # （如「探视昏迷的培根」vector 排第3，要召够才进得了融合 top）。
        pool_n = k + 3
        vec = self._search(self._diary_agent, context, pool_n, method="vector")
        kw = self._search(self._diary_agent, context, pool_n, method="keyword")
        # RRF 简化融合：按出现位次给分，两路都中的排前
        score: dict[str, float] = {}
        rec: dict[str, dict] = {}
        for lst in (vec, kw):
            for rank, s in enumerate(lst):
                name = str(s.get("name") or "")
                if not name:
                    continue
                score[name] = score.get(name, 0.0) + 1.0 / (rank + 1)
                rec.setdefault(name, s)
        ranked = sorted(score, key=lambda n: score[n], reverse=True)
        out: list[dict[str, str]] = []
        for name in ranked[:k]:
            content = str(rec[name].get("content") or "").strip()
            if content:
                out.append({"name": name, "content": content})
        return out

    def retrieve_cards(self, context: str, *, topic_k: int = 2, max_trust: int | None = None,
                       query_type: str = "global") -> list[dict[str, Any]]:
        """检索，**按 query_type 分路**：
        - detail（细节/专名，如「培根怎么样」「安娜什么关系」）：**只走日记直检**（BM25+vector），
          不走话题卡——话题卡只有偏好类，对专名召回全是垃圾（培根→拿手菜），会添噪声。
        - global（偏好/共感/泛问，如「你爱吃什么」「讲讲过去」）：走**话题卡为主**（谈资指引
          开场句/共情方式都在）+ 日记直检补充。
        全空 → []（上层回退人设/老实说不记得）。
        """
        if not self.enabled or not context.strip():
            return []
        out: list[dict[str, Any]] = []
        all_ids: list[str] = []
        # ① 话题卡：仅 global 走（detail 跳过，避免偏好卡对专名添噪声）
        if query_type != "detail":
            cards = self.find_topic_cards(context, k=topic_k, max_trust=max_trust)
            for c in cards:
                all_ids.extend(c["diary_ids"])
            diaries = self.get_diaries_by_ids(all_ids)
            for c in cards:
                out.append({
                    "topic_id": c["topic_id"],
                    "title": c["title"],
                    "diary_ids": c["diary_ids"],
                    "text": self.render_card(c, diaries),
                })
        # ② 日记直检（BM25+vector）。detail 时是主力、多取几篇；global 时补充。
        seen_diary = set(all_ids)
        diary_k = 5 if query_type == "detail" else 4
        for d in self.search_diaries_direct(context, k=diary_k):
            did = d["name"].replace("diary_", "")
            if did in seen_diary:
                continue
            seen_diary.add(did)
            # title 用日记的【可读标题】（content 第一行 "# 2026-01-24 探视昏迷的培根老师"
            # 去掉日期），而不是纯ID——否则话题池里全是 diary::M2026-.. 机器码，判断器
            # 认不出「这是培根话题」，导致连问培根每轮都判新话题、永不接续。
            title = self._diary_title(d["content"]) or f"经历·{did}"
            out.append({
                "topic_id": f"diary::{did}",
                "title": title,
                "diary_ids": [did],
                "text": f"〔莉娜的相关经历〕{self._clean(d['content'])}",
            })
        return out

    @staticmethod
    def _diary_title(content: str) -> str:
        """从日记正文取可读标题：首行 '# 2026-01-24 探视昏迷的培根老师' → '探视昏迷的培根老师'。"""
        first = content.lstrip().split("\n", 1)[0].lstrip("# ").strip()
        # 去掉开头的日期 2026-01-24
        return re.sub(r"^\d{4}-\d{2}-\d{2}\s*", "", first).strip()

    def retrieve_text(self, context: str, *, topic_k: int = 2, max_trust: int | None = None) -> str:
        """返回注入文本（兼容旧接口）。检索不到 → ""。"""
        cards = self.retrieve_cards(context, topic_k=topic_k, max_trust=max_trust)
        return "\n\n———\n\n".join(c["text"] for c in cards) if cards else ""
