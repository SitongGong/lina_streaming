"""莉娜「可检索行为规则 / 情绪表」客户端 —— 走本地 EverOS agent 轨道 lina_rules。

与人设(persona_memory.py)同款机制：行为规则(behavior_rules)和情绪表(mood_format_spec)
按小节导入 EverOS（见 scripts/import_rules_to_everos.py），运行时按当前用户话语义检索
top-k 条相关规则注入，而非把整份规则全量塞进 prompt。

- behavior：21 条口语/续聊规则等，检索 top-k 条本轮用得上的。
- mood：情绪表（每个情绪词一块），检索 top-k 个本轮可能用到的情绪。
  （情绪标记的格式说明 + 三条铁律是「机制底线」，仍由 mood_format_spec 常驻，不在此检索。）

没配本地 EverOS（EVEROS_RULES 未开/服务没起）时 enabled=False，上层回退整份注入。
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_URL = "http://127.0.0.1:8090"
METHOD = "vector"


class RulesMemory:
    """检索行为规则 / 情绪表的 EverOS 客户端（同步）。"""

    def __init__(self, base_url: str = "", agent_id: str = "lina_rules",
                 *, method: str = METHOD, timeout: float = 6.0) -> None:
        self._base_url = (base_url or os.environ.get("EVEROS_LOCAL_URL") or DEFAULT_LOCAL_URL).rstrip("/")
        self._agent_id = agent_id
        self._method = method
        self._timeout = timeout
        # 显式开关：默认关。EVEROS_RULES=1 才启用规则检索（否则上层用整份注入）。
        self._enabled_flag = os.environ.get("EVEROS_RULES", "").strip() in ("1", "true", "True")

    @property
    def enabled(self) -> bool:
        return self._enabled_flag and bool(self._base_url and self._agent_id)

    def _search(self, query: str, k: int) -> list[dict]:
        if not self.enabled or not query.strip():
            return []
        body = {"agent_id": self._agent_id, "query": query, "method": self._method,
                "top_k": max(1, min(int(k or 6), 50))}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/v1/memory/search", json=body)
            if resp.status_code != 200:
                logger.warning("RulesMemory search %s: %s", resp.status_code, resp.text[:160])
                return []
            return ((resp.json() or {}).get("data") or {}).get("agent_skills") or []
        except Exception as exc:
            logger.warning("RulesMemory search failed: %s", exc)
            return []

    def _retrieve(self, query: str, k: int, prefix: str) -> str:
        """检索某类(behavior/mood)规则，按 description 的 '<prefix> · ' 前缀过滤，
        多检一些再筛，拼成可注入文本（去掉 skill 正文里的多余空行）。"""
        # 两类混在同一轨道，按前缀过滤；为保证筛后够 k 条，先多取。
        skills = self._search(query, k * 3)
        picked: list[str] = []
        for s in skills:
            if not isinstance(s, dict):
                continue
            desc = str(s.get("description") or "")
            if not desc.startswith(prefix):
                continue
            content = str(s.get("content") or "").strip()
            if content:
                picked.append(content)
            if len(picked) >= k:
                break
        return "\n\n".join(picked)

    def retrieve_behavior(self, query: str, k: int = 8) -> str:
        return self._retrieve(query, k, "behavior")

    def retrieve_mood(self, query: str, k: int = 6) -> str:
        return self._retrieve(query, k, "mood")
