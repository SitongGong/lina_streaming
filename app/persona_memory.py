"""莉娜「可检索人设」客户端 —— 走本地 EverOS 的 agent 轨道（agent_skill）。

与对话记忆（云 EverMem，user 轨道，memory_client.py）是**两套不同的服务**：
- 对话记忆：云 api.evermind.ai，/api/v1/memories（复数），Bearer 鉴权，user_id。
- 人设/日记：**本地 EverOS**，/api/v1/memory（单数）/search，无鉴权，agent_id。

这里只管后者：把人设 md 导入本地 EverOS agent 轨道后（见
scripts/import_persona_to_everos.py），chat 时按本轮话题语义检索莉娜的世界观/
性格/兴趣等设定，返回 Chunk（与原 CharacterRAG.retrieve 同结构），注入到
prompt 的「角色设定参考」块——**取代原人设 BM25 检索**。

核心人设 person_setup.md 不走这里（仍全量进 prompt，是 CharacterRAG.core_text）。

检索方法用 **vector（纯语义）**：实测对人设最准（查中文「性格」能命中英文 Big
Five 测试），且不走 cross-encoder rerank（本机 8009 rerank 端点格式与 EverOS 期望
不兼容，hybrid 会 404）。

日记将来同理：写进同一 agent 轨道，用同一个 client 检索。

没配本地 EverOS（EVEROS_LOCAL_URL 未设/服务没起）时 enabled=False，自动回退原
CharacterRAG，不影响现网。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .rag import Chunk

logger = logging.getLogger(__name__)

# 本地 EverOS 默认地址（everos server start --port 8090）。
DEFAULT_LOCAL_URL = "http://127.0.0.1:8090"

# 人设检索用 vector：语义准、不触发 rerank 404。
PERSONA_METHOD = "vector"
DEFAULT_TOP_K = 6


class PersonaMemory:
    """本地 EverOS agent 轨道检索客户端（同步）。"""

    def __init__(
        self,
        base_url: str = "",
        agent_id: str = "lina",
        *,
        method: str = PERSONA_METHOD,
        timeout: float = 6.0,
    ) -> None:
        self._base_url = (
            base_url or os.environ.get("EVEROS_LOCAL_URL") or DEFAULT_LOCAL_URL
        ).rstrip("/")
        self._agent_id = agent_id
        self._method = method
        self._timeout = timeout
        # 显式开关：默认关闭，置 EVEROS_PERSONA=1 才启用（避免本地 EverOS 没起时
        # 每轮都白打一次超时）。
        self._enabled_flag = os.environ.get("EVEROS_PERSONA", "").strip() in ("1", "true", "True")

    @property
    def enabled(self) -> bool:
        return self._enabled_flag and bool(self._base_url and self._agent_id)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[Chunk]:
        """语义检索莉娜人设，返回 Chunk 列表（替换 CharacterRAG.retrieve 的产出）。

        失败/空 → 返回 []，上层据此回退原 CharacterRAG。
        """
        if not self.enabled or not query.strip():
            return []
        body = {
            "agent_id": self._agent_id,
            "query": query,
            "method": self._method,
            "top_k": max(1, min(int(k or DEFAULT_TOP_K), 50)),
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/v1/memory/search", json=body)
            if resp.status_code != 200:
                logger.warning("PersonaMemory search %s: %s", resp.status_code, resp.text[:160])
                return []
            data = (resp.json() or {}).get("data") or {}
        except Exception as exc:
            logger.warning("PersonaMemory search failed: %s", exc)
            return []
        return _skills_to_chunks(data.get("agent_skills") or [])


def _skills_to_chunks(skills: list[Any]) -> list[Chunk]:
    """把 agent_skill 检索结果转成 Chunk（content 里带了「来源·章节」前缀）。"""
    chunks: list[Chunk] = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        content = str(s.get("content") or "").strip()
        if not content:
            continue
        # description 形如 "world · 科技水平"，拆成 source / heading。
        desc = str(s.get("description") or s.get("name") or "").strip()
        if " · " in desc:
            source, heading = desc.split(" · ", 1)
        else:
            source, heading = "persona", desc or "设定"
        chunks.append(Chunk(text=content, source=source.strip(), heading=heading.strip()))
    return chunks
