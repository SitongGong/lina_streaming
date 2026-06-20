"""EverMem 云记忆客户端 —— 对话长期记忆走云服务。

取代原 user_facts（分桶 + BM25）的「用户长期记忆」职责：把每轮对话写入
EverMem 云（api.evermind.ai），检索时用其 hybrid 召回（语义 + 关键词），
解决「换个说法问就检不到」的问题。

对照实现：/root/11mio/retrieval/evermemos_client_personal_v1.py
（同一套 official v1 personal 契约，已实测云端鉴权通）。

契约端点（注意是 memories 复数）：
- POST /api/v1/memories          写入对话
- POST /api/v1/memories/search   检索
- POST /api/v1/memories/flush    强制提取
- POST /api/v1/memories/get      取

鉴权：Authorization: Bearer <api_key>。

注意：莉娜「自己的事」（self_facts / 日记）**不走这里** —— 日记走本地
EverOS 写 md 建索引，self_facts 暂留原 JSON 层。这里只管「用户对话记忆」。

纯同步实现（用 httpx.Client），与现有 web.py 的同步调用风格一致；
写入/flush 由 web 层放到后台线程，不阻塞回复。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# 云服务默认地址（与 11mio DEFAULT_EVERMEMOS_CLOUD_URL 一致）。
DEFAULT_BASE_URL = "https://api.evermind.ai"

# 检索默认参数。hybrid = 语义 + 关键词 RRF 融合，是云端推荐默认。
DEFAULT_METHOD = "hybrid"
DEFAULT_MEMORY_TYPES = ("profile", "episodic_memory", "raw_message")
DEFAULT_TOP_K = 10

# 注入文本的总长上限（字符），防止把 prompt 撑爆。
MAX_RENDER_CHARS = 1800


class MemoryClient:
    """EverMem 云记忆客户端（同步）。

    一个进程一个实例即可，按 user_id 隔离不同对话用户。
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        *,
        method: str = DEFAULT_METHOD,
        timeout: float = 8.0,
        write_timeout: float = 30.0,
    ) -> None:
        self._base_url = (base_url or os.environ.get("EVERMEM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.environ.get("EVERMEM_API_KEY", "")
        self._method = method or DEFAULT_METHOD
        self._timeout = timeout
        self._write_timeout = write_timeout

    @property
    def enabled(self) -> bool:
        """没配 key 时直接降级（不报错），让上层照旧走原 user_facts。"""
        return bool(self._base_url and self._api_key)

    # ---- 内部 HTTP ----

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body, headers=self._headers())
        if resp.status_code != 200:
            logger.warning("EverMem %s -> %s: %s", path, resp.status_code, resp.text[:200])
            return {}
        try:
            payload = resp.json()
        except Exception:
            return {}
        # 响应封装：{request_id, data: {...}} 或 {data:{...}}；统一取 data。
        if isinstance(payload, dict):
            data = payload.get("data")
            return data if isinstance(data, dict) else payload
        return {}

    # ---- 写入 ----

    def add_turn(
        self,
        user_id: str,
        user_text: str,
        ai_text: str,
        *,
        session_id: str = "",
    ) -> str:
        """写入一轮对话（user + assistant）。返回提取状态字符串（accumulated/extracted/""）。"""
        if not self.enabled or not user_id:
            return ""
        now_ms = int(time.time() * 1000)
        body: dict[str, Any] = {
            "user_id": user_id,
            "messages": [
                {"role": "user", "timestamp": now_ms, "content": user_text},
                {"role": "assistant", "timestamp": now_ms + 1, "content": ai_text},
            ],
            "async_mode": False,
        }
        if session_id:
            body["session_id"] = session_id
        data = self._post("/api/v1/memories", body, self._write_timeout)
        return str(data.get("status") or "").strip().lower()

    def flush(self, user_id: str, *, session_id: str = "") -> None:
        """强制把缓冲的消息提取成记忆（一般在会话结束时调）。"""
        if not self.enabled or not user_id:
            return
        body: dict[str, Any] = {"user_id": user_id}
        if session_id:
            body["session_id"] = session_id
        self._post("/api/v1/memories/flush", body, self._write_timeout)

    # ---- 检索 ----

    def search(self, user_id: str, query: str, *, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """检索用户记忆。返回原始 data（含 episodes/profiles/raw_messages）。"""
        if not self.enabled or not user_id or not query.strip():
            return {}
        body = {
            "query": query,
            "method": self._method,
            "memory_types": list(DEFAULT_MEMORY_TYPES),
            "top_k": max(1, min(int(top_k or DEFAULT_TOP_K), 100)),
            "filters": {"user_id": user_id},
        }
        return self._post("/api/v1/memories/search", body, self._timeout)

    def search_text(self, user_id: str, query: str, *, top_k: int = DEFAULT_TOP_K) -> str:
        """检索并整理成注入 prompt 用的文本（对齐原 user_facts_text 的角色）。

        失败/空 → 返回空串，上层据此决定是否回退原 user_facts。
        """
        data = self.search(user_id, query, top_k=top_k)
        if not data:
            return ""
        return _render_search(data)


def _first_nonempty(d: dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = str(d.get(k) or "").strip()
        if v:
            return v
    return ""


def _render_search(data: dict[str, Any]) -> str:
    """把云检索结果拼成简洁的记忆文本块。

    **优先取 profiles 的 profile_data.embed_text** —— 这是云端提炼的、干净的
    中文用户画像（如「宠物: 养了一只名叫团子的猫」「学业: 正在准备考研」），
    最适合注入中文 prompt，且带 score 排序。

    episode 的 summary/episode 是英文且存在说话人名字错位（云端 LLM 会把
    user 内容摘要成「Lina told...」），**不注入**，仅在没有任何 profile 时
    用 atomic_fact 兜底。总长截断到 MAX_RENDER_CHARS。
    """
    lines: list[str] = []

    # 1) profiles —— 主力，中文 embed_text。
    profiles = data.get("profiles") or []

    def _profile_score(p: Any) -> float:
        try:
            return float(p.get("score") or 0.0)
        except Exception:
            return 0.0

    for p in sorted([p for p in profiles if isinstance(p, dict)], key=_profile_score, reverse=True):
        pd = p.get("profile_data")
        txt = ""
        if isinstance(pd, dict):
            txt = str(pd.get("embed_text") or "").strip()
        if not txt:
            txt = _first_nonempty(p, "content", "summary")
        if txt:
            lines.append(f"- {txt}")

    # 2) 没有任何 profile 时，用 episode 的 atomic_fact 兜底（英文，聊胜于无）。
    if not lines:
        for ep in data.get("episodes") or []:
            if not isinstance(ep, dict):
                continue
            for af in ep.get("atomic_facts") or []:
                if isinstance(af, dict):
                    fact = str(af.get("atomic_fact") or af.get("content") or "").strip()
                    if fact:
                        lines.append(f"- {fact}")

    # 3) 仍为空，兜底 raw_messages 原文。
    if not lines:
        for rm in data.get("raw_messages") or []:
            if isinstance(rm, dict):
                txt = str(rm.get("content") or "").strip()
                if txt:
                    lines.append(f"- {txt}")

    if not lines:
        return ""

    out: list[str] = []
    total = 0
    for ln in lines:
        if total + len(ln) > MAX_RENDER_CHARS:
            break
        out.append(ln)
        total += len(ln) + 1
    return "\n".join(out)
