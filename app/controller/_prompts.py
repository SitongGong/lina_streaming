"""Tiny prompt-template loader for the controller.

读 `prompts/` 下的 .txt/.md 模板。在磁盘读取之上叠加一个**全局 override 层**：
网页「提示词」编辑器保存的覆盖内容会注册到这里，load_prompt 优先返回它，
没有才读磁盘。这样 controller 的 few-shot / module / 模板等文件也能被网页改、
且即时生效（无需重启）。

override 的 key = 相对 `prompts/` 的路径，如 "controller/fewshot/comfort.txt"。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# 进程级全局 override 层。网页保存提示词时由 web 层写入；为空时完全等价于读磁盘。
_OVERRIDES: dict[str, str] = {}


def set_prompt_overrides(overrides: dict[str, str]) -> None:
    """整体替换 override 层（网页保存/切换版本时调用）。只接受 prompts/ 下的相对路径 key。"""
    global _OVERRIDES
    _OVERRIDES = {str(k): str(v) for k, v in (overrides or {}).items()}


def get_prompt_overrides() -> dict[str, str]:
    return dict(_OVERRIDES)


@lru_cache(maxsize=128)
def _read_disk(rel_path: str) -> str:
    fp = _PROMPTS_DIR / rel_path
    if not fp.exists():
        return ""
    return fp.read_text(encoding="utf-8")


def read_prompt_file(rel_path: str) -> str:
    """直接读磁盘原始内容（不经 override），给网页取「默认值」用。"""
    return _read_disk(rel_path)


def load_prompt(rel_path: str) -> str:
    """加载模板：先查全局 override 层，没有再读磁盘。缺失返回空串。"""
    if rel_path in _OVERRIDES:
        return _OVERRIDES[rel_path]
    return _read_disk(rel_path)


# ---- JSON 文件内某个字段的细粒度加载 / override ----
# override key 形如 "controller/advisor_rules.json#lenient_typos.decision_rules"
# （'#' 后是 JSON 内部的点路径）。网页编辑器按这种 key 存覆盖，专家可单独改一条规则。

import json as _json  # noqa: E402


@lru_cache(maxsize=64)
def _read_json(rel_path: str) -> dict:
    txt = _read_disk(rel_path)
    if not txt:
        return {}
    try:
        data = _json.loads(txt)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def _dig(data: dict, dotpath: str):
    cur = data
    for part in dotpath.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def make_json_key(rel_path: str, dotpath: str) -> str:
    return f"{rel_path}#{dotpath}"


def read_json_value(rel_path: str, dotpath: str, fallback: str = "") -> str:
    """直接读磁盘 JSON 里 dotpath 处的值（不经 override），给网页取默认值用。"""
    v = _dig(_read_json(rel_path), dotpath)
    return str(v) if v is not None else fallback


def load_json_value(rel_path: str, dotpath: str, fallback: str = "") -> str:
    """加载 JSON 字段：先查 override（key = 'rel_path#dotpath'），再读磁盘。缺失返回 fallback。"""
    key = make_json_key(rel_path, dotpath)
    if key in _OVERRIDES:
        return _OVERRIDES[key]
    v = _dig(_read_json(rel_path), dotpath)
    return str(v) if v is not None else fallback
