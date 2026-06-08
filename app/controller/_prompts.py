"""Tiny prompt-template loader for the controller.

11mio ships a heavier `PromptLoader` with namespaces and caching; we just
need to read a handful of .txt files from `prompts/`. Kept inside the
controller package so the rest of lina has no new global concept.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@lru_cache(maxsize=64)
def load_prompt(rel_path: str) -> str:
    """Load a prompt template by path relative to `prompts/`.

    Returns the empty string if the file is missing — callers that need
    strictness can check for that.
    """
    fp = _PROMPTS_DIR / rel_path
    if not fp.exists():
        return ""
    return fp.read_text(encoding="utf-8")
