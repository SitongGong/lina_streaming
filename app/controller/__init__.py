"""Controller package: per-turn micro-decision layer.

Given a single turn's context (user text, history, mood, flags), the
controller produces a `LinaPromptPlan` that tells the rest of the engine
*how* to handle this turn: which RAG sources to query, how many history
turns to keep, which differentiated prompt modules to inject, how long /
what tone the reply should be, whether to require concrete examples or
callbacks, and so on.

Two layers:
- `rule_router`: deterministic regex rules for stable scenarios. 0 LLM calls.
- `controller`:  fan-out fallback. Each Plan field is judged by one tiny
                 gpt-5-mini advisor running concurrently under a hard deadline.

If neither layer can produce a Plan (no key, all advisors fail), a
fallback Plan that mirrors the engine's pre-controller behavior is used.
"""

from .schema import LinaPromptPlan, LinaTurnContext
from .controller import LinaController, build_default_controller
from .rule_router import reload_rule_patterns
from .composer import LinaPromptComposer, LinaPromptBundle
from ._prompts import (
    set_prompt_overrides,
    get_prompt_overrides,
    read_prompt_file,
    read_json_value,
    load_json_value,
    make_json_key,
)

__all__ = [
    "LinaPromptPlan",
    "LinaTurnContext",
    "LinaController",
    "build_default_controller",
    "LinaPromptComposer",
    "LinaPromptBundle",
    "set_prompt_overrides",
    "get_prompt_overrides",
    "read_prompt_file",
    "read_json_value",
    "load_json_value",
    "make_json_key",
    "reload_rule_patterns",
]
