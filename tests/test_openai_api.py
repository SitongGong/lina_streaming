#!/usr/bin/env python3
"""用 OpenAI 官方 SDK 实测 Lina 的 /v1/chat/completions 接口。

验证：非流式、流式、多轮对话、（可选）鉴权。
跑之前确保服务已启动：python run_web.py --host 0.0.0.0 --port 8000

用法：
    python tests/test_openai_api.py
    python tests/test_openai_api.py --base http://127.0.0.1:8000/v1 --key dummy
"""
from __future__ import annotations

import argparse
import sys
import time

from openai import OpenAI


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", default="dummy", help="未启用鉴权时填任意非空值")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base, api_key=args.key)
    ok = 0
    total = 0

    # ---- 1) 非流式 ----
    total += 1
    print("=" * 60)
    print("[1] 非流式 chat.completions.create")
    t0 = time.monotonic()
    r = client.chat.completions.create(
        model="lina",
        messages=[{"role": "user", "content": "你好，简单介绍一下你自己"}],
    )
    dt = time.monotonic() - t0
    content = r.choices[0].message.content
    print(f"  回复: {content!r}")
    print(f"  id={r.id}  finish={r.choices[0].finish_reason}  耗时={dt:.1f}s")
    print(f"  usage: prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")
    assert content and r.choices[0].finish_reason == "stop"
    assert "[mood" not in content and "[segments" not in content, "不该泄露内部标记"
    ok += 1
    print("  ✅ 通过（有回复、无内部标记泄露）")

    # ---- 2) 流式 ----
    total += 1
    print("=" * 60)
    print("[2] 流式 stream=True")
    t0 = time.monotonic()
    stream = client.chat.completions.create(
        model="lina",
        messages=[{"role": "user", "content": "给我讲讲你最近鉴定的一件遗物"}],
        stream=True,
    )
    chunks = []
    first_chunk_t = None
    for ch in stream:
        d = ch.choices[0].delta.content
        if d:
            if first_chunk_t is None:
                first_chunk_t = time.monotonic() - t0
            chunks.append(d)
    full = "".join(chunks)
    print(f"  流式回复: {full!r}")
    print(f"  收到 {len(chunks)} 个文本块  首块延迟={first_chunk_t:.1f}s  总长={len(full)}字")
    assert chunks and full.strip()
    assert "[mood" not in full and "[segments" not in full
    ok += 1
    print("  ✅ 通过（流式分块返回、无内部标记泄露）")

    # ---- 3) 多轮（把历史放进 messages）----
    total += 1
    print("=" * 60)
    print("[3] 多轮对话（messages 带历史）")
    r = client.chat.completions.create(
        model="lina",
        messages=[
            {"role": "user", "content": "我养了只猫叫团子"},
            {"role": "assistant", "content": "诶，猫啊。它平时黏不黏你？"},
            {"role": "user", "content": "挺黏的，你还记得它叫什么吗？"},
        ],
    )
    content = r.choices[0].message.content
    print(f"  回复: {content!r}")
    print(f"  （检查她是否记得'团子'）含'团子': {'团子' in content}")
    assert content
    ok += 1
    print("  ✅ 通过（多轮历史被接受、能回应）")

    print("=" * 60)
    print(f"结果：{ok}/{total} 通过")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
