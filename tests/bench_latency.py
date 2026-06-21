"""三档配置延迟对比：无 controller / 加 controller / 加 controller+EverMem+日记。

对同一组 case，分别用三种 CharacterEngine 各跑一遍，统计每档的总耗时与分阶段耗时
（controller+判别器 / 检索 / 主模型推理），看每加一层各付出多少延迟。

  无 controller        : controller=None，无日记、无 EverMem（最快基线）
  +controller          : 接 controller（20 路 advisor 判定），日记/EverMem 关
  +controller+EverMem  : controller + 本地日记检索 + EverMem 用户记忆（全功能）

跑法（服务不用起，脚本直接构造引擎）：
  cd /root/lina_evermemos
  python tests/bench_latency.py
依赖：.env 里的 ANTHROPIC key；第三档需本地 EverOS(8090) + EVERMEM_API_KEY。
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# 同一组 case：覆盖 寒暄/共情/问经历(专名)/追问/偏好 —— 跨场景才有代表性。
CASES = [
    "你好呀",
    "我最近工作好累，心很烦",
    "你跟培根老师什么关系",
    "他现在还好吗",
    "安娜平时帮你做什么",
    "你平时爱吃什么",
]


def _new_conv():
    from app.conversation import Conversation
    return Conversation(session_id="bench", messages=[])


def run_config(name: str, *, with_controller: bool, with_memory: bool, cases) -> dict:
    """构造一档引擎，对 cases 逐条跑，累计分阶段耗时。返回 {总, 各阶段, 每条}。"""
    from app.character import CharacterEngine
    from app.config import resolve_controller_settings

    # 第三档才让日记/EverMem 生效；前两档强制关，保证只测「加 controller」这一层。
    if with_memory:
        os.environ["EVEROS_DIARY"] = "1"
        os.environ["EVEROS_PERSONA"] = "1"
    else:
        os.environ["EVEROS_DIARY"] = ""
        os.environ["EVEROS_PERSONA"] = ""

    controller = None
    if with_controller:
        from app.controller.controller import build_default_controller
        cfg = resolve_controller_settings()
        controller = build_default_controller(
            api_key=cfg["api_key"], model=cfg["model"],
            base_url=cfg.get("base_url"), provider=cfg.get("provider"),
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_KEY") or ""
    engine = CharacterEngine(
        api_key=api_key, static_dir=ROOT / "static",
        controller=controller,
    )
    # 第三档：若 web 层会传 user_memory_text（EverMem 检索结果），这里也模拟取一次。
    mem_client = None
    if with_memory:
        try:
            from app.memory_client import MemoryClient
            mem_client = MemoryClient()
            if not mem_client.enabled:
                mem_client = None
        except Exception:
            mem_client = None

    agg = {"controller_ms": 0.0, "retrieval_ms": 0.0, "inference_ms": 0.0}
    rows = []
    for msg in cases:
        conv = _new_conv()
        user_memory_text = ""
        if mem_client is not None:
            try:
                user_memory_text = mem_client.search_text("bench::default", msg, top_k=10)
            except Exception:
                user_memory_text = ""
        t0 = time.perf_counter()
        res = engine.chat(conv, msg, user_memory_text=user_memory_text)
        wall = round((time.perf_counter() - t0) * 1000, 1)
        tm = res.timings or {}
        for k in agg:
            agg[k] += float(tm.get(k) or 0)
        rows.append((msg, wall, tm))
        print(f"    [{msg[:14]:<14}] 墙钟 {wall:>7.1f}ms  "
              f"ctrl {tm.get('controller_ms', 0):>6}  "
              f"检索 {tm.get('retrieval_ms', 0):>6}  "
              f"推理 {tm.get('inference_ms', 0):>7}")
    n = len(cases)
    avg = {k: round(v / n, 1) for k, v in agg.items()}
    avg["total_ms"] = round(sum(avg.values()), 1)
    return {"name": name, "avg": avg, "rows": rows}


def main():
    _load_env()
    configs = [
        ("无 controller", dict(with_controller=False, with_memory=False)),
        ("+controller", dict(with_controller=True, with_memory=False)),
        ("+controller+EverMem+日记", dict(with_controller=True, with_memory=True)),
    ]
    results = []
    for name, kw in configs:
        print(f"\n### {name}（{len(CASES)} case）")
        try:
            results.append(run_config(name, cases=CASES, **kw))
        except Exception as e:
            print(f"    ✗ 失败: {e}")

    print("\n" + "=" * 64)
    print(f"{'配置':<26}{'controller':>11}{'检索':>9}{'推理':>10}{'单轮均值':>11}")
    print("-" * 64)
    base = None
    for r in results:
        a = r["avg"]
        delta = ""
        if base is not None:
            delta = f"  (+{a['total_ms'] - base:.0f}ms)"
        else:
            base = a["total_ms"]
        print(f"{r['name']:<26}{a['controller_ms']:>9}ms{a['retrieval_ms']:>7}ms"
              f"{a['inference_ms']:>8}ms{a['total_ms']:>9}ms{delta}")
    print("=" * 64)
    print("注：推理时长受网络与模型负载波动影响；controller/检索是本地可控的额外开销。")


if __name__ == "__main__":
    main()
