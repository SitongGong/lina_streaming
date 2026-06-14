#!/usr/bin/env python3
"""Controller 分类准确性评测。

读取 cases.jsonl（人工标注的期望场景），逐条跑真实 controller，对比：
- 规则层：matched_rule 是否等于 expect_rule（"" 表示期望走 LLM/不命中规则）
  → 每个场景的 precision / recall / F1 + 混淆矩阵。
- flag 级：expect_flags 里写明的字段是否判对（suppress/lenient/use_world…）。
- trace_source 分布：rule / llm / fallback 各占多少（fallback 高 = 分类能力没用上）。
- 一致性（--consistency N）：同一条跑 N 次，看 LLM 路径判定稳不稳
  （不稳本身就是"是否需要训练"的信号）。

用法：
    export OPENAI_API_KEY=...        # 否则 LLM 路径全走 fallback，只能测规则层
    python tests/controller/run_eval.py
    python tests/controller/run_eval.py --consistency 3   # 额外测一致性
    python tests/controller/run_eval.py --only-rules      # 只测规则层（不调 LLM，快）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 让脚本能 import app.*（项目根加入 path）
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 复刻 run_web 的 .env 加载（否则 controller 没 key → 全 fallback）
def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            cases.append(json.loads(line))
    return cases


def build_ctx(case: dict):
    from app.controller.schema import LinaTurnContext
    return LinaTurnContext(
        user_text=case.get("user_text", ""),
        history=tuple(tuple(p) for p in case.get("history", [])),
        is_proactive=case.get("is_proactive", False),
        is_farewell=case.get("is_farewell", False),
        is_continuation=case.get("is_continuation", False),
        gap_seconds=case.get("gap_seconds", 0.0),
        prior_trust=case.get("prior_trust", 3),
        has_cross_session_memory=case.get("has_cross_session_memory", False),
    )


# expect_rule（规则层场景名）→ LLM 路径下「正确判别」应开的那个 module/flag。
# 用于 --force-llm 模式：把规则层场景翻译成"LLM 应该点亮哪个模块"，从而能评 LLM 区分力。
RULE_TO_MODULE = {
    "user_vent": "module_user_vent",
    "modern_action_request": "module_action_boundary",
    "world_immersion": "module_world_immersion",
    "self_introspection": "module_self_introspection",
    "relationship_recall": "module_relationship_recall",
    # 下面这些是规则层专属（问候/告别/短反应/久别/续说/主动/空），LLM 路径没有对应
    # 的单一 module，区分力主要看长度/语气/flag，不在 module 命中里评。
}


def llm_scene_signature(plan) -> set:
    """LLM 这轮点亮了哪些"场景模块"。用于 force-llm 下判别它把输入归到了哪类。"""
    on = set()
    for m in ("module_user_vent", "module_action_boundary", "module_world_immersion",
              "module_self_introspection", "module_relationship_recall"):
        if getattr(plan, m, False):
            on.add(m)
    return on


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(Path(__file__).parent / "cases.jsonl"))
    ap.add_argument("--consistency", type=int, default=0, help="同条跑 N 次测稳定性")
    ap.add_argument("--only-rules", action="store_true", help="只测规则层，不构造 LLM controller")
    ap.add_argument("--force-llm", action="store_true",
                    help="屏蔽规则层，强制所有 case 走 LLM fan-out——压测 LLM 的区分能力")
    args = ap.parse_args()

    _load_dotenv()
    from app.controller import build_default_controller
    from app.config import resolve_openai_api_key

    if args.only_rules:
        # 规则层独立测：不给 key，has_llm=False，但规则层照常跑
        ctrl = build_default_controller(api_key=None)
    else:
        ctrl = build_default_controller(api_key=resolve_openai_api_key())
    if args.force_llm:
        # 屏蔽规则层：所有 case 都走 LLM fan-out，纯压测 gpt-5-mini 的场景区分能力。
        ctrl._rule_router.route = lambda ctx: None
        print(">>> FORCE-LLM 模式：规则层已屏蔽，全部走 LLM fan-out")
    print(f"controller has_llm = {ctrl.has_llm}\n" + "=" * 70)

    cases = load_cases(Path(args.cases))

    # 规则层混淆：以 matched_rule 为预测。期望 "" 视作类别 <LLM>。
    confusion: dict[str, Counter] = defaultdict(Counter)
    rule_labels = set()
    src_counter = Counter()
    flag_stats = Counter()      # f"{field}:ok" / f"{field}:bad"
    flag_errors: list[str] = []
    rule_errors: list[str] = []

    # force-llm 专用统计：LLM 是否把"有明确场景"的输入归到了正确的 module。
    llm_scene_hit = 0
    llm_scene_total = 0
    llm_scene_errors: list[str] = []

    for case in cases:
        ctx = build_ctx(case)
        plan = ctrl.dispatch_sync(ctx)
        src_counter[plan.trace_source] += 1

        # --- force-llm：评 LLM 把输入归到了哪个场景模块 ---
        if args.force_llm:
            exp_rule = case.get("expect_rule") or ""
            want_mod = RULE_TO_MODULE.get(exp_rule)
            if want_mod:  # 只对能映射到 module 的明确场景计分
                llm_scene_total += 1
                got_mods = llm_scene_signature(plan)
                if want_mod in got_mods:
                    llm_scene_hit += 1
                else:
                    llm_scene_errors.append(
                        f"  [{case['id']}] 应开 {want_mod}，实开 {sorted(got_mods) or '无'} "
                        f"| {case.get('note','')} | u={case.get('user_text','')!r}"
                    )

        exp = case.get("expect_rule", None)
        got = plan.matched_rule or "<LLM>"
        if exp is not None:
            exp_label = exp or "<LLM>"
            rule_labels.add(exp_label); rule_labels.add(got)
            confusion[exp_label][got] += 1
            if got != exp_label:
                rule_errors.append(
                    f"  [{case['id']}] 期望={exp_label} 实得={got} | {case.get('note','')} | u={case.get('user_text','')!r}"
                )

        for field, want in (case.get("expect_flags") or {}).items():
            actual = getattr(plan, field, None)
            ok = (actual == want)
            flag_stats[f"{field}:{'ok' if ok else 'bad'}"] += 1
            if not ok:
                flag_errors.append(
                    f"  [{case['id']}] {field}: 期望={want} 实得={actual} | u={case.get('user_text','')!r}"
                )

    # ---- 规则层 P/R/F1 ----
    print("\n### 规则层 每场景 Precision/Recall/F1")
    print(f"{'场景':22} {'P':>6} {'R':>6} {'F1':>6}  支持")
    labels = sorted(rule_labels)
    macro_f = []
    for lab in labels:
        tp = confusion[lab][lab]
        fn = sum(v for k, v in confusion[lab].items() if k != lab)
        fp = sum(confusion[other][lab] for other in labels if other != lab)
        support = sum(confusion[lab].values())
        p, r, f = prf(tp, fp, fn)
        if support:
            macro_f.append(f)
        print(f"{lab:22} {p:6.2f} {r:6.2f} {f:6.2f}  {support}")
    if macro_f:
        print(f"{'— macro F1 —':22} {'':>6} {'':>6} {sum(macro_f)/len(macro_f):6.2f}")

    # ---- 混淆矩阵（只打有错的行）----
    print("\n### 混淆矩阵（期望→实得，仅列有误分的）")
    for exp_label in labels:
        row = confusion[exp_label]
        wrong = {k: v for k, v in row.items() if k != exp_label}
        if wrong:
            print(f"  期望 {exp_label}: " + ", ".join(f"{k}×{v}" for k, v in row.items()))

    # ---- trace 来源 ----
    print("\n### trace_source 分布")
    total = sum(src_counter.values())
    for s, c in src_counter.most_common():
        print(f"  {s:10} {c:3}  ({c/total*100:.0f}%)")

    # ---- flag 准确 ----
    if flag_stats:
        print("\n### flag 级判定（expect_flags 标注的字段）")
        fields = sorted({k.rsplit(":", 1)[0] for k in flag_stats})
        for fld in fields:
            ok = flag_stats[f"{fld}:ok"]; bad = flag_stats[f"{fld}:bad"]
            print(f"  {fld:28} {ok}/{ok+bad} 对")

    # ---- 错误明细 ----
    if rule_errors:
        print(f"\n### 规则层误分类明细（{len(rule_errors)} 条）")
        print("\n".join(rule_errors))
    if flag_errors:
        print(f"\n### flag 判错明细（{len(flag_errors)} 条）")
        print("\n".join(flag_errors))

    # ---- force-llm：LLM 场景区分准确率 ----
    if args.force_llm and llm_scene_total:
        acc = llm_scene_hit / llm_scene_total
        print(f"\n### 【压测】LLM 场景区分准确率（规则层屏蔽，纯 LLM 判别）")
        print(f"  明确场景命中：{llm_scene_hit}/{llm_scene_total} = {acc*100:.1f}%")
        if llm_scene_errors:
            print(f"  判错 {len(llm_scene_errors)} 条：")
            print("\n".join(llm_scene_errors))

    # ---- 一致性 ----
    if args.consistency > 1 and ctrl.has_llm:
        print(f"\n### LLM 一致性（每条跑 {args.consistency} 次，看 matched_rule+关键flag 是否每次相同）")
        unstable = []
        for case in cases:
            if case.get("expect_rule"):
                continue  # 规则命中的是确定性的，不用测
            sigs = set()
            for _ in range(args.consistency):
                plan = ctrl.dispatch_sync(build_ctx(case))
                sigs.add((plan.matched_rule, plan.use_self_facts, plan.module_world_immersion,
                          plan.suppress_trailing_question, plan.lenient_typos))
            if len(sigs) > 1:
                unstable.append(f"  [{case['id']}] {len(sigs)} 种结果 | u={case.get('user_text','')!r}")
        if unstable:
            print(f"  不稳定 {len(unstable)} 条：")
            print("\n".join(unstable))
        else:
            print("  全部稳定 ✅")

    print("\n" + "=" * 70)
    print("解读：规则层看 P/R/F1+混淆矩阵；走LLM的看 flag准确+一致性；")
    print("fallback% 高=分类能力没用上；不稳定条=训练候选。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
