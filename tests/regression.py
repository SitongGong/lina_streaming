"""Failure case 多轮回归测试（对应 docs/FAILURE_CASES.md）。

每个 case = 一段多轮对话 + 对某些轮的断言（回复里**该出现/不该出现**的关键词，
或 plan 字段、[diary] 决策）。每次改完一键全量跑，确认旧 case 不退化、新 case 真解决。

前提：服务在 8080 + 本地 EverOS 8090 跑着（带 EVEROS_DIARY=1 EVEROS_PERSONA=1）。
跑法：python tests/regression.py            # 全部
      python tests/regression.py C6 C7 C8   # 只跑指定 case

断言是**软断言**（LLM 输出有随机性）：不通过只标记 ⚠ 并打印实际回复，供人判断，
不当硬失败——重点是把所有 case 一次性跑出来让人扫一眼，而非 CI 红绿。
每个 case 用独立 session（连续性 case 在同一 session 内多轮）。
"""
from __future__ import annotations
import sys, time, re, json
import httpx

BASE = "http://127.0.0.1:8080"
LOG = "/tmp/lina_evermemos.log"

# case: (id, 说明, [(用户句, 断言)])
# 断言 dict: {"has": [...任一出现即过], "no": [...都不出现才过], "diary": "标记子串"}
# 断言为 None = 只发不判（铺垫轮）
CASES = [
    ("C2_问经历不回避", [
        ("你能给我讲讲你的过去吗，最近都经历了些什么",
         {"no": ["没什么好说", "就这样", "你想听哪方面", "没什么特别"]}),
    ]),
    ("C5_倾诉先共情不抢话", [
        ("我最近工作真的很累，心很烦", {"no": ["我也", "我那时", "我之前"]}),
    ]),
    ("C6_代词追问接续", [
        ("你跟你的老师培根关系怎么样", None),
        ("他还好吗", {"has": ["昏迷", "不太好", "撑", "醒", "病", "睡", "誊", "医院"], "no": ["你是谁", "不认识"]}),
    ]),
    ("C7_叙事经历检索", [
        ("你有没有去过首都那种大城市", None),
        # 同一件事的合理表述：盘查官/配方纸/安全嫌疑/头衔/堵门口…任一即算命中那段经历
        ("进城的时候被为难过吗", {"has": ["盘查", "铁尺", "箱扣", "配方", "符号", "头衔", "安全嫌疑", "堵", "城门"]}),
    ]),
    ("C8_突然切话题不稀释", [
        ("你工房里都有什么工具", None),
        ("培根老师现在怎么样了", {"has": ["昏迷", "不太好", "撑", "誊", "醒", "睡", "病", "医院"], "no": ["还好吧"]}),
    ]),
    # C10 看的是「前后一致」（同话题跨轮不矛盾），不强求特定状态词——培根近况的合理表述
    # 有「昏迷/还在睡/医院/醒了一会儿/誊抄」等多种，只要都在讲他的真实近况即可。
    # 真正的「矛盾」（这轮好转下轮昏迷）靠人看 diary 是否锁定同篇判断。
    ("C10_连问人物前后一致", [
        ("培根老师现在怎么样了", {"has": ["昏迷", "睡", "医院", "醒", "誊", "撑", "不太好"]}),
        ("他还好吗", {"has": ["昏迷", "睡", "医院", "醒", "誊", "撑", "不太好"]}),
        ("我想了解他的近况", {"has": ["昏迷", "睡", "医院", "醒", "誊", "撑", "不太好"]}),
    ]),
    ("C11_寒暄不误触发", [
        ("你好呀", {"diary_no": True}),   # diary_no: 这轮不该检索
        ("你那边天气怎么样", {"diary_yes": True}),
    ]),
    ("C1_间接告别停主动", [
        ("我还有点事，咱们改天再聊吧", {"farewell": True}),
    ]),
    # C13 分类：偏好类问题该走话题卡（diary 命中里有 A/B 话题卡）；专名类该重点日记直检
    ("C13_偏好类走话题卡", [
        ("你平时爱吃什么", {"diary_yes": True}),   # global，应命中口味/食物话题卡
    ]),
    ("C13_专名类走日记", [
        ("安娜和你是什么关系", {"has": ["安娜", "工房", "佣", "汤", "照", "一起"]}),  # detail，日记直检
    ]),
]


def run_case(c, cid_base):
    cid = f"reg_{cid_base}_{int(time.time()*1000)%100000}"
    h = {"X-Client-Id": cid, "Content-Type": "application/json"}
    cl = httpx.Client(timeout=60)
    cl.post(f"{BASE}/api/register", json={"username": cid, "password": "test123456"}, headers=h)
    cl.post(f"{BASE}/api/auth", json={"api_key": "0"}, headers=h)
    sid = f"s_{cid}"
    results = []
    for turn, (msg, assertion) in enumerate(c[1], 1):
        logpos = _log_lines()
        try:
            r = cl.post(f"{BASE}/api/chat", json={"message": msg, "session_id": sid}, headers=h)
            d = r.json()
            reply = (d.get("reply") or "").replace("\n", " ")
            plan = d.get("plan") or {}
        except Exception as e:
            results.append((msg, "ERR", str(e), ""))
            continue
        diary_line = _diary_since(logpos)
        verdict = _check(assertion, reply, plan, diary_line)
        results.append((msg, verdict, reply, diary_line))
        time.sleep(0.3)
    return results


def _log_lines():
    try:
        return sum(1 for _ in open(LOG, encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


def _diary_since(pos):
    try:
        lines = open(LOG, encoding="utf-8", errors="ignore").readlines()[pos:]
    except Exception:
        return ""
    for ln in lines:
        if "[diary]" in ln:
            return re.sub(r"\x1b\[[0-9;]*m", "", ln).split("[diary]")[-1].strip()
    return ""


def _check(a, reply, plan, diary):
    if a is None:
        return "·"  # 铺垫轮
    if "has" in a and not any(k in reply for k in a["has"]):
        return "⚠ 缺" + "/".join(a["has"][:3])
    if "no" in a and any(k in reply for k in a["no"]):
        bad = [k for k in a["no"] if k in reply]
        return "⚠ 现" + "/".join(bad)
    if a.get("farewell") and not plan.get("user_farewell"):
        return "⚠ farewell未判出"
    if a.get("diary_no") and ("检索" in diary or "接续" in diary):
        return "⚠ 不该检索却检索了"
    if a.get("diary_yes") and "不需日记" in diary:
        return "⚠ 该检索却没检索"
    return "✓"


def main():
    only = set(sys.argv[1:])
    print(f"=== 回归测试 {time.strftime('%H:%M:%S')} ===\n")
    npass = nwarn = 0
    for c in CASES:
        cid = c[0].split("_")[0]
        if only and cid not in only and c[0] not in only:
            continue
        print(f"### {c[0]}")
        for msg, v, reply, diary in run_case(c, cid):
            mark = v
            if v == "✓": npass += 1
            elif v.startswith("⚠") or v == "ERR": nwarn += 1
            print(f"  [{mark}] 用户: {msg}")
            print(f"        莉娜: {reply[:90]}")
            if diary:
                print(f"        diary: {diary[:80]}")
        print()
    print(f"=== 通过 {npass} | 需看 {nwarn} ===")


if __name__ == "__main__":
    main()
