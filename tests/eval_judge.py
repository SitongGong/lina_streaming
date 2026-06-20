"""大规模综合评测：多剧本 × LLM 裁判自动打分。

覆盖维度（用户要求）：
  D1 日记/经历召回准确    D2 话题/偏好召回准确    D3 共情用户不抢话
  D4 记住用户偏好(长程)   D5 连续追问接续/不编造  D6 新会话诚实不装熟

每个剧本是一段连续对话（同 session）。每轮可带「期望」给 judge 参考。
judge（Claude sonnet）按 0/1/2 给每轮打分 + 一句理由：
  2=完全符合期望  1=基本可以但有瑕疵  0=明显问题（编造/装熟/抢话/答非所问/前后矛盾）
跑完出：各维度均分 + 所有 0/1 分轮次清单（供定位）。

跑法：服务在 8080 + 本地 EverOS 8090（带 EVEROS_DIARY=1 EVEROS_PERSONA=1）。
      python tests/eval_judge.py            # 全部剧本
      python tests/eval_judge.py D1 D4       # 指定维度
"""
from __future__ import annotations
import sys, time, json, os
from pathlib import Path
import httpx

# 先加载 .env（拿 ANTHROPIC key 给 judge / 被测服务同源），否则 judge 无 key。
def _load_env():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

BASE = "http://127.0.0.1:8080"
LOG = "/tmp/lina_evermemos.log"

# 一个【连贯长对话】剧本（~55 轮，同 session），把所有维度编织进真实对话流：
# 开场→埋用户偏好→闲聊→问莉娜经历(专名)→共情→泛问偏好→连续追问→长程回忆校验。
# 维度标签用于分维统计；None=无特定期望只看自然合理。
SCRIPT_LONG = [
    # —— 开场 + 埋用户信息（D4 长程记忆锚点）——
    ("D6", "你好呀，还记得我是谁吗", "全新会话应诚实说想不起来，不装熟"),
    ("-", "我叫江野，做游戏策划的", "记下名字职业"),
    ("-", "我猫叫年糕，一只布偶", "记下宠物"),
    ("-", "我最爱吃辣，无辣不欢", "记下口味偏好"),
    ("-", "对了我不喝酒，过敏", "记下禁忌"),
    # —— 问莉娜经历（D1 专名/日记召回）——
    ("D1", "你跟培根老师什么关系", "师徒，基于日记"),
    ("D5", "他现在还好吗", "接续，昏迷/没醒，不编好转"),
    ("D5", "你上次去看他是什么时候", "接续，基于日记"),
    ("D1", "安娜是谁呀", "工房佣人，基于日记"),
    ("D5", "她平时都帮你做什么", "接续安娜，具体"),
    ("D1", "你有没有遇到过特别难、特别委屈的事", "讲日记真实困境(工房被收回/盘查)"),
    ("D5", "后来你是怎么扛过来的", "接续，前后一致"),
    # —— 泛问偏好（D2 话题卡）——
    ("D2", "你平时爱吃什么", "偏好，自然有质感"),
    ("D2", "你喜欢什么季节", "偏好"),
    ("D2", "一个人在工房会无聊吗", "独处感受，有生活"),
    ("D2", "你有没有讨厌的人", "可聊伦茨等或含蓄"),
    # —— 共情（D3 不抢话）——
    ("D3", "唉，说点别的，我最近项目被毙了好烦", "先共情用户，不抢话讲自己"),
    ("D3", "感觉自己做什么都不行", "安抚，不附和贬低"),
    ("D3", "你有没有过这种全盘否定自己的时候", "可带自己经历但仍关注用户"),
    ("D3", "谢谢你，听我说这些", "温和"),
    # —— 闲聊噪声（拉开距离，压长程记忆）——
    ("-", "今天天气还行", None),
    ("-", "哈哈", None),
    ("-", "你那边几点了", None),
    ("-", "嗯嗯", None),
    ("-", "随便聊聊吧", None),
    # —— 长程记忆校验（D4：前面埋的，现在隔了二十多轮问）——
    ("D4", "你还记得我是做什么工作的吗", "游戏策划"),
    ("D4", "我家猫叫什么来着", "年糕/布偶"),
    ("D4", "我能喝酒吗", "不能，过敏"),
    ("D4", "我口味重还是淡来着", "重/爱辣"),
    # —— 再问莉娜经历，看绕回是否一致（D5）——
    ("D5", "对了，培根老师现在到底怎么样了", "和前面一致，昏迷"),
    ("D5", "你之前说的那个被为难的事，再讲讲", "接续前面困境，一致"),
    # —— 收尾 ——
    ("D3", "今天聊得挺好的，谢谢你", "温和"),
    ("D6", "我先走了，改天聊", "得体收尾"),
]


def build_scripts(turns_target):
    """把长剧本铺成 (dim, name, turns) 单元；turns_target 控制总轮数（重复填充噪声达到上百轮）。"""
    base = [(d, u, e) for d, u, e in SCRIPT_LONG]
    return [("LONG", f"连贯长对话×{len(base)}轮", base)]


SCRIPTS = build_scripts(0)

JUDGE_SYS = """你是对话质量评审。被测角色「西比莉娜(莉娜)」是1760年代的炼金术学徒数字人，
有自己的日记/经历，要：基于真实记忆回答不编造、记住用户说过的事、共情用户时先关注用户
不抢着讲自己、对不认识的用户诚实不装熟。
给你：用户这句话、莉娜的回复、本轮的期望(可能为空)、(可选)本轮检索到的日记线索。
按 0/1/2 打分：2=完全符合期望且自然；1=基本可以但有小瑕疵；0=明显问题
(编造与日记冲突的事实/装熟/抢话/答非所问/前后矛盾/该记得却忘了)。
只输出 JSON：{"score":0|1|2,"reason":"一句话理由"}"""


def judge_client():
    sys.path.insert(0, ".")
    from app.config import resolve_controller_settings
    cfg = resolve_controller_settings()
    from openai import OpenAI
    # 用 anthropic 兼容端点起同步 client 做裁判（sonnet 更会判）
    base = cfg.get("base_url") or "https://api.anthropic.com/v1/"
    return OpenAI(api_key=cfg["api_key"], base_url=base), "claude-sonnet-4-6"


def judge(cl, model, user, reply, expect, diary):
    prompt = (f"用户：{user}\n莉娜：{reply}\n期望：{expect or '（无特定期望，看是否自然合理）'}\n"
              f"检索线索(可能不全)：{diary[:300] or '无'}\n"
              "注意：检索线索只是片段、未必含全部依据。**只有当回答与莉娜设定/常识明显矛盾"
              "时才判编造**；线索没提到但合理的细节不算编造。")
    for _ in range(2):
        try:
            r = cl.chat.completions.create(
                model=model, max_tokens=200,
                messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": prompt}],
            )
            raw = (r.choices[0].message.content or "").strip()
            i, j = raw.find("{"), raw.rfind("}")
            if i >= 0 and j > i:
                d = json.loads(raw[i:j+1])
                return int(d.get("score", 1)), str(d.get("reason", ""))
        except Exception:
            time.sleep(0.5)
    return -1, "judge_err"


def _diary_since(pos):
    try:
        lines = open(LOG, encoding="utf-8", errors="ignore").readlines()[pos:]
    except Exception:
        return ""
    import re
    for ln in lines:
        if "[diary]" in ln:
            return re.sub(r"\x1b\[[0-9;]*m", "", ln).split("[diary]")[-1].strip()
    return ""


def _logpos():
    try:
        return sum(1 for _ in open(LOG, encoding="utf-8", errors="ignore"))
    except Exception:
        return 0


def main():
    only = set(sys.argv[1:])
    cl, jm = judge_client()
    # 自动存档：每次跑把完整输出写到 tests/eval_runs/<时间戳>.txt（留档可追溯）。
    runs_dir = Path(__file__).resolve().parent / "eval_runs"
    runs_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    archive = open(runs_dir / f"run_{stamp}.txt", "w", encoding="utf-8")
    import builtins as _bi
    def print(*a, **k):  # noqa: A001 — 同时输出到终端和存档文件
        _bi.print(*a, **k)
        _bi.print(*a, **{**k, "file": archive, "flush": True})
    by_dim = {}
    lows = []
    for dim, name, turns in SCRIPTS:
        if only and dim not in only:
            continue
        cid = f"judge_{dim}_{int(time.time()*1000)%100000}"
        h = {"X-Client-Id": cid, "Content-Type": "application/json"}
        sc = httpx.Client(timeout=60)
        sc.post(f"{BASE}/api/register", json={"username": cid, "password": "test123456"}, headers=h)
        sc.post(f"{BASE}/api/auth", json={"api_key": "0"}, headers=h)
        sid = f"s_{cid}"
        print(f"\n### {name}（{len(turns)} 轮，judge={jm}）", flush=True)
        _d4_waited = False
        for idx, (tdim, user, expect) in enumerate(turns, 1):
            # 长程记忆校验段(D4)之前补足 EverMem 云端建索引延迟（实测写入→可检索约 5-8s）。
            # 真实用户两轮间隔远大于此，无影响；只有这种快速自动评测会撞上索引窗口，
            # 不补等待会把「还没建好索引」误判成「莉娜忘了用户事实」。仅测试侧等待，不改任何线上逻辑。
            if tdim == "D4" and not _d4_waited:
                print("        … 等 EverMem 云端建索引(15s)再做长程记忆校验 …", flush=True)
                time.sleep(15)
                _d4_waited = True
            pos = _logpos()
            try:
                rp = sc.post(f"{BASE}/api/chat", json={"message": user, "session_id": sid}, headers=h).json()
                reply = (rp.get("reply") or "").replace("\n", " ")
            except Exception as e:
                reply = f"(请求失败:{e})"
            diary = _diary_since(pos)
            # tdim == "-" 的是铺垫/噪声轮，仍打分但归到 "闲聊" 维度
            score_dim = tdim if tdim != "-" else "闲聊"
            s, reason = judge(cl, jm, user, reply, expect, diary)
            by_dim.setdefault(score_dim, []).append(s)
            flag = {2: "✓", 1: "△", 0: "✗", -1: "?"}.get(s, "?")
            print(f"  {idx:02d}[{flag}{s}|{score_dim}] U:{user[:22]} | L:{reply[:46]}", flush=True)
            if s <= 1:
                print(f"        ↳ {reason}", flush=True)
                lows.append((score_dim, user, reply, reason, diary[:60]))
            time.sleep(0.3)
    print("\n===== 维度均分 =====")
    for dim, scores in by_dim.items():
        valid = [s for s in scores if s >= 0]
        avg = sum(valid)/len(valid) if valid else 0
        print(f"  {dim}: {avg:.2f}/2  ({len(valid)}轮)")
    print(f"\n低分轮次 {len(lows)} 条（已在上面标 △/✗）")


if __name__ == "__main__":
    main()
