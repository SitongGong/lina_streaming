# Lina 改造交接备忘

本文件记录 `/root/lina`（数字人「西比莉娜」测试项目）近期两批改造：
1. **Controller / 主动发言 / 切段 / 引用** 等机制（原在 `/root/lina_old`，已合并进来）
2. **新版自带的 ASR/TTS 语音** + 多客户端凭证架构

> 背景：`/root/lina_old` 是「基线 + 我们做的对话增强」；`/root/lina` 是「基线 + 语音/多客户端」。
> 本次把 `lina_old` 的增强**合并进** `lina`，并装好了语音所需的 GPU 模型。`lina_old` 仅作历史参考，运行以 `/root/lina` 为准。

---

## 一、目录 / 模块总览

```
app/
  config.py          # 路径 + key 解析（Anthropic / OpenAI）+ 主动发言节奏参数
  conversation.py    # 会话持久化（每会话一个 JSON）；含 pending_segments / client_id
  rag.py             # BM25 检索（静态人设 / 本会话历史 / 跨会话记忆）+ 时间衰减
  character.py       # 角色引擎：拼 prompt、调 Claude、切段、引用、主动发言、续说
  voice.py           # 语音引擎：ASR（听）+ TTS（说），懒加载、GPU 单例
  web.py             # Flask 服务：多客户端凭证、聊天/主动/续说/语音 端点、prompt 编辑器
  templates/chat.html# 单页前端：聊天 UI + 主动发言计时器 + 引用 UI + 语音 mic
  controller/        # ★ 本次合并：每轮「怎么回」的决策层
    schema.py        #   LinaTurnContext（输入）/ LinaPromptPlan（输出）
    rule_router.py   #   规则层：正则秒判常见场景，零 LLM
    experts.py       #   14 个 gpt-5-mini 微顾问（fan-out）
    controller.py    #   调度：规则优先 → fan-out → fallback；含 pick_proactive_topic
    composer.py      #   Plan → 动态 system 尾巴文本
    _prompts.py      #   极简 prompt 模板加载器
prompts/             # ★ 本次合并
  controller/        #   control_flag/int/text.txt（advisor 模板）+ proactive_topic.txt
  modules/           #   各场景模块文本（安抚/边界/沉浸/续说/welcome_back 等）
```

---

## 二、核心机制（设计要点）

### 1. Controller —— 每轮决策层
- **入口** `LinaController.dispatch(ctx)`：① 规则层 `rule_router` 能秒判就直接出 `LinaPromptPlan`（0 LLM）；② 判不准 → fan-out 14 个微顾问并发（每个只判一个字段），6s 总 deadline；③ 全失败/无 key → fallback Plan（≈改造前行为）。
- **永不阻塞、永不抛错**：最坏返回 fallback。
- **controller LLM = gpt-5-mini**（OpenAI），是 reasoning 模型，调用时用 `max_completion_tokens` + `reasoning_effort="minimal"`（不是 `max_tokens`/`temperature`，那俩会 400）。
- **开关**：环境变量 `LINA_CONTROLLER=off` 关掉 LLM 微顾问（规则层仍跑）；`OPENAI_API_KEY` 缺失自动降级到规则层。
- **Plan 驱动下游**：`retrieve_k / history_window / query_hint / use_static_* / 模块开关 / sentences / max_reply_chars / tone_hint / hook_*` 等，全部由 Plan 控制 RAG 和 system prompt。

### 2. 两段式 system prompt（保住 prompt cache）
- 上半块：`core_text + BEHAVIOR_RULES + MOOD_FORMAT_SPEC + SEGMENT_PROTOCOL_SPEC`，带 `cache_control`，**不变 → 命中缓存**。
- 下半块：controller 出的本轮模块/约束，每轮变，不缓存。无内容时不加（等价改造前）。

### 3. 回复切段 + 续说（解决回复过长）
- 主模型用末尾隐藏标记 `[segments: 要点A || 要点B]` 声明「还有几小段没说」。
- `parse_segments_tag` 剥出来：第一段展示，其余 park 进 `conversation.pending_segments`。
- 用户没接话 → 前端短计时器调 `/api/continue` → `engine.continue_segment()` 逐段补完；用户一发消息自动作废。
- **强制规则**：`SEGMENT_PROTOCOL_SPEC` 要求「超过一个意思就必须拆」，配合收紧的默认长度（普通闲聊 sentences≤2 / chars≤45）。

### 4. 主动发言（退避节奏 + 话头挑选 + welcome_back）
- 前端闲置轮询（每 2s）两阶段：先续说未说完的段，再主动找话。
- **节奏 = 指数退避 + 抖动**：`下次等待 = engage_base_ms × engage_multiplier^已主动次数 × (1±jitter)`，到 `max_nudges` 改告别并停。参数在 `config.resolve_proactive_pacing()`，可环境变量覆盖，经 `/api/status` 下发前端。
- **话头挑选（依据论文 MapDia #9 / PaRT #4）**：`controller.pick_proactive_topic()` 让 gpt-5-mini 从最近历史挑「最该重提、且与用户相关」的话头，驱动主动开口。失败则退回规则的静态 query_hint。
- **welcome_back**：用户离开 > 30 分钟回来（按 `Message.ts` 算 `gap_seconds`），带时间感打招呼 + 勾历史。

### 5. 检索时间衰减（依据 LD-Agent #8）
- `rag.py` 两个历史检索函数：BM25 分数 × 时间衰减（半衰期 7 天）。近期记忆优先于「碰巧词匹配的老记忆」。默认 `recency=True`。

### 6. 引用回复（微信式）
- 前端点莉娜某条消息「引用」→ 发送带 `quoted_text` → prompt 里加 `<用户引用>` 块，让模型明确承接那句。粒度=整条气泡，一次性（发完即清）。

### 7. 语音（新版自带，已装好模型）
- `voice.py`：ASR `Qwen3-ASR-1.7B`（cuda:0）+ TTS `Qwen3-TTS-12Hz-1.7B-CustomVoice`（cuda:1），懒加载、进程内单例。
- `character.py` 的 `chat_stream()`（流式）被语音路径 `/api/voice/chat` 用；它也走 controller（`_prepare` 共用）。

---

## 三、关键文件与改动点

| 文件 | 本次相关改动 |
|---|---|
| `app/controller/*`、`prompts/*` | 从 lina_old 整包移植（自包含） |
| `app/conversation.py` | 加 `pending_segments` 字段（新版已有 `client_id` 保留） |
| `app/config.py` | 加 `resolve_openai_api_key()` + `resolve_proactive_pacing()` |
| `app/rag.py` | 历史/跨会话检索加时间衰减（`_recency_factor` / `_top_k_weighted`） |
| `app/character.py` | `_prepare()` 改为 controller 感知（chat + chat_stream 共用）；加切段解析、引用、`continue_segment`、`proactive`/`proactive_farewell`/`_run_proactive`、SEGMENT_PROTOCOL_SPEC；ChatResult 加 plan/controller_trace/pending_segments/is_continuation |
| `app/web.py` | 注入 controller 到引擎；加 `/api/continue`、`/api/proactive`；`/api/chat` 加 quoted_text + 响应字段；`/api/status` 下发 pacing。**全部接在新版多客户端（X-Client-Id）架构上** |
| `app/templates/chat.html` | 主动发言开关+节奏配置 UI、两套计时器（续说/退避主动）、引用按钮+引用条、pacing 下发。**语音 mic/replay/SSE 未动** |
| `requirements.txt` | 已含 openai/json-repair（controller）+ numpy/soundfile（语音）注释 |

---

## 四、运行 / 环境

### Key（`/root/lina/.env`，已 gitignore）
```
ANTHROPIC_API_KEY=...   # 主模型 Claude
OPENAI_API_KEY=...      # controller 的 gpt-5-mini
```
> 注意新版是**多客户端**：服务全局不预连 key，每个浏览器在右上角「连接」时用自己的 key；填 `0` 走服务端默认（.env）key。

### 启动（后台、脱离 session）
```bash
cd /root/lina
export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY' .env | cut -d= -f2- | tr -d ' "')
export ANTHROPIC_API_KEY=$(grep '^ANTHROPIC_API_KEY' .env | cut -d= -f2- | tr -d ' "')
setsid bash -c "OPENAI_API_KEY='$OPENAI_API_KEY' ANTHROPIC_API_KEY='$ANTHROPIC_API_KEY' \
  nohup python /root/lina/run_web.py --host 0.0.0.0 --port 8000 < /dev/null > /tmp/lina_web.log 2>&1 &"
```
访问：Cursor/SSH 转发 8000 → 浏览器 `http://localhost:8000`。停止：`pkill -f run_web.py`。日志：`/tmp/lina_web.log`。

### 语音依赖（已装）
- GPU：2× NVIDIA L20X（各 143GB），驱动 CUDA 12.8
- `torch 2.10.0+cu128` / `torchaudio`（已有）
- `qwen-tts==0.1.1`（带 transformers 4.57.3）
- `qwen-asr==0.0.6`（`pip install --no-deps`，复用 4.57.3，绕开与 tts 的 transformers 版本冲突）
- `nagisa==0.2.11`（asr 隐藏依赖）、系统 `sox`
- 模型权重已缓存在 `~/.cache/huggingface/hub/`（首次加载 ~90s，后续快）

### 可调环境变量（节奏 / 开关）
```
LINA_CONTROLLER=off            # 关掉 controller 的 LLM 微顾问（规则层仍跑）
LINA_CONTROLLER_TIMEOUT=600    # 调大 controller 超时（debug 打断点时用）
LINA_ENGAGE_BASE_MS=30000      # 主动发言首次基础间隔
LINA_ENGAGE_MULTIPLIER=1.8     # 退避倍数
LINA_ENGAGE_JITTER=0.3         # 主动发言抖动
LINA_MAX_NUDGES=3              # 主动几次后告别
LINA_CONTINUE_BASE_MS=3000     # 续说间隔
LINA_ASR_DEVICE=cuda:0 / LINA_TTS_DEVICE=cuda:1   # 语音模型 GPU 分配
```

---

## 五、怎么测主动发言「勾人」

普通闲聊感受不到是**正常的**（钩子要稀疏）。要这样造条件：
1. 先聊几句**和你自己有关、且留尾巴**的事（如「我最近搬家累死了」「我那只猫一直叫」）。
2. 然后**停手别回话**。
3. 把顶栏「首次 N 分钟」调成 **1**（默认 3 太久）。
4. 等约 1 分钟 → 她应主动捡你的话头（「你那只一直叫的猫适应新家了没？」），而不是泛问「在吗」。
- 想快速看退避+告别：设「首次 1 分钟·共 2 次后告别」。
- 想看续说：问「讲讲你最近最奇怪的一件事，慢慢说」然后别回话，她会一段段补。

---

## 六、相关论文（依据，详见 proactive_paper_list.md）
- LD-Agent (#8) → 时间衰减检索
- MapDia (#9) → 从历史挑话头做主动发言的目标
- PaRT (#4) → 以用户画像为锚生成话题
- XiaoIce (#3) / Replika (#15) → 「被记得」比「被找」更影响留存
- Critic Guidance (#14) → 防止主动变骚扰（对应未做的 C 级「用户热情度调节」，待办）

## 七、已知 / 待办
- **C 级未做**：用户热情度判断（敷衍→收着点、提前告别），等 A+B 验证后再说。
- **controller 偶发 `APIConnectionError`**：本机到 OpenAI 网络偶尔抖动，已有 fail-safe（退回规则层/fallback），不影响对话。如频繁可给 controller 调用加重试。
- ~~流式切段的显示问题~~ **已修**：`chat_stream` 现在遇到 `[` 就扣住不推（可能是 segments 标记开头），流末判断——是 `[segments:]` 就丢、是普通方括号就照常显示。标记不再进字幕、不再被 TTS 念。普通方括号文字（如「他说[这个词]」）完整保留。
