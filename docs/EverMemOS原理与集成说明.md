# EverMemOS 原理 / 我们如何使用 / 日记如何接入 / embedding 存哪

> 给 debug 和后续维护用。基于 lina_evermemos 当前实现，路径/数字/模型均已核实。

---

## 零、用到的模型 + 一轮对话的整体流程

### 0.1 模型清单（各环节用谁）
| 环节 | 模型 | 在哪配 | 作用 |
|---|---|---|---|
| **主回复模型** | `claude-sonnet-4-6` | `character.py:DEFAULT_MODEL` | 真正生成莉娜的话 |
| **controller 判定** | `claude-haiku-4-5`（默认走 Claude） | `config.py` | 每轮的微判断（tone/切句/need_diary/告别…），快而便宜 |
| **本地 EverOS embedding** | `Qwen3-Embedding-4B`（vLLM 8008） | `EverOS/.env` | 给人设/日记/话题的 md 建向量索引 + 检索 |
| 本地 EverOS LLM | `Qwen/Qwen3-8B`（vLLM 7778） | `EverOS/.env` | 只 user 轨道抽取用（我们没用这条） |
| 本地 EverOS rerank | `Qwen3-Reranker-4B`（8009） | `EverOS/.env` | 端点不兼容，**我们没用，检索走 vector** |
| 用户对话记忆 | EverMem **云** `api.evermind.ai` | `EVERMEM_API_KEY` | 云端自己跑提取/检索 |

### 0.2 一轮对话的整体流程
```
用户发一句话
 1. controller（claude-haiku）读 本句 + 近几轮历史 → 产出 plan：
      tone / 切句 / 检索开关 / need_diary（要不要调日记）/ user_farewell（用户告别没）…
 2. 按 plan 取三类记忆（都注入到 character.py 的注入块，不碰主模型 prompt 模板）：
      ① 用户记忆：MemoryClient 打 EverMem 云 search（按 user_id=cid）→「关于用户的记忆」块
      ② 人设：PersonaMemory 打本地 EverOS（agent_id=lina, vector）→「角色设定参考」块
      ③ 日记：plan.need_diary 为真 → DiaryMemory 两级检索（见 第三节）→「莉娜的日记」块
 3. 组装 system（CORE 人设，带 cache）+ 注入块 + 历史 → 调 主模型 claude-sonnet-4-6 生成回复
 4. 回复后台：把这轮对话写回 EverMem 云（add）；滑出窗口的轮次触发 flush/概括
```
关键：**controller 只做轻判断、不枚举具体话题**；**主模型 prompt 模板不动**，所有记忆
都从 controller-side 的注入块进。

---

## 一、EverMemOS（本地 EverOS）原理

EverOS 是一个**独立的 HTTP 记忆服务**（不是库），数据本地存在 `~/.everos`（我们改到
了 `/root/lina_everos_data`）。核心是「**Markdown 为真相源 + 两套索引**」的三件套：

```
Markdown (真相源)  +  SQLite (状态/队列/审计)  +  LanceDB (向量 + BM25 索引)
```

- **Markdown**：每条记忆是一个 `.md` 文件，是唯一权威数据。删了索引能从 md 重建。
- **SQLite**：记录每个 md 的处理状态、cascade 队列、审计。
- **LanceDB**：把 md 内容 embedding 成向量 + 建 BM25，供检索。**可丢弃重建**。

### 1.1 两条记忆轨道
- **user 轨道**（`user_id`）：用户的事 → episode / atomic_fact / profile / foresight。
  这条轨道的写入要走 `/add` → 边界检测 → **LLM 抽取**（需要大模型）。
- **agent 轨道**（`agent_id`）：agent（莉娜）自己的事 → agent_case / **agent_skill**。

### 1.2 关键机制：cascade（我们用的就是它）
cascade 是个常驻守护：**监听 md 文件变化 → 用 embedding 把它写进 LanceDB**。
- **cascade 全程不调 LLM**（已查源码 + 验证）——它只做 embedding 索引。
- 所以：**只要把 md 写到对的目录、带对的 frontmatter，cascade 自动建索引，不需要
  大模型**。我们的人设/日记/话题全走这条路，绕开了「user 轨道 LLM 抽取」的依赖。

### 1.3 依赖的模型
- **Embedding**：本机 vLLM `Qwen3-Embedding-4B`（`http://127.0.0.1:8008/v1`）——
  cascade 建索引、检索都用它。**模型名必须是 `Qwen3-Embedding-4B`（不带 Qwen/ 前缀）**。
- **LLM**：只有 user 轨道的 `/add` 抽取才需要（我们目前没用这条），配在 7778。
- **Rerank**：8009，但本机端点格式与 EverOS 期望不兼容（404），**我们检索一律用
  `method=vector` 绕开 rerank**。

### 1.4 检索方式
`POST /api/v1/memory/search`，三种 method：keyword(BM25) / vector(语义) / hybrid。
我们用 **vector**（语义准、不触发 rerank 404）。

---

## 二、我们如何使用 EverOS

服务启动（端口 8090，独立数据目录）：
```bash
cd /root/EverOS && everos server start --host 127.0.0.1 --port 8090 \
    --env-file /root/EverOS/.env
```
关键配置在 `/root/EverOS/.env`：LLM→7778，Embedding→8008(`Qwen3-Embedding-4B`)，
Rerank→8009，`EVEROS_MEMORY__ROOT=/root/lina_everos_data`。

lina_evermemos 这边有 **三套记忆**，各管一摊、互不串：

| 记忆 | 服务 | 客户端 | 隔离键 | 开关 |
|---|---|---|---|---|
| 用户对话记忆 | EverMem **云** (api.evermind.ai) | `app/memory_client.py` | `user_id`=cid | `EVERMEM_API_KEY` |
| 莉娜人设 | 本地 EverOS 8090 | `app/persona_memory.py` | `agent_id=lina` | `EVEROS_PERSONA=1` |
| 莉娜日记/谈资 | 本地 EverOS 8090 | `app/diary_memory.py` | `agent_id=lina_diary` / `lina_topic` | `EVEROS_DIARY=1` |

启动 lina_evermemos（带全套记忆）：
```bash
cd /root/lina_evermemos
EVEROS_DIARY=1 EVEROS_PERSONA=1 EVEROS_LOCAL_URL=http://127.0.0.1:8090 \
EVERMEM_API_KEY=<云key> python run_web.py --host 0.0.0.0 --port 8080
# 这些 env 也已写进 .env，run_web 的 _load_dotenv 会自动读
```

注入点都在 `app/character.py` 的 `_build_user_content`（controller-side 注入块，
**非主模型 prompt**），与「角色设定参考」「关于用户的记忆」并列：
- 人设 → 「角色设定参考」块（PersonaMemory 取代原 BM25）
- 日记 → 「莉娜的日记」块（DiaryMemory，由 `plan.need_diary` 门控）
- 用户记忆 → 「关于用户的记忆」块（MemoryClient）

---

## 三、日记如何加入框架

### 3.1 数据
来自上游 PR#12，在 `diary/`：179 篇日记（=180 条记忆）+ 谈资索引（话题树）。
每条记忆结构化（时间/地点/人物/正文/情绪…），并有 `应关联话题卡` 字段列话题ID
（`A01-01|大类|角度`）。话题↔日记是多对多。

### 3.2 导入（写 md → cascade 自动索引）
两个脚本，把数据写成 agent_skill md 文件：

- **日记**：`scripts/import_diary_to_everos.py`
  - 每条记忆 → 一个 agent_skill，`agent_id=lina_diary`
  - 写到 `<root>/default/default/agents/lina_diary/skills/skill_diary_<记忆ID>/SKILL.md`
  - content = 标题+事实骨架+正文+情绪+地点…，末尾「话题: A01-01 …」做可检索标签
- **话题卡**：`scripts/import_topics_to_everos.py`
  - 被日记引用过的话题三段组合 → agent_skill，`agent_id=lina_topic`，1620 个

agent_skill 的 frontmatter 必填：`type/agent_id/name/description/confidence/maturity_score`
（缺 agent_id 或 name，cascade 拒绝索引）。

### 3.3 检索链路（两级，统一走话题层）—— 你问的「话题→日记怎么操作」

代码在 `app/diary_memory.py`，由 controller 的 `plan.need_diary` 门控（判定本轮要
聊莉娜的生活才检索，否则跳过省开销）。**为什么要先过话题层**：上百个话题不可能塞
给 controller 枚举判断，所以「话题识别」本身也做成 EverOS 检索——话题卡是可检索实体。

详细三步：

**第①级 `find_topics(context)`** —— 把当前对话语义匹配到莉娜的话题：
- 拿「近几轮 + 本轮用户的话」当 query，打 `agent_id=lina_topic` 的 search（vector）。
- 命中若干话题卡，每个话题卡 description 形如 `A01-01 天气与季节 · 两边今天的天气`。
- 解析出 `(话题ID, 中文角度描述)`，例如 `("C03-02", "修道院/学城生活 学徒时代")`。
- **关键坑（已修）**：话题ID 代号 `C03-02` 对 embedding 是无意义噪声，**不能**拿它去
  拼第②级的 query（曾导致召回跑偏：问修道院却召回培根探病）。所以这里要把**中文
  角度描述**留下来给第②级用，ID 只用于追溯。

**第②级 `find_diaries(context, topics)`** —— 用话题把日记捞出来：
- query = 「第①级得到的话题**中文描述** + 用户上下文」拼起来（不含 ID 代号）。
- 打 `agent_id=lina_diary` 的 search（vector），召回 top-k 篇日记。
- 日记 md 当初导入时，content 末尾写了 `话题: A01-01 C03-02 …` 标签，所以「话题描述」
  能在语义上把对应话题的日记拉上来；同时用户原话也参与语义匹配，双重保证对题。

**第③步 `retrieve_text` 拼注入文本**：
- 把召回日记的正文（去掉末尾「话题:」标签行）拼成「莉娜的日记」块。
- 多个话题、每个话题多篇日记时，search 已按相关度排序取 top，**不会全塞**——这正是
  「两个话题对应多篇日记，不可能全放进 prompt」靠 EverOS 检索压缩解决的点。

一句话：**当前对话 →（语义）匹配到话题 →（用话题中文描述+原话语义）捞出对题日记 → 注入**。
两级都是 vector 语义检索，controller 全程不碰具体话题，话题再多也不怕。

### 3.4 重新导入 / 重建
- 改了日记数据 → 重跑导入脚本（覆盖 md）→ cascade 自动重新索引。
- 索引坏了 → `rm -rf /root/lina_everos_data/.index/lancedb` → 重启 EverOS，
  scanner 会从所有 md 重建（md 是真相源）。

---

## 四、日记的 embedding 存在哪里？（你问的）

**存在 LanceDB 的 `agent_skill.lance` 这张表里**：

```
/root/lina_everos_data/.index/lancedb/agent_skill.lance     ← 向量在这（约 11MB）
```

注意：**人设(lina) + 日记(lina_diary) + 话题(lina_topic) 三者都是 agent_skill 类型，
所以它们的 embedding 全在同一张 `agent_skill.lance` 表里**，靠表内的 `agent_id`
字段区分（检索时按 agent_id 过滤）。

完整存储位置对照：

| 东西 | 路径 |
|---|---|
| **向量（embedding）** | `/root/lina_everos_data/.index/lancedb/agent_skill.lance` |
| 其它类型向量 | 同目录下 `episode.lance / atomic_fact.lance / user_profile.lance / foresight.lance / agent_case.lance`（我们没用到） |
| **Markdown 真相源（日记）** | `/root/lina_everos_data/default/default/agents/lina_diary/skills/skill_diary_*/SKILL.md`（180 篇） |
| Markdown（话题） | `…/agents/lina_topic/skills/…`（1620 个） |
| Markdown（人设） | `…/agents/lina/skills/…`（49 个） |
| 状态/队列/审计 | `/root/lina_everos_data/.index/sqlite/system.db` 等 |

**embedding 本身不是永久权威**——它从 md 算出来、可随时从 md 重建。要改/删某条日记，
改对应的 `SKILL.md`，cascade 会重新 embedding 覆盖 `agent_skill.lance` 里那条。

---

## 五、debug 时的常用观测

- 看检索过程：服务日志里 `[diary]` 行打了 `need_diary / 命中话题 / 命中日记`。
  ```bash
  tail -f /tmp/lina_evermemos.log | grep -E "\[diary\]|\[plan\]"
  ```
- 直接测某条 query 召回（不走主模型）：
  ```python
  import httpx
  httpx.post("http://127.0.0.1:8090/api/v1/memory/search",
      json={"agent_id":"lina_diary","query":"你最近过得怎么样","method":"vector","top_k":3})
  ```
- 看 EverOS cascade 是否索引成功：`/root/lina_everos_data/server.log` 里
  `cascade_worker_processed ... kind=agent_skill ... upserted=1`。
