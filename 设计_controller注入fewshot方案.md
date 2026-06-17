# 设计方案：用 controller 按场景注入不同 few-shot cases

> 状态：**设计稿，待你定/试**，本文档不含已落地代码。
> 约束：**只动 controller 侧**（`prompts/controller/` + `app/controller/`），不碰主模型 prompt。
> 目标：让 controller 每轮判断"本轮属于哪种场景"，只把**对应的 1–3 个 few-shot 示例**
> 注入到 composer 的动态尾块，引导主模型的具体说话方式，而不在主模型 prompt 里写死示例。

---

## 0. 为什么要这个机制

现在调主模型行为有两档手段：
1. **约束句**（已落地）：如 `suppress_trailing_question` / `lenient_typos`，在本轮约束块加一句"别这样/要这样"。够用于**单一、可一句话说清**的规则。
2. **few-shot 示例**（本方案）：当"光说规则说不清、得给样例"时——尤其语气、节奏、错别字处理这类**靠模仿比靠描述更有效**的能力——给几个"坏例→好例"对比，主模型照着学。

few-shot 比纯约束句强在：**示范** > **描述**。但代价是 token，所以必须**按场景精准注入**，不能整套常驻。

---

## 1. 核心设计：场景标签驱动的示例库

### 1.1 建一个 few-shot 库（纯数据文件，controller 侧）
```
prompts/controller/fewshot/
  no_trailing_question.txt    # 不强行提问
  typo_tolerance.txt          # 错别字善意理解
  short_reaction.txt          # 短接话别啰嗦
  comfort.txt                 # 安抚的口吻
  excitement.txt              # 兴奋点的连珠炮（正向鼓励）
  boundary.txt                # 现代请求的茫然以对
  ...（一个场景一个文件，后续随时加）
```
每个文件放 **2–4 个 `坏例 → 好例` 对**（对比式比纯正例更有效）。格式见 §4。

### 1.2 Plan 加一个字段（`schema.py`）
```python
fewshot_tags: tuple[str, ...] = ()   # 本轮要注入哪些示例文件（按 tag）
```
- 在 `__post_init__` 里用 `_unique_keep_order` 去重、并**对照一个白名单 `_VALID_FEWSHOT`** 过滤非法 tag（和现在 `_VALID_MODULES` 一样的写法）。
- 设一个**上限**（如最多 2 个 tag），避免一次注入太多示例撑爆 token。

### 1.3 谁来决定挂哪些 tag

和现有 module 机制完全对称，两条路：

**A. 规则层（`rule_router.py`）— 明确场景直接挂**
直接复用现在每个分支的 `matched_rule` 映射到 tag。例如：
| matched_rule | 注入的 fewshot tag |
|---|---|
| `user_vent` | `comfort`, `no_trailing_question` |
| `short_reaction` | `short_reaction` |
| `modern_action_request` | `boundary` |
| `world_immersion` | `excitement` |
| `self_introspection` / `relationship_recall` | `no_trailing_question` |
（可直接在我已加的 `_apply_behavior_defaults` 那个钩子里顺手塞 `fewshot_tags`，零新增散落点。）

**B. 微顾问（`experts.py`）— 模糊场景用 LLM 判**
加一个 `TextAdvisor`（或多个 `BoolAdvisor`）产出 `fewshot_tags`。但 `TextAdvisor` 现在返回单串，标签是列表——两个做法：
- **做法①（推荐，省事）**：用**多个 `BoolAdvisor`**，每个判一个 tag 要不要挂（`fewshot_comfort`/`fewshot_excitement`…），merge 时收集为真的那些拼成 `fewshot_tags`。复用现成 BoolAdvisor，**零新类**。
- **做法②**：新增一个返回 JSON 数组的 `TagsAdvisor`（要改 `_normalize_fields` 解析数组 + 白名单过滤）。更省并发槽，但要写新类。

**lenient_typos 已经有 BoolAdvisor 了**——few-shot 的 `typo_tolerance` 可以直接**搭它的便车**：当 `lenient_typos=True` 时自动挂 `typo_tolerance` tag，不必再判一次。同理 `suppress_trailing_question=True` → 自动挂 `no_trailing_question`。**这样 few-shot 几乎不增加新的判断开销。**

### 1.4 composer 注入（`composer.py`）
在 `compose()` 里、构造约束块之后，按 `plan.fewshot_tags` 逐个 `load_prompt("controller/fewshot/<tag>.txt")`，拼成一个"参考示例"块加进 `blocks`：
```
【本轮参考示例 — 只学其说话方式，不要照抄内容】
（这里是 1-3 组 坏例→好例）
```
关键点：
- 放进**动态尾块**（block 2），**不进 cached 块** → 不同 few-shot 不破坏 prompt 缓存。
- 加 `trace` 字段 `fewshot_tags` / `fewshot_chars`，方便 debug 面板观察。

---

## 2. 端到端数据流（落地后）

```
用户输入
  → rule_router：命中规则 → 顺手塞 fewshot_tags（明确场景）
       未命中 → fan-out 微顾问 → suppress/lenient/各 fewshot_* Bool → merge 收集 tags
  → Plan.fewshot_tags（去重、白名单过滤、≤2）
  → composer：读对应 fewshot/*.txt，拼成「参考示例」块 → 动态尾块
  → 主模型：看着坏例→好例，照着说话方式回复
```

---

## 3. 改动清单（落地时，全在 controller 侧）

| 文件 | 改动 |
|---|---|
| `prompts/controller/fewshot/*.txt` | **新增**示例文件（一场景一文件） |
| `app/controller/schema.py` | 加 `fewshot_tags` 字段 + `_VALID_FEWSHOT` 白名单 + post_init 过滤/去重/上限 |
| `app/controller/rule_router.py` | `_apply_behavior_defaults` 里按 `matched_rule` 塞 tag；并由 suppress/lenient 自动带出对应 tag |
| `app/controller/experts.py` | （做法①）加几个 `fewshot_*` BoolAdvisor；或（做法②）一个 TagsAdvisor |
| `app/controller/controller.py` | `_merge` 收集为真的 `fewshot_*` → `fewshot_tags`；`_fallback_plan` 给空 |
| `app/controller/composer.py` | 按 tags 读文件、拼「参考示例」块、加 trace |

**主模型 prompt（`character.py` / `prompts/modules/`）：一字不动。**

---

## 4. few-shot 文件写法（决定效果，建议规范）

每组用"坏例→好例"对比，标注清楚谁是用户、哪个错哪个对。示例 `typo_tolerance.txt`：
```
# 参考示例：用户打错字时，善意理解、别揪着

[用户] 我今天去图书�guan借书
[✗ 别这样] 图书…什么？你说的 guan 是什么？我没听过这个词。
[✓ 要这样] 哦你去图书馆啦——借到想看的没？

[用户] 我最近在学钢请
[✗ 别这样] 钢请是什么？是某种古代乐器吗？
[✓ 要这样] 学钢琴啊，挺好——弹到哪一步了？
```
示例 `no_trailing_question.txt`：
```
# 参考示例：别每条都甩问号，陈述/附和/分享也能接住话

[用户] 我昨天看了场话剧，挺震撼的
[✗ 别这样] 是吗？哪部话剧？讲什么的？你跟谁去的？  ← 连珠炮、像审问
[✓ 要这样] 唔，能让你说"震撼"的可不多。  ← 先接住，不急着问
```

**规范建议：**
- 每文件 2–4 组；每组尽量短（坏例+好例各 1–2 行）。
- 一定带**坏例**——只给好例，模型学不到"边界在哪"。
- 内容贴莉娜人设（炼金/遗物/戏剧/香草背景），别用跳戏的现代例子。

---

## 5. 风险 / 取舍（供你决策）

1. **token**：每注入一个 tag ≈ 几百字。靠"≤2 个 tag"上限 + 只在动态尾块控制；不破缓存。
2. **过拟合示例**：模型可能照抄示例里的具体话（"图书馆""话剧"）。缓解：示例里明确写"只学说话方式、不要照抄内容"，且示例题材分散。
3. **判断开销**：若每个 tag 都加一个 BoolAdvisor，并发顾问数会涨。缓解：优先**搭车现有 flag**（suppress/lenient 直接带出 tag），只给少数真需要 LLM 判的场景加顾问。
4. **示例维护**：场景多了文件会多。好处是**加场景=加文件+加 tag，零侵入**，不动主逻辑。

---

## 6. 建议的最小起步（如果你要试）

先不做"微顾问判 tag"，只做**最省的一条链路**验证机制通不通：
1. schema 加 `fewshot_tags` + 白名单（5 行）。
2. composer 加注入逻辑（10 行）。
3. rule_router 的 `_apply_behavior_defaults` 里：`suppress_trailing_question` 为真就带 `no_trailing_question`，`lenient_typos` 为真就带 `typo_tolerance`（复用已有判断，**不加任何新顾问**）。
4. 写 `no_trailing_question.txt` + `typo_tolerance.txt` 两个文件。

跑通、看效果满意后，再按 §1.3 扩展到"微顾问判更多场景 tag"。
