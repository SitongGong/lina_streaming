# Controller LLM 是否需要训练 —— 评测报告

> 目的：用数据回答"controller 的场景判定该继续用 prompt 工程，还是值得训练一个专用模型"。
> 方法：构建压力测试集 → 屏蔽规则层强制走 LLM → 测准确率/稳定性 → 统计成本。
> 所有脚本可复现，见文末。

---

## 一、结论（分两个维度）

| 维度 | 结论 |
|---|---|
| **准确率角度** | **不迫切需要训练**。gpt-5-mini 开箱即用，场景区分准确率 **97.9%**，核心判定稳定。训练能提升的空间 <3%，且集中在"人类都难标"的语义边界。 |
| **成本角度** | **训练有明确价值**。现状每轮并发 **19 个** LLM 请求、**≈6589 input token**、端到端 **~2.85s**。训练能内化判定标准 → prompt 大幅缩短、可减少 advisor 数、甚至砍掉规则层。这是训练的真正卖点。 |

**一句话**：现在的 controller "够用且准"，但"贵且慢"。是否训练，取决于把它当成"提准确率"（不划算）还是"降本提速"（划算）的项目。**在训练之前，有一个不用训练就能显著降本的中间方案（见第五节）。**

---

## 二、测试方法

1. **压力测试集**：118 条人工标注 case（`tests/controller/cases.jsonl`），覆盖 9 类场景：
   标准场景、歧义共现、表层陷阱、错别字、隐式延续、正向回应、需上下文、冒犯/套隐私、边界模糊。
2. **强制走 LLM**（`--force-llm`）：屏蔽规则层，让每条都走 gpt-5-mini fan-out —— 纯测 LLM 的判别力，排除规则层"抢答"的干扰。
3. **一致性**：挑 16 条最难的 case，每条跑 5 次，区分"场景判定"和"风格 flag"两个口径分别统计。
4. **成本**：统计每个 advisor 的 prompt 长度、每轮总 token、实测端到端耗时。

---

## 三、准确率与稳定性数据

### 3.1 场景区分准确率（force-llm，规则层屏蔽）
- **明确场景命中：46/47 = 97.9%**
- 错别字识别（lenient_typos）：7/7 = 100%
- 自我事实检索判断（use_self_facts）：3/3 = 100%
- 唯一失分："你害怕什么东西吗"（单条边界，非系统性）

LLM 正确处理了规则层之前判错的难题：
- 陷阱"帮我查这个**古代符文**" → 正确判 world（没被"帮我查"骗去现代请求）
- 歧义"你好啊我最近好烦" → 正确判 user_vent
- 隐式"团子好一些了" → 正确判 relationship_recall + 温柔语气（这正是之前体感"死板"的 case，LLM 判对了）

### 3.2 一致性（16 条难 case ×5 次）
- **场景区分稳定：13/16 = 81%**（把输入归到哪类，多次一致）
- 含风格 flag 全一致：7/16 = 44%
- **摇摆的 3 条都是"程度差异"而非"判到无关场景"**：
  | case | 摇摆情况 | 性质 |
  |---|---|---|
  | "好一些了" | 3次recall / 1次vent / 1次两者 | 报喜 vs 关切，边界 |
  | "我觉得自己什么都做不好" | 4次vent / 1次+self | 发泄 vs 自省，确实兼具 |
  | "你平时一个人不无聊吗" | 4次world / 1次无 | 轻微抖动 |
- 其余"风格抖"（如 suppress 4/5↔5/5）是温度采样正常波动，**不影响场景归类**，调 flag 默认值即可，不需训练。

**这 3 条就是"训练候选"** —— 它们是 prompt 也难说清的语义边界。但仅 3/16，且都未判到无关场景，不致命。

---

## 四、成本数据（现状）

| 指标 | 数值 |
|---|---|
| 每轮并发 LLM 请求数 | **19 个**（每个 advisor 一个独立请求）|
| 每轮 input token | **≈ 6589**（19 个 prompt 相加）|
| 单个 advisor prompt | ~520 字符 / ~347 token |
| 走 LLM 端到端耗时 | **平均 2.85s/轮**（1.7–4.5s，受最慢 advisor + 网络重试）|
| 规则层命中时 | ~0s，0 token |

**token 花在哪（=训练能省的）**：每个 advisor prompt = 模板(~200 tok) + 判定规则说明(target_desc+decision_rules, ~100-145 tok) + 用户输入+历史。**19 个 advisor 各自重复带一遍输入历史、各驮一段判定规则**——这是最大浪费，也正是训练能消除的（标准内化进权重，prompt 只需给输入）。

---

## 五、不用训练就能降本的中间方案（建议先做）

19 个**单字段**请求 → 合并成几个**多字段**请求。同一类判断本可一次问完：

| 合并组 | 包含的 advisor | 一次问出 |
|---|---|---|
| 场景模块组 | module_user_vent / action_boundary / world_immersion / relationship_recall / self_introspection | 5 个 bool |
| 风格组 | suppress_trailing_question / lenient_typos / allow_doubt_wrap / enforce_mood_continuity | 4 个 bool |
| 长度组 | sentences / max_reply_chars / allow_segment / tone_hint | 4 个字段 |
| 检索组 | use_self_facts / use_world / use_sample_conversations / query_hint / history_window / hook_* | 检索相关 |

**预期收益**：请求数 19 → ~4，输入历史只带 4 遍而非 19 遍 → token 和并发压力都大幅下降，延迟也降（少了 fan-out 的尾延迟）。**风险**：一次问多字段，单字段准确率可能略降，需用本测试集回归验证。

---

## 六、对两个优化方向的回应

1. **压缩规则、多数交给 LLM**：✅ 数据支持。LLM 准 97.9%、核心判定稳定，完全有能力接管。规则层应收缩到只留"零歧义系统态"（空输入/续说/主动/纯标点），其余交 LLM——避免规则层抢答误判（之前体感"死板"的根因之一）。
2. **训练 → prompt 变短、砍掉规则层**：✅ 这是训练的真正价值（降本，非提准）。但建议**先做第五节的合并方案**拿到低成本收益，再评估训练 ROI。

---

## 七、可复现

```bash
export OPENAI_API_KEY=...
# 准确率（强制走 LLM）
python tests/controller/run_eval.py --force-llm
# 一致性（16 难 case ×5）
python tests/controller/run_consistency.py --runs 5
# 成本（prompt 体量 + 耗时）
python tests/controller/measure_cost.py --timing
```
测试集：`tests/controller/cases.jsonl`（118 条，可继续扩充）。
