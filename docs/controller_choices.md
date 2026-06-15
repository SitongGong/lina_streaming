# Controller 在不同情形下的选择（复合/歧义/陷阱压力测试）

> 主模型/controller: claude-haiku-4-5-20251001。**来源**=规则层秒判 还是 LLM 微顾问判。

| # | 考察点 | 输入 | 来源 | 场景(rule) | 场景模块 | 查自我清单 | 查跨会话 | 句/字上限 | 抑制提问 | 容错 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 问候+回访 | 你好，还记得我和你说过的事吗 | 规则层 | relationship_recall | relationship_recall | ✓ | ✓ | 3/70 | ✓ | ✓ |
| 2 | 问候+发泄 | 嗨，我最近压力好大 | LLM | - | user_vent |  |  | 2/50 | ✓ | ✓ |
| 3 | 问候+问她自己 | 你好呀，你是谁来着 | 规则层 | self_introspection | self_introspection | ✓ |  | 3/80 | ✓ | ✓ |
| 4 | 问候+兴奋点 | 早上好，你最近鉴定了什么遗物 | 规则层 | world_immersion | world_immersion | ✓ |  | 3/80 | ✓ | ✓ |
| 5 | 问候+现代请求 | 你好，帮我查个天气 | 规则层 | modern_action_request | action_boundary |  |  | 2/55 | ✓ | ✓ |
| 6 | 告别+回访 | 我先走了，对了你之前说的戏剧叫啥 | 规则层 | world_immersion | world_immersion | ✓ |  | 3/80 | ✓ | ✓ |
| 7 | 告别+正向 | 拜拜，今天聊得真开心 | LLM | - | 无 |  |  | 1/35 | ✓ | ✓ |
| 8 | 回访+错字 | 还记得我那只猫的过min吗 | 规则层 | relationship_recall | relationship_recall | ✓ | ✓ | 3/70 | ✓ | ✓ |
| 9 | 兴奋点+错字 | 你研究的那个古带语好神奇 | LLM | - | world_immersion,relationship_recall |  |  | 3/50 |  | ✓ |
| 10 | 像现代请求实为兴奋点 | 帮我查查这个古代符文什么意思 | 规则层 | modern_action_request | action_boundary |  |  | 2/55 | ✓ | ✓ |
| 11 | 像请求实为兴奋点 | 你能不能帮我看看这块碑文 | 规则层 | world_immersion | world_immersion | ✓ |  | 3/80 | ✓ | ✓ |
| 12 | AI自指(带夸) | 你是不是机器人啊，好聪明 | LLM | - | self_introspection,action_boundary |  |  | 2/45 | ✓ | ✓ |
| 13 | 靠上下文延续 | 是的 | LLM | - | world_immersion,relationship_recall |  |  | 1/25 | ✓ | ✓ |
| 14 | 短回应靠上下文 | 还行吧 | LLM | - | 无 |  |  | 1/30 | ✓ | ✓ |
| 15 | 极短/情绪模糊 | 嗯…… | LLM | - | user_vent |  |  | 1/28 | ✓ | ✓ |
| 16 | 否认+疲惫 | 我没事，就是有点累 | LLM | - | user_vent |  |  | 2/50 | ✓ | ✓ |
| 17 | 报喜+疲惫 | 终于搞定了，累死我了 | 规则层 | user_vent | user_vent |  |  | 3/70 | ✓ | ✓ |
| 18 | 自我否定(无显式发泄词) | 我觉得自己挺没用的 | LLM | - | user_vent |  |  | 3/70 | ✓ | ✓ |
| 19 | 问她喜好+互动 | 你喜欢喝什么？我请你 | LLM | - | world_immersion | ✓ |  | 2/50 | ✓ | ✓ |
| 20 | 问自己+问世界 | 讲讲你自己吧，还有你们那个世界 | 规则层 | self_introspection | self_introspection | ✓ |  | 3/80 | ✓ | ✓ |

## 怎么看
- **来源=规则层**：正则秒判（0 LLM，快）；**来源=LLM**：规则没命中，交 19 个微顾问判。
- 复合句（问候+X）若 X 是实质诉求，规则层按优先级把 X 排在问候前，多数能判对；判不准的会落到 LLM。
- 看『查自我清单/查跨会话』：回访/问自己类才开，闲聊不开 —— 省检索、避免无关注入。