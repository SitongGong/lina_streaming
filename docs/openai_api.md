# Lina OpenAI 兼容接口

Lina 提供一个与 OpenAI **完全兼容**的 `/v1/chat/completions` 接口，
任何支持 OpenAI 的客户端（官方 SDK、LangChain、各类工具）都能直接调用，
支持流式（`stream=true`）和非流式。

> 回复只返回**纯文本**对话内容（Lina 内部的情绪标记 `[mood:…]`、分段标记
> `[segments:…]` 已在服务端剥除，调用方拿到的就是干净的回复）。

---

## Endpoint

```
POST  http://<HOST>:<PORT>/v1/chat/completions
Content-Type: application/json
Authorization: Bearer <API_KEY>     # 仅当服务端配置了 LINA_API_KEYS 时需要
```

`model` 字段填任意值即可（如 `"lina"`），服务端固定用 Lina 的角色引擎，不按 model 路由。

---

## 鉴权

由服务端环境变量 `LINA_API_KEYS` 控制：

- **未配置** → 不鉴权，任何请求都放行（适合内网 / 调试）。
- **配置了**（逗号或空格分隔多个 key）→ 调用方必须在
  `Authorization: Bearer <key>` 里带其中一个，否则返回 `401`。

```bash
# 服务端启用鉴权（启动前设置）
export LINA_API_KEYS="sk-lina-abc123,sk-lina-def456"
```

---

## 用法

### 1. OpenAI 官方 Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<HOST>:<PORT>/v1",
    api_key="sk-lina-abc123",   # 没启用鉴权时填任意非空字符串即可
)

# 非流式
resp = client.chat.completions.create(
    model="lina",
    messages=[{"role": "user", "content": "你好，你是谁？"}],
)
print(resp.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="lina",
    messages=[{"role": "user", "content": "给我讲讲你最近鉴定的遗物"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### 2. curl（非流式）

```bash
curl http://<HOST>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-lina-abc123" \
  -d '{
    "model": "lina",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 3. curl（流式）

```bash
curl -N http://<HOST>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-lina-abc123" \
  -d '{"model":"lina","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

---

## 多轮对话

接口是**无状态**的（每次请求独立、不在服务端存会话）。要多轮，
按标准 OpenAI 用法把历史放进 `messages`：

```python
messages = [
    {"role": "user", "content": "我养了只猫叫团子"},
    {"role": "assistant", "content": "诶，猫啊。它平时黏不黏你？"},
    {"role": "user", "content": "挺黏的，老踩我书"},   # 本轮输入
]
```
服务端取**最后一条 user 消息**作为本轮输入，其余 user/assistant 作为上下文历史。
`system` 消息会被忽略（Lina 的人设由服务端固定，不接受外部覆盖）。

---

## 响应格式

### 非流式 — `chat.completion`
```json
{
  "id": "chatcmpl-xxxx",
  "object": "chat.completion",
  "created": 1718000000,
  "model": "lina",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "嗯……我叫西比莉娜，炼金术士学徒。"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 3561, "completion_tokens": 125, "total_tokens": 3686}
}
```

### 流式 — `chat.completion.chunk`（SSE）
```
data: {"id":"chatcmpl-xxxx","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxxx",...,"choices":[{"index":0,"delta":{"content":"昨天送来"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxxx",...,"choices":[{"index":0,"delta":{"content":"一个铁盒。"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxxx",...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 注意 / 限制

- **无状态**：不存会话，也不接入登录用户的「自我清单 / 跨会话记忆」（那些依赖 Web 登录身份）。需要记忆请自己维护 `messages`。
- **回复仍经过 controller**：错字容错、抑制连环提问、few-shot 等行为微调都生效——不是裸模型。
- **服务端需配置默认 key**：接口用服务端默认的 Anthropic key（`ANTHROPIC_API_KEY`，经 `.env` 或环境变量）。未配置则返回 `503`。
- **上线公网务必设 `LINA_API_KEYS`**：否则任何人都能调用、消耗你的额度。
