---
name: openai-responses-api
description: OpenAI Responses API (POST /v1/responses)—migration from Chat Completions, Items vs messages, instructions/input, output_text, previous_response_id, store and ZDR, tools, structured text.format, and function tool shapes. Use when building or migrating to client.responses, agentic flows, or comparing Responses to chat.completions.
---

# OpenAI Responses API

Primary reference: [Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses).

**Chat Completions remains supported; Responses is recommended for all new projects.** The Responses API is the unified surface for agent-like apps: built-in tools, multi-turn state, multimodal text/images, and tighter integration with reasoning models.

## Mental model: messages vs items

| | Chat Completions | Responses |
|---|------------------|-----------|
| Endpoint | `POST /v1/chat/completions` | `POST /v1/responses` |
| SDK | `client.chat.completions.create` | `client.responses.create` |
| Input | `messages[]` (roles + content) | `input` — **string**, **message list**, or items |
| Output | `choices[].message` | Typed `response` with `id` and `output[]` **Items** |
| Parallel samples | `n` > 1 | No `n`; single generation |
| Assistant text | `choices[0].message.content` | **`response.output_text`** (SDK helper) |

Items are a union type (`message`, `reasoning`, `function_call`, `function_call_output`, etc.). They are separate objects in `output`, unlike a single Chat message that bundles several concerns.

## Why prefer Responses (per OpenAI)

- **Reasoning models**: Richer tool usage; (from GPT-5.4) Chat Completions with `reasoning: none` does not support tool calling.
- **Agentic loop**: One request can run multiple tool steps (`web_search`, `file_search`, `code_interpreter`, `image_generation`, computer use, MCP, custom functions).
- **Cost / cache**: Internal tests cite ~40–80% better cache utilization vs Chat Completions.
- **State**: `store: true` (default) preserves context across turns; or chain with `previous_response_id`.
- **Inputs**: Top-level `instructions` (system) + `input` (string or list) for clearer semantics than only `messages`.

## Minimal calls

**String input (simplest):**

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="Write a one-sentence bedtime story about a unicorn.",
)
print(response.output_text)
```

**Chat-compatible message list** (drop-in for basic flows): `input` can be the same style of list you used for `messages` (roles + content).

**Recommended shape** when you want clean separation of system vs user:

```python
response = client.responses.create(
    model="gpt-5",
    instructions="You are a helpful assistant.",
    input="Hello!",
)
print(response.output_text)
```

## Response shape

- `response.id` — stable id for the response.
- `response.output` — list of **Items** (e.g. `type: "reasoning"`, `type: "message"` with `content[]` including `type: "output_text"` and `text`).
- Use **`output_text`** in the SDK instead of hand-walking `output` when you only need the final user-visible string.

## Storage

- Responses are **stored by default** (server-side). Chat Completions: stored by default for **new** accounts.
- **Disable storage:** `store: false` on either API when you must not retain data on OpenAI’s side.

## Multi-turn context

**1. Manual:** Build `input` like a transcript: append **previous `response.output`** to your context, then add the next user item(s), and call `create` again.

**2. Chaining (simpler):** With `store: true`, pass **`previous_response_id`** so the next call continues from the prior response without resending the full history:

```python
res1 = client.responses.create(
    model="gpt-5",
    input="What is the capital of France?",
    store=True,
)
res2 = client.responses.create(
    model="gpt-5",
    input="And its population?",
    previous_response_id=res1.id,
    store=True,
)
```

**Conversations API:** The migration guide also points to **persistent conversations** as an alternative to fully manual item lists. Use the current Conversations docs for object shapes.

## Zero Data Retention (ZDR) and stateless + reasoning

If you **cannot** use stateful storage:

- Set `store: false`.
- Add `include: ["reasoning.encrypted_content"]` so the API returns **encrypted reasoning items** you can pass back on the next request (in-memory use; ZDR orgs may get `store=false` enforced).

This keeps workflows stateless while still using reasoning in the loop.

## Function / tool definitions (differs from Chat)

1. **Shape:** Chat Completions uses an **externally tagged** `{"type":"function","function":{...}}` style; Responses uses an **internally tagged** flatter object: `{"type":"function","name":...,"parameters":...}`.
2. **Strictness:** In Responses, tools are **strict by default**; Chat Completions tools are **non-strict** by default.

Tool calls and user-supplied tool results are **separate Item types** correlated by `call_id`. See the function-calling / tools docs for the exact request and response item shapes.

## Structured outputs

Not `response_format`. Use:

```text
text: { format: { type: "json_schema", name, strict, schema } }
```

Details and Pydantic/`text_format` live in the [structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) and the project skill `openai-structured-outputs`.

## Native built-in tools

Pass OpenAI tools by type, e.g.:

```python
client.responses.create(
    model="gpt-5.5",
    input="Who is the current president of France?",
    tools=[{"type": "web_search"}],
)
```

No need to fake `web_search` as a custom remote function for the common case.

## Migration strategy

- **Superset / incremental:** Responses is a superset in spirit; you can move high-value flows first and keep Chat Completions elsewhere until ready.
- **Assistants API:** OpenAI is moving agent patterns to Responses; the Assistants API is deprecated (see the API docs for timeline). Prefer Responses for new agent builds.

## Related docs

- Migration (this skill’s main source): https://developers.openai.com/api/docs/guides/migrate-to-responses
- Structured outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Function calling (Responses): follow links from the migration guide to the current “Function calling” / “Using tools” sections
