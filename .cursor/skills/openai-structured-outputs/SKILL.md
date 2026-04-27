---
name: openai-structured-outputs
description: OpenAI Structured Outputs (JSON Schema) for the Responses API and Chat Completions, including Pydantic/Zod helpers, refusals, edge cases, streaming events, and schema constraints. Use when implementing structured LLM responses with the OpenAI Python/JS SDKs (text.format / response_format), parsing with Pydantic or Zod, or handling refusals and incomplete responses.
---

# OpenAI Structured Outputs

Structured Outputs constrains the model to a supplied [JSON Schema](https://json-schema.org/). Benefits: reliable type-safety, programmatically detectable safety **refusals**, simpler prompting.

Reference: [Structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Decision: which API and which form

| Use case | Recommended path |
|---|---|
| Structure the assistant's reply to the user | **Responses API** with `text.format` (or `response.parse(text_format=Model)`); fallback Chat Completions `response_format` / `parse(response_format=Model)` |
| Hook the model up to tools / DB / UI | **Function calling** (with `strict: true` for SO on tool args) |
| Need only valid JSON (no schema enforcement) | JSON mode: `text.format = { type: "json_object" }` (Responses) or `response_format = { type: "json_object" }` (Chat) |

**Default to the Responses API** — the official guide focuses non-function-calling examples there. Use Chat Completions only if your codepath requires it.

Supported models: GPT-4o (`gpt-4o-2024-08-06`, `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`) **and later** snapshots. Older models (`gpt-4-turbo`, `gpt-3.5-turbo`) only support JSON mode.

## SDK helpers (preferred)

Define the schema once in code via **Pydantic** (Python) or **Zod** (JS) and let the SDK generate/validate the JSON Schema. This avoids drift between schema and types.

### Responses API — Python (`client.responses.parse`)

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed  # already typed as CalendarEvent
```

`response.output_parsed` is the SDK shortcut. Walk `response.output[].content[]` only when you need to inspect refusals or other content items.

### Responses API — JavaScript (`openai.responses.parse`)

```javascript
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const openai = new OpenAI();

const CalendarEvent = z.object({
  name: z.string(),
  date: z.string(),
  participants: z.array(z.string()),
});

const response = await openai.responses.parse({
  model: "gpt-4o-2024-08-06",
  input: [
    { role: "system", content: "Extract the event information." },
    { role: "user", content: "Alice and Bob are going to a science fair on Friday." },
  ],
  text: { format: zodTextFormat(CalendarEvent, "event") },
});

const event = response.output_parsed;
```

### Chat Completions — Python (`client.chat.completions.parse`)

```python
completion = client.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)
event = completion.choices[0].message.parsed
```

### Chat Completions — JavaScript

```javascript
import { zodResponseFormat } from "openai/helpers/zod";

const completion = await openai.chat.completions.parse({
  model: "gpt-4o-2024-08-06",
  messages: [...],
  response_format: zodResponseFormat(CalendarEvent, "event"),
});
const event = completion.choices[0].message.parsed;
```

## Manual JSON Schema (no SDK helper)

The two APIs use **different shapes** — easy to get wrong.

### Responses API — `text.format`

`name`, `schema`, `strict` live at the format level (NOT nested under another `json_schema` key):

```python
response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input=[...],
    text={
        "format": {
            "type": "json_schema",
            "name": "math_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "output": {"type": "string"},
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": False,
                        },
                    },
                    "final_answer": {"type": "string"},
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False,
            },
        }
    },
)
print(response.output_text)
```

### Chat Completions — `response_format`

`name` and `schema` are nested **under** `json_schema`:

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "math_response",
            "strict": True,
            "schema": { ... },  # same JSON Schema body
        },
    },
)
print(response.choices[0].message.content)
```

Note: the **first** request with a given schema has extra latency while the API compiles it; subsequent requests with the same schema do not.

## Refusals

Models may refuse user-generated input for safety. A refusal does **not** match your schema, so the API surfaces it explicitly.

### Detecting refusals

- **Chat Completions parse:** `completion.choices[0].message.refusal` (string) is set instead of `parsed`/`content`.
- **Responses API:** iterate `response.output[].content[]`; an item with `type == "refusal"` carries `item.refusal`.

```python
msg = completion.choices[0].message
if msg.refusal:
    handle_refusal(msg.refusal)
else:
    use(msg.parsed)
```

```python
for output in response.output:
    if output.type != "message":
        continue
    for item in output.content:
        if item.type == "refusal":
            handle_refusal(item.refusal)
        elif item.type == "output_text":
            use_text(item.text)
```

When using `responses.parse`, prefer `response.output_parsed` for the happy path and only walk `output` when you need to detect refusals.

## Edge cases (always handle)

A response may be valid JSON, a refusal, or **truncated/blocked**.

### Responses API

- `response.status == "incomplete"` with `response.incomplete_details.reason == "max_output_tokens"` → response was cut off; do not parse `output_parsed`.
- `response.incomplete_details.reason == "content_filter"` → safety filter halted generation.
- Otherwise check `response.status == "completed"` before consuming `output_parsed` / `output_text`.

### Chat Completions

Inspect `choices[0].finish_reason`:

- `"length"` → max tokens hit; result likely truncated.
- `"content_filter"` → blocked; partial.
- `"stop"` → normal completion (or your stop token); safe to parse.
- Plus `message.refusal` as above.

## Streaming

Use SDK stream helpers — they parse partial JSON and surface dedicated events.

### Responses (`client.responses.stream` / `openai.responses.stream`)

Event types to watch:

- `response.output_text.delta`, `response.output_text.done`
- `response.refusal.delta`
- `response.error`
- `response.completed`

Get the final object with `stream.get_final_response()` (Python) / `stream.finalResponse()` (JS).

### Chat Completions (`client.beta.chat.completions.stream` / `openai.beta.chat.completions.stream`)

Event types: `content.delta` (with `parsed` snapshot), `content.done`, `refusal.delta`, `refusal.done`, `error`. Get final with `stream.get_final_completion()` / `stream.finalChatCompletion()`.

## Supported schema subset

Structured Outputs accepts a **subset** of JSON Schema. If you violate it with `strict: true`, the API errors.

### Allowed types
`string`, `number`, `integer`, `boolean`, `object`, `array`, `enum`, `anyOf`.

### String constraints
`pattern`, `format` (one of: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`).

### Number constraints
`multipleOf`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`.

### Array constraints
`minItems`, `maxItems`.

(Fine-tuned models do **not** yet support the per-type constraints above, nor `pattern`/`format`/`patternProperties`.)

### Hard requirements (very common pitfalls)

1. **Root must be an `object`** — not `anyOf`, not an array. Zod discriminated unions at the top level are invalid; wrap them in an object property.
2. **`additionalProperties: false`** must be set on **every** object schema.
3. **All keys must be in `required`** for every object. Optional fields are emulated with nullable unions: `"type": ["string", "null"]`.
4. **Limits**: ≤ 5000 total object properties, ≤ 10 nesting levels, ≤ 1000 enum values across all enums, ≤ 15,000 chars for any single enum >250 entries, ≤ 120,000 chars total for property/definition/enum/const names.
5. **Not supported**: `allOf`, `not`, `dependentRequired`, `dependentSchemas`, `if`/`then`/`else`.
6. **Supported**: `$defs` and recursion via `"$ref": "#"` or named refs.
7. **Key ordering**: output preserves the order of keys in your schema — useful for chain-of-thought style outputs (put reasoning before final answer).

## Best practices

- **Schema-as-code**: prefer Pydantic/Zod helpers; if you must hand-write JSON Schema, generate one from the other in CI to prevent drift.
- **User-generated input**: include in the prompt what to do when the input doesn't fit the schema (e.g., return empty arrays, a sentinel sentence). Otherwise the model will hallucinate a schema-shaped answer.
- **Quality**: name keys clearly; add `description`s on important fields; iterate via evals.
- **Reduce mistakes**: split complex tasks into smaller schemas; provide few-shot examples in the system prompt.
- **Place reasoning before conclusions** in the schema (key ordering matters for output quality).

## JSON mode (legacy fallback)

Only use when Structured Outputs is unsupported (older models). Enable with:

- Responses: `text: { format: { type: "json_object" } }`
- Chat Completions: `response_format: { type: "json_object" }`

You **must** mention "JSON" in the prompt or the API errors. JSON mode validates JSON-ness only; you must validate against your schema in app code and check `finish_reason` / `status` for truncation/refusal.

## Resources

- Guide: https://developers.openai.com/api/docs/guides/structured-outputs
- Python helpers: https://github.com/openai/openai-python/blob/main/helpers.md#structured-outputs-parsing-helpers
- Node helpers: https://github.com/openai/openai-node/blob/master/helpers.md#structured-outputs-parsing-helpers
- Cookbook: https://developers.openai.com/cookbook/examples/structured_outputs_intro
