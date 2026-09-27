# Phase 7.5 Groq live provider smoke gate

## Pre-change audit

AgentTree routes all model calls through `BaseProvider` and normalized
`ProviderRequest`/`ProviderResponse`. `GroqProvider` supplies Groq's fixed
OpenAI-compatible endpoint to `OpenAICompatibleProvider`; no Groq branch exists
in orchestration. `create_provider("groq", ...)` uses that class. The adapter
keeps the Bearer credential private, blocks redirects, bounds HTTP bodies,
normalizes errors, caches discovered models, and supports synchronous chat
plus Chat Completions SSE. Its SSE parser accumulates indexed Tool call
fragments into one complete `ProviderResponse` before `ToolSession` validates
and authorizes calls. `AgentTreeConfig(provider_streaming=True)` selects native
streaming; `AgentOutputDelta` is a bounded, transient host channel. Durable
operation, Tool, collaboration, artifact, and final-output events remain in the
execution journal. Artifacts are separate from model text and require an
explicitly bound runtime Tool or host call.

`tests/test_live_providers.py` currently checks Groq model discovery and one
nonstreaming greeting. It does **not** check cache refresh, native streaming,
live output before terminal status, live Tool round trips, Root synthesis,
artifact emission, Manager collaboration, or the background runtime. The
combined four-provider test requires all four credentials. Offline tests cover
these contracts with loopback/fake responses, including streamed Tool
fragment assembly, but cannot establish Groq service behavior.

The generic live-test model chooser accepts returned text modality metadata.
Groq's published `/models` response does not promise those modality fields, so
a successful discovery could be incorrectly skipped. The smoke gate should
select a model **from the authenticated account's discovered IDs**, using
official model capability documentation or an explicit `GROQ_TEST_MODEL`
override for ambiguous results. A model appearing in documentation alone is
insufficient. Model-level streaming and local Tool behavior must still be
observed in live calls.

The credential-free chooser regression now verifies that sparse Groq
discovery selects only an account-visible, documented chat model and avoids
an audio model. This corrects a **test assumption**; it does not change the
production provider adapter or assert a live model capability.

## Current official contract compared with the adapter

- [Groq API reference](https://console.groq.com/docs/api-reference): Bearer
  authentication, `GET /openai/v1/models`, and `POST /openai/v1/chat/completions`
  match the configured endpoint. Chat streaming uses data-only SSE and ends
  with `[DONE]`; `stream_options` is available only with `stream: true`.
- [Groq local Tool calling](https://console.groq.com/docs/tool-use/local-tool-calling):
  local functions require a model request containing schemas, complete Tool
  calls, application-side execution, and a subsequent Tool result turn.
  `tool_choice="auto"` may elect not to call a Tool. The adapter sends schemas
  and Tool history through the same chat endpoint, and Core owns authorization.
- [Groq supported models](https://console.groq.com/docs/models) lists text
  models and capabilities, but account discovery is authoritative for what can
  be tested. Model IDs and capabilities can change. A discovered text model may
  not support local Tool use; test model selection must account for that.
- [Groq error codes](https://console.groq.com/docs/errors) and
  [rate limits](https://console.groq.com/docs/rate-limits) document 401/403,
  400, and 429 responses. The adapter maps status codes to safe typed errors
  and does not include provider response bodies or credentials in messages.
  The gate must avoid deliberate rate-limit traffic.
- Groq may return usage for nonstreaming completions and an optional final
  streaming usage chunk. AgentTree records only reported normalized usage.
  Fragmentation of a particular live Tool call is model-dependent; offline
  fixtures remain the proof of fragment assembly if a live call does not
  fragment.

## Credential and gate status

`GROQ_API_KEY`: **missing** in this environment. No real authentication,
discovery, generation, streaming, Tool call, Root synthesis, artifact, Manager
collaboration, or background run can be claimed. No live model ID was selected
or tested. Existing credential-gated tests must continue to skip cleanly.

The live gate passes only after an authenticated Groq discovery and real calls
prove generation, native streaming, a host-visible `AgentOutputDelta` before
terminal status, a model-requested local Tool round trip, Root synthesis,
normalized reported usage, safe credential handling, and a passing offline
suite. Artifact and Manager collaboration live smokes should be attempted
after those primary checks. Tool or artifact nonselection by the model should
be recorded as model behavior unless a contract defect is demonstrated.
