# Phase 7 streaming and artifact audit

This audit preceded the first implementation pass. The initial Phase 7 attachment ended mid-section 88; the continuation brief later supplied sections 88–164, which are addressed in `docs/artifacts-and-streaming.md` and the Phase 7 regression tests.

## Current output and persistence

`FinalResult` has `final_output` for Root's human-facing answer, `content`/`orchestration` for structured diagnostics, usage, metadata, and trace. It has no artifact references. `AgentResult.output` and `ToolResult.output` can contain structured values, but neither defines an artifact protocol; treating arbitrary Markdown code fences as files would be unsafe. `ProviderResponse` has text, complete tool calls, usage, and optional raw response. `BaseProvider.generate()` is synchronous. `ProviderCapabilities.streaming` exists but defaults to unknown. No provider implements a normalized stream contract.

`ExecutionRuntime` and both stores have bounded cursor reads over a durable per-execution `DurableEvent` sequence. Operation transitions and their events commit together in SQLite. There is no wait/iterator API; a host must poll. Phase checkpoints, stable logical operation keys, physical attempts, and committed-result reuse already cover Root planning/synthesis, Manager decomposition/review/collaboration, Specialist execution/revisions, provider calls, and Tool calls. A new stream must reuse this journal rather than create another event database.

The current providers include Gemini's `models.generate_content`, an OpenAI-compatible HTTP chat adapter shared by Groq, OpenRouter, and Cerebras, plus a separate optional OpenAI SDK adapter. Gemini and the compatible adapter force nonstreaming calls. ToolSession owns authorization and full tool-call validation; streamed partial function arguments must never reach `ToolSession.invoke`. Existing usage collection counts only normalized completed `ProviderResponse` values. Cancellation/deadline checks surround provider and Tool turns; a streaming iterator must check them between chunks.

## Integration choices

- Add a backward-compatible `FinalResult.artifacts` tuple of safe references, with the Root answer remaining in `final_output`.
- Use an execution-scoped artifact session to validate typed content and trusted producer identity. Store immutable content under generated IDs within a dedicated root, never under a model-supplied logical path. File intents remain data; Core does not apply them.
- Tie artifact creation to a stable logical key and commit metadata/content before the producer operation returns. Recovery can then reuse the same reference. Orphan physical content after a crash before metadata commit may remain unlisted; it must not appear in final results.
- Add a cursor-based event iterator over the Phase 6 journal. Polling SQLite with a bounded interval and page size is sufficient; multiple consumers own independent cursors. It is an in-process API, not HTTP streaming.
- Define an optional provider stream contract that yields normalized complete response at the end and text deltas beforehand. High-frequency text deltas should be transient and bounded; lifecycle, artifact, and committed-output milestones stay durable. The existing `generate()` path remains valid for providers without streaming.
- A structured provider-neutral artifact declaration should be explicit. A host/runtime helper and a designated Tool contract can emit artifacts; random code fences in free text cannot be authoritative.

## Official protocol references

- [Google's Generate Content streaming documentation](https://ai.google.dev/gemini-api/docs/generate-content/text-generation) documents `models.generate_content_stream` yielding `GenerateContentResponse` chunks. [Google's migration guide](https://ai.google.dev/gemini-api/docs/migrate-to-interactions) notes complete function-call objects in this legacy stream, unlike incremental argument deltas in the newer Interactions API. AgentTree currently uses `generate_content`, so the former is the compatible integration point.
- [OpenAI streaming API reference](https://platform.openai.com/docs/api-reference/responses-streaming/response/function_call_arguments/delta) demonstrates incremental function argument data. The shared compatible adapter uses Chat Completions rather than Responses, so its stream parser must accumulate indexed chat `tool_calls` deltas to complete calls before authorization and invocation.

The design does not add a second orchestrator, network service, distributed queue, or automatic workspace file application.
