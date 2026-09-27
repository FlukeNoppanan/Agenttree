# Artifacts and streaming

Phase 7 adds execution-owned structured outputs and two independent streams.

## Structured artifacts

An `ArtifactSession` validates `text`, `code`, `json`, `file`, `patch`, and
`reference` content and stores immutable bytes under a generated artifact ID.
Its `ArtifactRef` records the content hash, size, media type, creator, and an
optional relative logical path. `create`, `modify`, and `delete` are declared
file intents. AgentTree never applies these intents to a workspace.

Hosts can call `current_artifact_session().create(...)` during an execution.
For model-directed output, explicitly register and bind
`create_artifact_tool()` to trusted agents. The tool accepts a structured
function call and returns an artifact ID and hash. Free-form Markdown code
fences are not treated as files. Root, Manager, and Specialist agents can use
the same tool when bound. Artifacts created by other registered tools are
also tracked in the execution session.

`FinalResult.artifacts` contains committed references. With a SQLite execution
store, the runtime uses a neighboring file artifact store by default. Hosts
can pass an explicit store. `handle.artifacts()` lists committed references;
`handle.artifact(artifact_id)` reads validated bytes for the same execution.
Size, count, path, and metadata limits are configured on `AgentTreeConfig`.

The model controls type, name, content, optional logical path, file intent,
media type, and an optional superseded artifact ID. The runtime assigns the
execution ID, artifact ID, owner, logical operation, timestamp, hash, size,
and physical storage path. The logical path is validated as a relative POSIX
path; it is never used as the storage filename. File store reads verify the
recorded SHA-256 and size. Cross-execution reads fail. Within a running Tree,
Root can inspect all artifacts; Manager sees its branch; Specialist and its
tools see artifacts produced by that Specialist. References are not injected
into unrelated Specialist contexts. A host-owned handle can inspect all
committed execution artifacts.

An artifact written inside an operation is staged first. It becomes visible
as a committed artifact only after the producing journal operation commits.
`artifact.staged` and `artifact.committed` events contain IDs and status, not
bodies. A process loss can leave orphan physical bytes; these are excluded
from committed listings until the operation is recovered. Existing committed
IDs and hashes survive restart. Revisions create new immutable artifacts and
may name a prior artifact from the same execution through
`supersedes_artifact_id`. The runtime validates that link. All committed
versions stay available in execution history. `FinalResult.artifacts` selects
only the latest Manager-approved Specialist revision and successful Root
artifacts. A rejected or superseded draft stays historical. Partial or failed
runs can still expose committed history through the handle without claiming
it as a successful final deliverable.

Root synthesis receives accepted artifact reference metadata (ID, type,
logical path, intent, media type, hash, and size). It does not receive large
artifact bodies automatically. Manager collaboration can mention an artifact
ID in its bounded, permission-checked message, but there is no automatic
cross-branch body transfer. Tools create artifacts only when explicitly
registered and bound, under the existing Tool authorization, budgets, journal
replay policy, timeout, and metrics. Artifact creation is a runtime output
primitive; the model cannot supply provenance fields.

## Execution events

`handle.stream_events(after_sequence=0, timeout=None, stop_on_terminal=True)`
iterates durable `DurableEvent` records in sequence order. Each consumer owns
its cursor. The iterator reads bounded pages and can resume after a process
restart with the last seen sequence. Ending an iterator does not cancel work.
Lifecycle, phase, tool, artifact, and committed operation events are durable.
`output.final.available` is a durable milestone containing the final answer's
hash and byte size. The final answer itself is retrieved from `handle.result()`
after terminal state. Durable cursor consumers remain independent and can
reconnect after a process restart.

## Provider streams

Set `AgentTreeConfig(provider_streaming=True)` to use native streaming when a
provider declares `capabilities.streaming=True`. Otherwise AgentTree calls
`generate()` as before. Gemini uses `generate_content_stream`; Groq,
OpenRouter, and Cerebras use the shared OpenAI-compatible Chat Completions SSE
parser. A custom OpenAI-compatible endpoint must opt in with `streaming=True`.
Other adapters currently use nonstreaming generation.

| Adapter | `generate()` | Native stream | Streamed tools | Offline tests | Live test |
| --- | --- | --- | --- | --- | --- |
| Gemini | Yes | Yes, `generate_content_stream` | Complete function calls | Adapter fixture | Not run; no credential |
| Groq | Yes | Yes, Chat Completions SSE | Indexed fragments assembled | Shared adapter fixture | Not run; no credential |
| OpenRouter | Yes | Yes, Chat Completions SSE | Indexed fragments assembled | Shared adapter fixture | Not run; no credential |
| Cerebras | Yes | Yes, Chat Completions SSE | Indexed fragments assembled | Shared adapter fixture | Not run; no credential |
| Custom compatible | Yes | Opt in with `streaming=True` | Indexed fragments assembled | Shared adapter fixture | Not run; endpoint dependent |

Usage is recorded only when a complete response reports it. Gemini reads
`usage_metadata`; compatible chat streams read an optional `usage` chunk.
There is no usage estimate for partial or lost attempts. The separate OpenAI
Responses adapter and Ollama adapter still use `generate()` here.

`generate_stream()` yields `ProviderStreamChunk` text deltas and exactly one
complete `ProviderResponse`. Tool call fragments are assembled before the
response reaches tool authorization. Partial text is transient within the
provider iterator and is not a committed provider result or durable execution
event. Cancellation and deadline checks run between chunks. The provider
operation remains within the Phase 6 journal attempt; a broken stream does
not commit a partial response. Usage is recorded from the complete response.

`handle.stream_output(timeout=...)` exposes future `AgentOutputDelta` records
while a run is active. Each delta has execution ID, role, Agent ID, logical
operation ID, attempt, process-local sequence, timestamp, and text. It is a
separate process-local channel because token deltas have different retention
than durable events. Root synthesis and Specialist generation emit human
readable text; structured Root planning, Manager decomposition/review,
collaboration JSON, and tool argument fragments are kept internal. Manager
status and Tool activity remain visible through durable events. No provider
SDK object, HTTP header, or credential is exposed through this contract.

Each live subscriber has an independent bounded queue of 64 deltas, and a run
allows at most 32 simultaneous subscribers. If a subscriber
falls behind, oldest deltas are dropped and the next received delta reports
`dropped_before`. Late subscribers receive only future deltas. There is no
token-perfect replay after disconnect or process loss. Use the durable event
cursor, execution status, artifact listing, and final result to reconstruct
state. Delta order is preserved within each provider operation; concurrent
Agents do not have a promised global text order. `max_provider_stream_bytes`
limits accumulated text, and `max_tool_argument_bytes` limits completed tool
arguments. Adapter parsers also bound raw SSE and tool fragments. Cancellation
and deadlines stop downstream commits; an already blocked physical provider
call may continue until its transport returns. Process loss leaves its
incomplete provider attempt uncertain; recovery starts a new physical attempt
under the existing journal policy. Usage from the lost partial stream is not
invented or counted.

The future Studio/API can map run start, status, cursor events, artifact list
and retrieval, and cancellation onto its own HTTP routes such as
`POST /api/v2/runs`, `GET /api/v2/runs/{run_id}/events?after=...`, and
`GET /api/v2/runs/{run_id}/artifacts`. SSE or another transport can carry live
deltas there; Core provides no HTTP server. A future Coding Agenttree client
can show live output and artifact previews, then ask its user to accept or
reject a file intent before applying it locally. Core stops at production and
never runs patch or Git apply.
