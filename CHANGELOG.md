# Changelog

This project records notable changes for each pre-release milestone.

## 0.2.2 - 2026-09-27

- Added Groq, OpenRouter, Cerebras, and custom OpenAI-compatible providers, model
  discovery, provider streaming, and per-Agent provider/model bindings.
- Added bounded Root, Manager, and Specialist Tool sessions; Manager
  collaboration; Root planning and final synthesis; and usage accounting.
- Added background execution, durable execution events and recovery, live
  output deltas, and artifact storage and selection.
- Committed terminal results and their durable completion events together so
  a completed handle cannot expose a result before its completion event.
- Offline regression and clean wheel installation qualify this Core release.
  Live provider calls remain unverified without configured credentials.
- Live deltas are process-local and do not replay after reconnect. Durable
  recovery requires a persistent execution store. Studio currently uses
  `InMemoryExecutionStore`, so its integration does not provide durable Core
  recovery. Live Manager collaboration depends on the selected model
  producing the expected output.

## 0.2.1 - 2026-09-05

- Fixed provider-powered triage so the orchestration engine supplies the
  normalized capabilities of registered Managers and rejects unknown output.
- Fixed provider-powered decomposition so each Manager can request only
  capabilities exposed by its registered, owned Specialists.
- Added routing-choice metadata to existing trace events without including
  provider credentials or raw responses.
- Preserved direct provider-strategy calls and existing rule-based, static,
  sequential, and LangGraph APIs.

## 0.2.0 - 2026-09-05

- Added the high-level `AgentTree` SDK for registering and running Root,
  Manager, and Specialist hierarchies.
- Added deterministic capability-based Manager and Specialist routing.
- Added provider-neutral contracts, offline `MockProvider`, and optional
  OpenAI, Gemini, and Ollama adapters.
- Added Manager and Root review with bounded revision loops.
- Added function tools, MCP tool discovery/execution, and optional real stdio
  and Streamable HTTP MCP clients.
- Added copy-safe workflow state and ordered execution traces.
- Added the default sequential backend and optional LangGraph backend.
- Added offline examples and a reproducible three-domain evaluation suite.
- Added release metadata, typed-package marker, developer documentation, and
  clean wheel installation validation.
- Added deterministic thesis, revision, and local MCP demonstrations plus a
  factual project scope/evidence map.
- Hardened empty provider-output normalization and invalid decision-provider
  response diagnostics during final QA.

AgentTree remains an alpha pre-release. There is no earlier published release
history recorded in this repository.
