# Phase 4 Manager collaboration audit (before implementation)

## Current Manager lifecycle and order

`AgentTree.run` creates a copied Task and a per-run orchestration context, provider routing context, usage collector, and ToolSession. The sequential backend calls planning, delegation, execution, Manager review, and final review in that order. Root planning and triage select Manager IDs. `OrchestrationEngine.delegate` iterates selected Managers in deterministic plan order and calls `BaseTaskDecomposer.decompose_with_capabilities` once for each. It selects only registered Specialists owned by that Manager. Execution then invokes selected Specialists sequentially. Manager review iterates each Manager's subtasks and runs the injected reviewer; bounded revisions re-execute only the owning Manager's Specialists. Root final review and synthesis remain after Manager work. The optional LangGraph backend calls the same phase operations.

Provider-backed decomposition and review use `_ProviderDecision._generate`, which resolves the Manager's provider/model binding, calls the shared ToolSession provider loop, records usage, and parses one strict JSON object. Static decomposition/review strategies do not call a provider. There is no Manager message contract, history, peer permission, or provider request for a peer response. Provider routing and Tool assignments are already external to Manager identity.

## Concurrency, failures, and integration

The workflow is synchronous and Managers do not currently run concurrently. The existing ToolSession uses bounded daemon workers for Tool invocation. Traces are phase-local clones or extensions; the SDK merges ToolSession events by timestamp into the final trace. Provider errors may propagate into a failed state or become failed Specialist results, while review/revision outcomes remain bounded.

The narrow integration point is the strict provider decision helper used by Manager decomposition and review: a validated collaboration request can be routed through one execution-scoped session, then its safe outcome can be supplied to the same Manager's next provider decision. A separate responder turn can resolve the target Manager's own provider/model binding and use the existing ToolSession. This leaves phase ordering, Specialist ownership, static strategies, Root authority, and backend equivalence intact.

Synchronous A → B → A recursion could deadlock or multiply calls. The response turn should be non-reentrant: it may use B's assigned Tools but may not start another collaboration request. The session must also bound messages per sender and execution, rounds per thread, request/response payloads, provider wait time, and decision continuations. Directional peer permissions must default to empty. A late timed-out worker must not mutate session history or emit trace events. Existing Trees without permissions must retain their exact provider-call and trace behavior.
