# AgentTree Multi-Domain Evaluation

## Objective

This offline evaluation tests whether the same public AgentTree SDK and
orchestration implementation can be configured for three application domains:
IT/technical incident analysis, HR/document review, and business/product
evaluation. Each domain has two predefined tasks. All inputs, provider outputs,
and personal or market data are synthetic.

The evaluation demonstrates framework properties in these scenarios. It does
not measure model intelligence, domain correctness, enterprise scalability, or
production reliability.

## Scenario setup

Every scenario creates a Root, one expected Manager, two expected Specialists,
an unrelated Manager, and unrelated Specialists. Static triage, decomposition,
and review strategies make the routing expectation reproducible. Separate
MockProviders execute expected Specialists offline. The IT configuration also
registers, binds, and explicitly invokes a FunctionTool through ToolExecutor.
Domain code lives in `examples/domains/`, outside `src/agenttree/`.

## Metrics

- **Reusability:** passes when all six scenarios complete with
  `AgentTree.run(Task)`, the sequential backend is shared, customization paths
  are outside the core, and core source files modified per domain is zero.
- **Extensibility:** passes when a probe adds a new Manager, Specialist, and
  capability through public registration/binding APIs and completes execution.
- **Capability-based routing:** passes per scenario when selected Manager and
  executed Specialist names exactly match expected names and exclude every
  unrelated registered agent.
- **Provider independence:** passes per scenario when equivalent primary and
  alternate MockProvider bindings use different provider identities yet return
  the same status, success flag, and normalized Specialist outputs.
- **Traceability:** passes per scenario when the final trace and every event use
  the Task ID, the public final state holds that trace, and required planning,
  delegation, execution, Manager review, final review, and result labels occur
  in order. Timestamps are not compared.
- **Tool integration:** passes when the registered/bound offline FunctionTool
  succeeds and emits the existing two ToolExecutor events.
- **Optional LangGraph equivalence:** when installed, the first IT scenario runs
  through sequential and LangGraph backends; status, success, outputs, and event
  labels must match. Absence is recorded as `NOT_RUN`, not a base failure.

## Running and interpreting results

From an editable installation at the repository root:

```sh
python evaluation/run_evaluation.py
```

The command prints a compact table and rewrites `evaluation/results.json` using
sorted, indented JSON with no timestamps or credentials. Each scenario row
includes pass/fail fields and evidence: expected/actual names, unrelated
selections, provider variants, normalized outputs, result shape, Task mutation
check, ordered trace positions, and tool proof. `overall` contains counts and
aggregate outcomes suitable for report tables. A nonzero process exit indicates
failure in a required metric (or the optional LangGraph comparison when run).

Results support the narrow conclusion that AgentTree's public configuration,
routing, provider, tool, and trace contracts behave as defined in these offline
fixtures. Broader scientific and operational claims require separate studies.
