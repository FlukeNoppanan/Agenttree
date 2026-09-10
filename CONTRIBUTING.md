# Contributing to AgentTree

AgentTree requires Python 3.10 or newer. Create and activate a virtual
environment, then install the project in editable mode:

```sh
python -m venv .venv
python -m pip install -e ".[dev]"
```

Run the complete offline checks before submitting a change:

```sh
python -m pytest -q
python -m compileall -q src tests examples evaluation
python evaluation/run_evaluation.py
```

Keep the framework core domain-independent. Application-specific agents,
capabilities, prompts, and policies belong in host applications or examples.
Lower-level packages must not import the `AgentTree` facade. Optional provider,
LangGraph, and MCP dependencies must remain import-safe when absent, and tests
must not contact live model providers or external MCP servers.

Use short imperative commit subjects and describe validation in pull requests.
Update `README.md`, relevant files under `docs/`, and the progress history in
`AGENTS.md` when a milestone changes public behavior or architecture.
