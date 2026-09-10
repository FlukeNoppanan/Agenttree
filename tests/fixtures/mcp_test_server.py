"""Deterministic local integration-test server; never part of production tools."""

import os
import sys

from mcp.server.fastmcp import FastMCP


server = FastMCP("AgentTree tests", log_level="ERROR")


@server.tool()
def echo(text: str) -> dict[str, str]:
    """Return exactly the supplied text."""
    return {"text": text}


@server.tool()
def add(a: int, b: int) -> dict[str, int]:
    """Add two integers deterministically."""
    return {"sum": a + b}


@server.tool()
def fail() -> str:
    """Produce an intentional tool error."""
    raise ValueError("Intentional test failure")


@server.tool()
def process_info() -> dict[str, str | int]:
    """Expose only fixture process identity and an explicit test override."""
    return {"pid": os.getpid(), "cwd": os.getcwd(), "marker": os.environ.get("AGENTTREE_TEST_MARKER", "")}


if __name__ == "__main__":
    if "--fail-startup" in sys.argv:
        raise SystemExit(2)
    server.run(transport="stdio")
