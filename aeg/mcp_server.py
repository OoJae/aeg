"""Aeg MCP server — the memory immune system as Model-Context-Protocol tools.

Any MCP client (Claude Desktop, Cursor, an agent runtime …) can drop Aeg in front
of its own memory: screened `remember`, immune-aware `recall`, and the full
detect → forget → reinforce → antibody response — every tool reuses the EXACT
gateway logic by calling the FastAPI app in-process over an ASGI transport (no
duplicated orchestration, same screening/immune/guard code path as HTTP).

    uv run aeg-mcp            # stdio server

Register in an MCP client with command `uv`, args `["run","aeg-mcp"]`, cwd = repo.
"""

from __future__ import annotations

import os

# Trusted local surface reached over stdio — there is no per-IP concept, so the
# per-IP rate limit is disabled here (the daily LLM budget still applies). Set
# before importing aeg.config so it is read at import.
os.environ.setdefault("AEG_RATE_LIMIT", "0")

from typing import Literal

import httpx
from mcp.server.fastmcp import FastMCP

from aeg import config
from aeg.gateway import create_app

config.AEG_RATE_LIMIT = 0  # belt-and-suspenders even if the env was preset

_app = create_app()
_transport = httpx.ASGITransport(app=_app)
mcp = FastMCP("aeg")


async def _call(method: str, path: str, **kw):
    """One in-process call to the gateway app (reuses all its logic)."""
    async with httpx.AsyncClient(transport=_transport, base_url="http://aeg") as c:
        r = await c.request(method, path, timeout=180.0, **kw)
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text[:500]}


@mcp.tool()
async def aeg_remember(
    content: str,
    source_kind: Literal["user", "document", "tool", "agent", "unknown"] = "user",
    identifier: str = "mcp",
    dataset: str = config.DATASET_MAIN,
) -> dict:
    """Screen and store a memory. Injection or known-attack content is blocked at
    the door and never cognified; clean content is typed (Claim/Source) and stored.
    Returns the screening verdict, quarantine status, and any antibody match."""
    return await _call("POST", "/remember", json={
        "content": content,
        "source": {"kind": source_kind, "identifier": identifier},
        "dataset": dataset,
    })


@mcp.tool()
async def aeg_recall(
    query: str,
    dataset: str = config.DATASET_MAIN,
    lane: Literal["graph", "graph_cot"] = "graph",
    top_k: int = 10,
) -> dict:
    """Answer a question from screened memory. Quarantined / poisoned content is
    absent by construction, so recall reflects only trusted knowledge."""
    return await _call("POST", "/recall", json={
        "query": query, "dataset": dataset, "lane": lane, "top_k": top_k,
    })


@mcp.tool()
async def aeg_scan(dataset: str = config.DATASET_MAIN, max_pairs: int = 20) -> dict:
    """Run the adaptive-immunity sweep: traverse the graph, detect contradictions
    with an LLM verifier, and quarantine the lower-trust claim."""
    return await _call("POST", "/scan", json={"dataset": dataset, "max_pairs": max_pairs})


@mcp.tool()
async def aeg_respond(dataset: str = config.DATASET_MAIN, claim_id: str | None = None) -> dict:
    """Verify quarantined memory against the evidence and permanently forget() what
    is confirmed bad; a defeated attack is recorded as an antibody."""
    body: dict = {"dataset": dataset}
    if claim_id:
        body["claim_id"] = claim_id
    return await _call("POST", "/respond", json=body)


@mcp.tool()
async def aeg_reinforce(dataset: str, claim_id: str, note: str | None = None) -> dict:
    """Strengthen a validated claim and bridge a verification note into permanent
    memory via improve()."""
    body: dict = {"dataset": dataset, "claim_id": claim_id}
    if note:
        body["note"] = note
    return await _call("POST", "/reinforce", json=body)


@mcp.tool()
async def aeg_quarantine(dataset: str = config.DATASET_MAIN) -> dict:
    """List the quarantine queue for a dataset."""
    return await _call("GET", "/quarantine", params={"dataset": dataset})


@mcp.tool()
async def aeg_contradictions(dataset: str = config.DATASET_MAIN) -> dict:
    """List detected contradictions for a dataset."""
    return await _call("GET", "/contradictions", params={"dataset": dataset})


@mcp.tool()
async def aeg_health() -> dict:
    """Memory health score and graph metrics."""
    return await _call("GET", "/health")


def main() -> None:
    """Entry point for the `aeg-mcp` console script (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
