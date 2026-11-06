"""FastMCP tool registration."""

from __future__ import annotations

try:
    from fastmcp import FastMCP
except ImportError:  # official Python SDK package name
    from mcp.server.fastmcp import FastMCP

from services.context import AppContext


mcp = FastMCP("Remote CUDA Development")
_context: AppContext | None = None


def set_context(context: AppContext) -> None:
    global _context
    _context = context


def get_context() -> AppContext:
    if _context is None:
        raise RuntimeError("MCP application context is not initialized")
    return _context
