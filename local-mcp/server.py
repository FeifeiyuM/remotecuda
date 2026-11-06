"""Local Remote CUDA MCP server entry point."""

from __future__ import annotations

import asyncio
import atexit
import logging

import anyio

from services.context import build_context
from tools import artifacts, benchmark, build, execution, gpu, server, sync, workspace  # noqa: F401
from tools import mcp, set_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("remote-cuda-mcp")
EVENT_LOOP_WAKE_INTERVAL = 0.05


async def run_stdio_with_wakeup() -> None:
    """Run the stdio server while keeping the local event loop responsive.

    FastMCP reads stdin through an AnyIO worker thread. Some local Python 3.13
    builds do not wake the selector when that thread schedules its result back
    on the event loop. A short timer is a harmless compatibility heartbeat and
    ensures incoming MCP frames are processed promptly.
    """

    async def wake_event_loop() -> None:
        while True:
            await anyio.sleep(EVENT_LOOP_WAKE_INTERVAL)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(wake_event_loop)
        try:
            await mcp.run_async(transport="stdio", show_banner=False)
        finally:
            task_group.cancel_scope.cancel()


def main() -> None:
    context = build_context()
    set_context(context)
    atexit.register(lambda: asyncio.run(context.close()))
    LOGGER.info("Remote CUDA MCP gateway ready (%s workspaces)", len(context.workspaces.list()))
    anyio.run(run_stdio_with_wakeup)


if __name__ == "__main__":
    main()
