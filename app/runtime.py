"""Event loop selection.

Windows defaults to `ProactorEventLoop`, which psycopg's async driver refuses to run
on — it needs a selector-based loop for its socket handling. Every entry point that
creates a loop must therefore create the right one, so this is centralised rather
than repeated (and forgotten) in each `__main__`.

`asyncio.run(..., loop_factory=...)` is used in preference to
`asyncio.set_event_loop_policy()`: policies are deprecated from Python 3.14, and a
factory is explicit about which loop *this* program uses instead of mutating global
interpreter state that later code may depend on.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any


def new_event_loop() -> asyncio.AbstractEventLoop:
    """A loop the database driver can actually use."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """`asyncio.run` with the correct loop for this platform.

    Use this instead of `asyncio.run` anywhere the process owns its own loop:
    workers, the reaper, scripts.
    """
    return asyncio.run(coro, loop_factory=new_event_loop)


__all__ = ["new_event_loop", "run"]
