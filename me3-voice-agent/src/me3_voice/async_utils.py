"""Small adapters for bounded synchronous work used by async runtime paths."""

import asyncio
from concurrent.futures import Future


async def await_thread_future(future: Future, *, poll_interval=0.005):
    """Wait for a submitted thread job without relying on cross-thread loop wakeups.

    The HPC runtime currently loses some selector wakeups from worker threads.
    A short loop-owned timer keeps the async loop responsive while still keeping
    blocking filesystem work off its thread.
    """
    if not isinstance(future, Future) or not 0 < poll_interval <= 0.1:
        raise ValueError("invalid thread future or polling interval")
    while not future.done():
        await asyncio.sleep(poll_interval)
    return future.result()
