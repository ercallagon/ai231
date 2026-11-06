"""Playback boundary; the supplied implementation acknowledges only memory writes."""

import asyncio
from typing import Protocol


class PlaybackSink(Protocol):
    async def __call__(self, pcm: bytes) -> None: ...
    async def stop(self) -> None: ...
    async def close(self) -> None: ...


class MemorySink:
    def __init__(self, *, delay=0, capacity=1024):
        self.delay, self.capacity = delay, capacity
        self.chunks = []
        self.stops = 0
        self.closed = False
        self.entered = asyncio.Event()

    async def __call__(self, pcm):
        self.entered.set()
        await asyncio.sleep(self.delay)
        if self.closed or len(self.chunks) >= self.capacity:
            raise ValueError("memory sink closed or retention limit reached")
        self.chunks.append(pcm)

    async def stop(self):
        self.stops += 1

    async def close(self):
        self.closed = True
