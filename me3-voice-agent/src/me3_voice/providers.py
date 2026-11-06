"""Provider boundaries. Deterministic mocks live here; documented network implementations live in remote.py.

An ASR provider receives PCM16 mono 16 kHz; TTS yields the same format.
Implementations must propagate cancellation and close their own network streams.
"""

import asyncio
import math
import struct
import json
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, TypeVar
from dataclasses import dataclass
from .contracts import MAX_AUDIO_BYTES

from .contracts import FRAME_SAMPLES, RATE


@dataclass(frozen=True)
class ASRResult:
    text: str
    final: bool = False
    provider_text: str | None = None


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict


class ASRStream(Protocol):
    async def append(self, pcm: bytes) -> ASRResult | None: ...
    async def finish(self) -> ASRResult: ...
    async def aclose(self) -> None: ...


class ASRProvider(Protocol):
    def open_stream(self) -> ASRStream: ...


class BufferedASRStream:
    """Explicit final-only compatibility adapter, not native streaming ASR."""

    def __init__(self, provider):
        self.provider, self.audio = provider, bytearray()

    async def append(self, pcm):
        if len(self.audio) + len(pcm) > MAX_AUDIO_BYTES:
            raise ValueError("ASR input exceeds 30 seconds")
        self.audio.extend(pcm)

    async def finish(self):
        text = await self.provider.transcribe(bytes(self.audio))
        return ASRResult(
            text, final=True, provider_text=getattr(self.provider, "last_raw_text", None)
        )

    async def aclose(self):
        self.audio.clear()


StreamItem = TypeVar("StreamItem", covariant=True)


class ClosableStream(Protocol[StreamItem]):
    def __aiter__(self) -> "ClosableStream[StreamItem]": ...
    async def __anext__(self) -> StreamItem: ...
    async def aclose(self) -> None: ...


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> ClosableStream[bytes]: ...


class LLMProvider(Protocol):
    def respond(self, history: Sequence[dict]) -> ClosableStream[str | ToolCall]: ...


class MockASR:
    def __init__(self, transcript: str = "Mock question", delay: float = 0):
        self.transcript = transcript
        self.delay = delay
        self.index = 0

    def open_stream(self):
        transcript = self.transcript
        if isinstance(transcript, (list, tuple)):
            if not transcript:
                raise ValueError("mock transcript script is empty")
            transcript = transcript[min(self.index, len(transcript) - 1)]
        self.index += 1
        return MockASRStream(transcript, self.delay)

    async def transcribe(self, pcm: bytes) -> str:
        stream = self.open_stream()
        try:
            await stream.append(pcm)
            return (await stream.finish()).text
        finally:
            await stream.aclose()


class MockASRStream:
    def __init__(self, text, delay):
        self.text, self.delay = text, delay
        self.voiced, self.closed, self.partial_sent = False, False, False

    async def append(self, pcm):
        if self.closed:
            raise ValueError("ASR stream is closed")
        self.voiced |= any(pcm)
        if self.voiced and not self.partial_sent:
            self.partial_sent = True
            return ASRResult(self.text.split(" ")[0])

    async def finish(self):
        if self.closed:
            raise ValueError("ASR stream is closed")
        await asyncio.sleep(self.delay)
        return ASRResult(self.text if self.voiced else "", final=True)

    async def aclose(self):
        self.closed = True


class MockTTS:
    """Non-speech tone: tests transport only, never intelligibility or acoustics."""

    def __init__(self, delay=0):
        self.delay = delay

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for index in range(4):
            await asyncio.sleep(self.delay)
            yield struct.pack(
                "<" + "h" * FRAME_SAMPLES,
                *(
                    int(1000 * math.sin(2 * math.pi * 440 * (index * FRAME_SAMPLES + n) / RATE))
                    for n in range(FRAME_SAMPLES)
                ),
            )


class MockLLM:
    async def respond(self, history: Sequence[dict]) -> AsyncIterator[str]:
        await asyncio.sleep(0)
        if history[-1]["role"] == "tool":
            value = json.loads(history[-1]["content"])
            if "summary" in value:
                yield value["summary"]
            else:
                yield f"Mock light is {'on' if value['enabled'] else 'off'}. Verified in memory only."
            return
        instruction = history[-1]["content"].strip().lower()
        if instruction.startswith("agent task: "):
            task = history[-1]["content"].strip()[12:].strip()
            if task:
                yield ToolCall("agent-task", "run_agent_task", {"task": task})
                return
        simulated = {
            "simulate light on": ("light_on", None),
            "simulate light off": ("light_off", None),
            "simulate reminders": ("list_reminders", None),
            "simulate time": ("get_time", None),
            "simulate weather": ("get_weather", None),
        }
        for prefix, action in (
            ("simulate color ", "set_color"),
            ("simulate brightness ", "set_brightness"),
            ("simulate timer ", "set_timer"),
            ("simulate alarm ", "set_alarm"),
            ("simulate cancel schedule", "cancel_schedule"),
            ("simulate temperature ", "set_temperature"),
            ("simulate reminder ", "create_reminder"),
        ):
            if instruction.startswith(prefix):
                value = None if action == "cancel_schedule" else instruction.removeprefix(prefix)
                if action == "set_alarm":
                    value = (
                        value.upper().replace("AM", " AM").replace("PM", " PM").replace("  ", " ")
                    )
                simulated[instruction] = action, value
                break
        if instruction in simulated:
            action, value = simulated[instruction]
            args = {"action": action}
            if value is not None:
                args["value"] = value
            yield ToolCall("demo-action", "run_device_action", args)
            return
        if instruction in {"set mock light on", "set mock light off"}:
            yield ToolCall(
                "light-change", "set_mock_light", {"enabled": instruction.endswith(" on")}
            )
            return
        if instruction == "mock light status":
            yield ToolCall("light-read", "read_mock_light", {})
            return
        turns = sum(message["role"] == "user" for message in history)
        yield f"Mock reply to turn {turns}: {history[-1]['content']}"
