"""Documented wire adapters. Live transport is opt-in; offline peers inject a transport.

Deepgram listen/speak v1 WebSockets and Groq-compatible chat-completions SSE.
Selecting chat-completions-v1 explicitly asserts that contract; no HPC API is guessed.
No retries, provider-body logging, hardware actions or implicit account discovery.
"""

import asyncio
import json
import os
import re
import io
import wave
import base64
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlencode

import aiohttp

from .configuration import ConfigurationError
from .contracts import FRAME_BYTES, MAX_AUDIO_BYTES
from .english import (
    QWEN_ASR_TASK_INSTRUCTIONS,
    EnglishTextRejected,
    QWEN_TTS_LEGACY_STYLE_INSTRUCTIONS,
    QWEN_TTS_STYLE_INSTRUCTIONS,
    SWIFT_ENGLISH_OUTPUT_POLICY,
    has_non_latin_letters,
    rejects_english_asr,
)
from .providers import ASRResult, BufferedASRStream, MockASR, MockLLM, MockTTS, ToolCall


class ProviderError(RuntimeError):
    """Safe symbolic failure; never carries a response body, URL or credential."""

    def __init__(
        self,
        message,
        *,
        code="provider_error",
        role=None,
        error_stage=None,
        error_boundary=None,
        error_detail=None,
    ):
        super().__init__(message)
        self.code = code
        self.role = role
        self.error_stage = error_stage
        self.error_boundary = error_boundary
        self.error_detail = error_detail


def qwen_tts_instructions(spec):
    """Keep configured style preferences while applying Swift's delivery guardrails."""
    configured = spec.instructions.strip() if isinstance(spec.instructions, str) else ""
    if not configured or configured in {
        QWEN_TTS_STYLE_INSTRUCTIONS,
        QWEN_TTS_LEGACY_STYLE_INSTRUCTIONS,
    }:
        return QWEN_TTS_STYLE_INSTRUCTIONS
    return configured + "\n\nRequired Swift delivery policy: " + QWEN_TTS_STYLE_INSTRUCTIONS


class LiveTransport:
    def __init__(self, spec, credentials=None):
        self.spec, self.requests, self.client = spec, 0, None
        self.credentials = credentials or {}
        self.on_event = lambda name: None

    def authorize(self):
        if not self.spec.allow_live or self.spec.request_limit == 0:
            raise ConfigurationError(
                "live requests disabled: set allow_live=true and request_limit to a positive cap or null for provider-enforced limits"
            )
        if self.spec.request_limit is not None and self.requests >= self.spec.request_limit:
            raise ProviderError(
                "configured provider request allowance exhausted; no request was sent",
                code="request_limit_reached",
            )
        key = self.credentials.get(self.spec.auth_env) or os.environ.get(self.spec.auth_env, "")
        if not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ConfigurationError(
                "configured project credential is missing or invalid; set the selected auth_env server-side"
            )
        self.requests += 1
        return key

    @staticmethod
    def status_error(status, role):
        code = (
            "provider_rate_limited"
            if status == 429
            else "provider_timeout"
            if status in {408, 504}
            else "provider_unavailable"
            if status >= 500
            else "provider_rejected"
        )
        stage = "tts" if role == "tts" else "reasoning"
        error_role = "tts" if role == "tts" else "llm"
        return ProviderError(
            f"{role} service HTTP {status}; payload excluded; no automatic retry",
            code=code,
            role=error_role,
            error_stage=stage,
        )

    def session(self):
        if self.client is None:
            self.client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self.spec.timeout_seconds,
                    sock_read=self.spec.timeout_seconds,
                ),
                trust_env=False,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        return self.client

    @asynccontextmanager
    async def socket(self, params):
        key = self.authorize()
        try:
            async with self.session().ws_connect(
                self.spec.endpoint + "?" + urlencode(params),
                headers={"Authorization": "Token " + key},
                max_msg_size=131072,
                heartbeat=10,
                timeout=aiohttp.ClientWSTimeout(ws_receive=self.spec.timeout_seconds, ws_close=1),
            ) as peer:
                yield Socket(peer, self.spec.timeout_seconds)
        except (ConfigurationError, ProviderError):
            raise
        except Exception:
            raise ProviderError("speech connection failed or timed out; payload excluded") from None

    async def events(self, body, *, terminal_type=None, total_timeout=True, role="llm"):
        key = self.authorize()
        service_name = "tts" if role == "tts" else "reasoning"
        error_stage = "tts" if role == "tts" else "reasoning"
        event_prefix = "tts" if role == "tts" else "llm"
        deadline = self.spec.timeout_seconds if total_timeout else None
        request_timeout = aiohttp.ClientTimeout(
            total=deadline,
            sock_connect=self.spec.timeout_seconds,
            # Streaming turns have no total deadline, but a connected server
            # must still produce data periodically. Otherwise Thinking can
            # persist forever after a silent/stalled SSE response.
            sock_read=self.spec.timeout_seconds,
        )
        headers_received = False
        first_event_received = False
        try:
            self.on_event(f"{event_prefix}_request_submission_started")
            async with asyncio.timeout(deadline):
                async with self.session().post(
                    self.spec.endpoint,
                    headers={"Authorization": "Bearer " + key, "Accept": "text/event-stream"},
                    json=body,
                    allow_redirects=False,
                    timeout=request_timeout,
                ) as response:
                    if response.status != 200:
                        raise self.status_error(response.status, service_name)
                    if response.content_type != "text/event-stream":
                        raise ProviderError(
                            f"{service_name} service did not return SSE",
                            role=role,
                            error_stage=error_stage,
                        )
                    headers_received = True
                    self.on_event(f"{event_prefix}_response_headers_received")
                    data = []
                    size = 0
                    first_event = True
                    async for raw in response.content:
                        if len(raw) > (262144 if terminal_type else 65536):
                            raise ProviderError("SSE line exceeds limit")
                        line = raw.decode("utf-8").rstrip("\r\n")
                        if line.startswith("data:"):
                            data.append(line[5:].lstrip(" "))
                            size += len(line)
                            if size > (262144 if terminal_type else 65536):
                                raise ProviderError("SSE event exceeds limit")
                        elif not line and data:
                            value = "\n".join(data)
                            data, size = [], 0
                            if value == "[DONE]":
                                return
                            try:
                                event = json.loads(value)
                            except json.JSONDecodeError:
                                failure = ProviderError(
                                    f"{service_name} service returned invalid event JSON",
                                    code="provider_invalid_response",
                                    role=role,
                                    error_stage="response_parse",
                                    error_detail="invalid_sse_json",
                                )
                                raise failure from None
                            if first_event:
                                self.on_event(f"{event_prefix}_first_sse_event")
                                first_event = False
                                first_event_received = True
                            yield event
                            if terminal_type and event.get("type") == terminal_type:
                                return
                    raise ProviderError(
                        "reasoning stream disconnected before DONE",
                        code="provider_incomplete_response",
                        role=role,
                        error_stage="response_parse",
                        error_detail="incomplete_sse",
                    )
        except ProviderError as exc:
            if exc.role is None:
                exc.role = role
            if exc.error_stage is None:
                exc.error_stage = error_stage
            raise
        except TimeoutError:
            boundary = (
                "response_headers"
                if not headers_received
                else "first_sse_event"
                if not first_event_received
                else "sse_stream"
            )
            wait_for = {
                "response_headers": "response headers",
                "first_sse_event": "the first SSE event",
                "sse_stream": "the next SSE event",
            }[boundary]
            raise ProviderError(
                f"{service_name} timed out waiting for {wait_for}; payload excluded",
                code="provider_timeout",
                role=role,
                error_stage=error_stage,
                error_boundary=boundary,
            ) from None
        except Exception:
            raise ProviderError(
                f"{service_name} connection failed or malformed stream; payload excluded",
                role=role,
                error_stage=error_stage,
            ) from None

    async def aclose(self):
        if self.client:
            await self.client.close()

    async def json_request(self, body):
        key = self.authorize()
        try:
            async with self.session().post(
                self.spec.endpoint,
                headers={"Authorization": "Bearer " + key},
                json=body,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise self.status_error(response.status, "model")
                raw = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    raw.extend(chunk)
                    if len(raw) > 65536:
                        raise ProviderError("service response exceeds limit")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError
                return result
        except ProviderError:
            raise
        except Exception:
            raise ProviderError(
                "service request failed or malformed result; payload excluded"
            ) from None

    async def transcribe(self, pcm):
        if not pcm or len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES:
            raise ProviderError("invalid bounded PCM16 input")
        key = self.authorize()
        wav = io.BytesIO()
        with wave.open(wav, "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(pcm)
        form = aiohttp.FormData()
        form.add_field("model", self.spec.model)
        form.add_field("response_format", "json")
        if self.spec.protocol == "groq-whisper-v1":
            form.add_field("language", "en")
        form.add_field("file", wav.getvalue(), filename="utterance.wav", content_type="audio/wav")
        try:
            async with self.session().post(
                self.spec.endpoint,
                headers={"Authorization": "Bearer " + key},
                data=form,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise self.status_error(response.status, "ASR")
                # Bound the body before decoding; reject unexpectedly large metadata.
                raw = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    raw.extend(chunk)
                    if len(raw) > 65536:
                        raise ProviderError("ASR response exceeds limit")
                value = json.loads(raw)
                text = value["text"]
                if not isinstance(text, str) or len(text) > 8000:
                    raise ValueError
                self.last_raw_text = text
                return text
        except ProviderError:
            raise
        except Exception:
            raise ProviderError(
                "ASR request failed or malformed response; payload excluded"
            ) from None


class FileTranscriptionASR:
    """OpenAI-compatible file transcription contract; final-only, not native streaming."""

    def __init__(self, spec, transport=None):
        self.spec, self.transport = spec, transport or LiveTransport(spec)

    def open_stream(self):
        return BufferedASRStream(self)

    async def transcribe(self, pcm):
        return await self.transport.transcribe(pcm)

    async def aclose(self):
        await self.transport.aclose()


class GroqWhisperASR(FileTranscriptionASR):
    """Documented Groq Whisper file transcription adapter."""


class VLLMQwen3ASR(FileTranscriptionASR):
    """Dedicated Qwen3-ASR through vLLM's OpenAI audio-transcriptions endpoint."""


class QwenAudioASR(GroqWhisperASR):
    """Qwen2-Audio via deployed vLLM chat input_audio; final-only audio input."""

    TRANSCRIPT_PREFIX = "The person says: '"

    async def transcribe(self, pcm):
        if not pcm or len(pcm) % 2 or len(pcm) > MAX_AUDIO_BYTES:
            raise ProviderError("invalid bounded PCM16 input")
        wav = io.BytesIO()
        with wave.open(wav, "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(pcm)
        result = await self.transport.json_request(
            {
                "model": self.spec.model,
                "messages": (
                    [{"role": "system", "content": "You are a helpful assistant."}]
                    if self.spec.transcription_mode == "prefill"
                    else []
                )
                + [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(wav.getvalue()).decode("ascii"),
                                    "format": "wav",
                                },
                            },
                            {
                                "type": "text",
                                "text": QWEN_ASR_TASK_INSTRUCTIONS,
                            },
                        ],
                    }
                ]
                + (
                    [{"role": "assistant", "content": self.TRANSCRIPT_PREFIX}]
                    if self.spec.transcription_mode == "prefill"
                    else []
                ),
                **(
                    {
                        "continue_final_message": True,
                        "add_generation_prompt": False,
                        "max_tokens": 256,
                    }
                    if self.spec.transcription_mode == "prefill"
                    else {"max_completion_tokens": 1024}
                ),
                "stream": False,
                "temperature": 0,
            }
        )
        try:
            choices = result["choices"]
            text = choices[0]["message"]["content"]
            if (
                len(choices) != 1
                or choices[0]["finish_reason"] != "stop"
                or not isinstance(text, str)
                or len(text) > 8000
            ):
                raise ValueError
            self.last_raw_text = text
            text = text.strip()
            if self.spec.transcription_mode == "prefill" and not text.startswith(
                self.TRANSCRIPT_PREFIX
            ):
                text = self.TRANSCRIPT_PREFIX + text
            # Remove only the complete quoted wrapper observed with the official
            # audio-analysis task cue. Never extract arbitrary quoted answers.
            wrapper = re.fullmatch(r"The person says[:,]?\s*(['\"])(.+)\1\.?", text, re.DOTALL)
            if wrapper:
                text = wrapper.group(2).strip()
            elif self.spec.transcription_mode == "prefill":
                raise ValueError
            if rejects_english_asr(text):
                raise EnglishTextRejected("asr")
            return text
        except EnglishTextRejected:
            raise
        except (KeyError, TypeError, IndexError, ValueError):
            raise ProviderError("Qwen ASR returned invalid/truncated transcript") from None


class QwenTTS:
    """Documented vLLM-Omni speech SSE, native 24 kHz -> internal mono 16 kHz.

    The terminal done event is required before confirming generated speech.
    Closing the HTTP stream cancels locally; remote GPU cancellation is unverified.
    """

    def __init__(self, spec, transport=None):
        self.spec, self.transport = spec, transport or LiveTransport(spec)
        self.on_event = lambda name: None
        self.transport.on_event = lambda name: self.on_event(name)

    async def synthesize(self, text):
        from .preprocessing import PCMPreprocessor

        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ProviderError("TTS text outside limits")
        if has_non_latin_letters(text):
            error = EnglishTextRejected("tts")
            raise error from None
        body = {
            "model": self.spec.model,
            "input": text,
            "voice": self.spec.voice,
            "instructions": qwen_tts_instructions(self.spec),
            "language": self.spec.language,
            "response_format": "pcm",
            "stream": True,
        }
        if self.spec.seed is not None:
            body["seed"] = self.spec.seed
        stream = self.transport.events(
            body,
            terminal_type="speech.audio.done",
            role="tts",
        )
        pre = PCMPreprocessor(24000, 1)
        native, pending = bytearray(), bytearray()
        total, sequence, done = 0, 0, False
        try:
            async for event in stream:
                kind = event.get("type")
                if kind == "speech.audio.delta":
                    # DGX3 may omit optional rate/channel metadata. Use the
                    # documented Qwen native mono format and reject conflicts.
                    try:
                        if (
                            event.get("sample_rate", 24000) != 24000
                            or event.get("channels", 1) != 1
                            or event.get("response_format") != "pcm"
                        ):
                            raise ValueError
                        audio = base64.b64decode(event["audio"], validate=True)
                    except (TypeError, ValueError, KeyError):
                        raise ProviderError("Qwen TTS returned invalid audio format") from None
                    total += len(audio)
                    if total > 24000 * 2 * 30:
                        raise ProviderError("Qwen TTS exceeds 30 second limit")
                    native.extend(audio)
                    while len(native) >= 2:
                        size = min(48000, len(native) // 2 * 2)
                        pending.extend(pre.push(bytes(native[:size]), sequence))
                        sequence += 1
                        del native[:size]
                        while len(pending) >= FRAME_BYTES:
                            yield bytes(pending[:FRAME_BYTES])
                            del pending[:FRAME_BYTES]
                elif kind == "speech.audio.done":
                    if native or not total:
                        raise ProviderError("Qwen TTS returned incomplete/empty PCM")
                    pending.extend(pre.finish())
                    while pending:
                        yield bytes(pending[:FRAME_BYTES])
                        del pending[:FRAME_BYTES]
                    done = True
                    break
                else:
                    raise ProviderError("Qwen TTS returned unsupported/error event")
            if not done:
                raise ProviderError("Qwen TTS disconnected before completion")
        except ProviderError as exc:
            if exc.role is None:
                exc.role = "tts"
            if exc.error_stage is None:
                exc.error_stage = "tts"
            raise
        finally:
            await stream.aclose()

    async def aclose(self):
        await self.transport.aclose()


class Socket:
    def __init__(self, peer, timeout):
        self.peer, self.timeout = peer, timeout

    async def send(self, value):
        if isinstance(value, bytes):
            await self.peer.send_bytes(value)
        else:
            await self.peer.send_json(value)

    async def receive(self):
        try:
            async with asyncio.timeout(self.timeout):
                message = await self.peer.receive()
            if message.type == aiohttp.WSMsgType.BINARY:
                return message.data
            if message.type == aiohttp.WSMsgType.TEXT:
                value = json.loads(message.data)
                if not isinstance(value, dict):
                    raise ProviderError("invalid speech event")
                return value
            raise ProviderError("speech stream disconnected before completion")
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("speech receive failed or timed out; payload excluded") from None


class DeepgramASR:
    def __init__(self, spec, transport=None):
        self.spec, self.transport = spec, transport or LiveTransport(spec)
        self.on_event = lambda name: None
        self.transport.on_event = lambda name: self.on_event(name)

    def open_stream(self):
        return DeepgramASRStream(self)

    async def aclose(self):
        await self.transport.aclose()


class DeepgramASRStream:
    def __init__(self, provider):
        self.provider = provider
        self.context = self.peer = self.reader = None
        self.parts = []
        self.hypothesis = ""
        self.last = None
        self.total = 0
        self.finishing = self.closed = False

    async def start(self):
        self.context = self.provider.transport.socket(
            {
                "model": self.provider.spec.model,
                "encoding": "linear16",
                "sample_rate": 16000,
                "channels": 1,
                "interim_results": "true",
                "endpointing": "false",
            }
        )
        self.peer = await self.context.__aenter__()
        self.reader = asyncio.create_task(self.read())

    async def read(self):
        while True:
            value = await self.peer.receive()
            if not isinstance(value, dict):
                raise ProviderError("unexpected ASR binary response")
            if value.get("type") == "Metadata" and self.finishing:
                return
            if value.get("type") == "Results":
                try:
                    text = value["channel"]["alternatives"][0]["transcript"]
                    if not isinstance(text, str) or type(value.get("is_final")) is not bool:
                        raise ValueError
                    if value["is_final"]:
                        if text:
                            self.parts.append(text)
                        self.hypothesis = ""
                    else:
                        self.hypothesis = text
                    if len(self.text()) > 8000:
                        raise ValueError
                except (KeyError, IndexError, TypeError, ValueError):
                    raise ProviderError("invalid or oversized ASR result") from None
            elif value.get("type") not in {"Metadata", "SpeechStarted", "UtteranceEnd"}:
                raise ProviderError("ASR service returned unsupported/error event")

    def text(self):
        return " ".join(self.parts + ([self.hypothesis] if self.hypothesis else []))

    async def append(self, pcm):
        if self.closed or self.finishing or len(pcm) % 2 or self.total + len(pcm) > MAX_AUDIO_BYTES:
            raise ProviderError("invalid ASR stream input")
        self.total += len(pcm)
        if self.peer is None:
            await self.start()
        if self.reader.done():
            await self.reader
            raise ProviderError("ASR ended before input commit")
        await self.peer.send(pcm)
        await asyncio.sleep(0)
        text = self.text()
        if text != self.last:
            self.last = text
            return ASRResult(text)

    async def finish(self):
        if self.closed or self.peer is None or self.finishing:
            raise ProviderError("ASR has no open input")
        self.finishing = True
        # CloseStream has a documented terminal Metadata response. Finalize's
        # optional from_finalize event alone cannot safely delimit an empty tail.
        await self.peer.send({"type": "CloseStream"})
        await self.reader
        return ASRResult(" ".join(self.parts), final=True)

    async def aclose(self):
        self.closed = True
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.peer:
            await self.context.__aexit__(None, None, None)
            self.peer = None


class DeepgramTTS:
    def __init__(self, spec, transport=None):
        self.spec, self.transport = spec, transport or LiveTransport(spec)

    async def synthesize(self, text):
        if not isinstance(text, str) or not 1 <= len(text) <= 500:
            raise ProviderError("TTS text outside limits")
        async with self.transport.socket(
            {"model": self.spec.model, "encoding": "linear16", "sample_rate": 16000}
        ) as peer:
            complete = False
            pending = bytearray()
            total = 0
            try:
                await peer.send({"type": "Speak", "text": text})
                await peer.send({"type": "Flush"})
                while True:
                    value = await peer.receive()
                    if isinstance(value, bytes):
                        total += len(value)
                        if total > MAX_AUDIO_BYTES:
                            raise ProviderError("TTS duration exceeds limit")
                        pending.extend(value)
                        # Emit usable PCM immediately, including partial packets;
                        # retain only a possible odd byte across wire messages.
                        while len(pending) >= 2:
                            size = min(FRAME_BYTES, len(pending) // 2 * 2)
                            yield bytes(pending[:size])
                            del pending[:size]
                    elif value.get("type") == "Flushed":
                        if pending or not total:
                            raise ProviderError("TTS completed with invalid/empty PCM")
                        complete = True
                        return
                    elif value.get("type") != "Metadata":
                        raise ProviderError("TTS service returned unsupported/error event")
            finally:
                if not complete:
                    with suppress(Exception):
                        async with asyncio.timeout(0.5):
                            await peer.send({"type": "Clear"})

    async def aclose(self):
        await self.transport.aclose()


class ChatReasoning:
    """Explicit chat-completions SSE contract; coherent text chunks, bounded tools.

    Tool calls are held until the stream completes, so incomplete/cancelled
    arguments cannot cause effects. Onit capabilities are stated from the
    bridge's explicit configuration.
    """

    def __init__(
        self,
        spec,
        transport=None,
        *,
        agent_enabled=False,
        agent_directory_enabled=False,
        agent_tavily_enabled=False,
        agent_github_readonly=False,
        legacy_mock_tools=True,
    ):
        self.spec, self.transport = spec, transport or LiveTransport(spec)
        self.on_event = lambda name: None
        self.agent_enabled = agent_enabled
        self.agent_directory_enabled = agent_directory_enabled
        self.agent_tavily_enabled = agent_tavily_enabled
        self.agent_github_readonly = agent_github_readonly
        self.legacy_mock_tools = legacy_mock_tools
        self.transport.on_event = lambda name: self.on_event(name)

    @staticmethod
    def messages(history):
        result = [
            {
                "role": "system",
                "content": "You are Swift, a concise assistant. "
                + SWIFT_ENGLISH_OUTPUT_POLICY
                + " Answer ordinary questions directly in complete grammatical sentences; "
                + "do not stop mid-phrase or return fragments. Use control_music for music and "
                "run_device_action for supported light, display, timer/alarm, time, weather, "
                "and reminder requests. RGB light colors may be standard names or six-digit "
                "#RRGGBB values; brightness accepts any whole percentage from 0 to 100. "
                "Treat explicit user spellings and corrections, especially letter-by-letter "
                "spelling, as authoritative; never silently normalize them. Do not invent a "
                "place identity or distance when uncertain; ask a concise clarification or "
                "say the distance cannot be verified. "
                "Use run_agent_task only for supported user-requested "
                "tasks. Follow tool receipts exactly: distinguish simulated, accepted, "
                "completed, unavailable, and verified; never claim stronger results. "
                "Temperature is a demo setpoint; timers and alarms have no buzzer.",
            }
        ]
        for index, message in enumerate(history):
            if message["role"] == "tool":
                call = message.get("tool_request")
                if not call:
                    raise ProviderError("tool history lacks its original request")
                turn_id = message.get("turn_id")
                spoken_receipt_follows = turn_id is not None and any(
                    later.get("role") == "assistant"
                    and later.get("turn_id") == turn_id
                    and isinstance(later.get("content"), str)
                    and later["content"].strip()
                    for later in history[index + 1 :]
                )
                if spoken_receipt_follows:
                    # The following assistant message already carries the
                    # user-facing receipt; replaying both tool protocol and
                    # prose adds tokens and has caused stalls on some vLLM
                    # Qwen deployments.
                    continue
                try:
                    raw_receipt = json.loads(message.get("content", "{}"))
                except (TypeError, json.JSONDecodeError):
                    raw_receipt = {}
                receipt = {
                    key: raw_receipt[key]
                    for key in ("result_status", "summary", "display", "mode")
                    if isinstance(raw_receipt, dict)
                    and key in raw_receipt
                    and isinstance(raw_receipt[key], (str, bool))
                }
                if "summary" in receipt:
                    receipt["summary"] = receipt["summary"][:240]
                if "display" in receipt:
                    receipt["display"] = receipt["display"][:120]
                context = {"task": call.get("name", "completed action"), **receipt}
                result.append(
                    {
                        "role": "assistant",
                        "content": "Previous task receipt: "
                        + json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                    }
                )
            else:
                result.append({"role": message["role"], "content": message["content"]})
        return result

    def _request_focus(self, history):
        """Select a single relevant task contract; ordinary chat gets no tools."""
        latest_user_index = next(
            (
                index
                for index in range(len(history) - 1, -1, -1)
                if history[index].get("role") == "user"
            ),
            -1,
        )
        if latest_user_index >= 0 and any(
            message.get("role") == "tool" for message in history[latest_user_index + 1 :]
        ):
            # After a task completes, this request summarizes its receipt; never
            # re-advertise the tool and repeat the side effect.
            return None
        user_text = next(
            (
                message.get("content", "")
                for message in reversed(history)
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "",
        )
        text = user_text.casefold()
        previous_tool = next(
            (
                message.get("tool_request", {}).get("name")
                for message in reversed(history)
                if message.get("role") == "tool" and isinstance(message.get("tool_request"), dict)
            ),
            None,
        )

        music_explicit = bool(
            re.search(r"\b(spotify|music|songs?|tracks?|playlist|album|artist|play)\b", text)
        )
        music_follow_up = previous_tool == "control_music" and bool(
            re.search(
                r"\b(actually|instead|no|it|that|next|previous|pause|resume|stop|volume|louder|quieter|down|up)\b",
                text,
            )
        )
        device_explicit = bool(
            re.search(
                r"\b(light|lights|lamp|led|brightness|bright|color|colour|timer|alarm|temperature|reminder|remind|weather|time)\b",
                text,
            )
        )
        explicit_web_search = bool(
            re.search(
                r"\b(search|research|look up|find)\b.{0,80}\b(web|online|internet|sources?|github|repository)\b",
                text,
            )
        )
        agent_verb = bool(
            re.search(
                r"\b(create|make|build|write|draft|generate|edit|modify|update|inspect|summari[sz]e|compare|research)\b",
                text,
            )
        )
        agent_subject = bool(
            re.search(
                r"\b(project|repo|repository|website|webpage|site|file|folder|directory|document|report|code|program|script|plan|draft|proposal|essay|story|readme|app|article|sources?)\b",
                text,
            )
        )
        agent_explicit = explicit_web_search or (agent_verb and agent_subject)
        music = music_explicit or music_follow_up
        # A mixed request needs a follow-up rather than exposing multiple large
        # schemas that the DGX Qwen tool parser has been observed to stall on.
        categories = int(music) + int(device_explicit) + int(agent_explicit)
        if explicit_web_search:
            return "run_agent_task" if self.agent_enabled else None
        if categories > 1:
            return None
        if music:
            return "control_music"
        if device_explicit:
            if "light" in text or "lamp" in text or "led" in text:
                if re.search(r"\b(on|off|turn on|turn off|switch on|switch off)\b", text):
                    return "set_mock_light" if self.legacy_mock_tools else "run_device_action"
                if re.search(r"\b(status|state|read|check)\b", text) and self.legacy_mock_tools:
                    return "read_mock_light"
            return "run_device_action"
        if agent_explicit:
            return "run_agent_task"
        return None

    @staticmethod
    def _music_prompt_history(history):
        """Keep a music request focused while retaining the last owned player receipt.

        General conversation and cancelled ASR turns add noise to a music tool
        request. Qwen's DGX tool path has stalled with otherwise irrelevant
        history. The UI keeps the full transcript; this only compacts the model
        request. A previous completed music action remains available for
        contextual commands such as "pause it" or "turn it down".
        """
        latest_user_index = next(
            (i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"),
            -1,
        )
        if latest_user_index < 0:
            return history

        previous_music_tool_index = next(
            (
                i
                for i in range(latest_user_index - 1, -1, -1)
                if history[i].get("role") == "tool"
                and isinstance(history[i].get("tool_request"), dict)
                and history[i]["tool_request"].get("name") == "control_music"
            ),
            -1,
        )
        if previous_music_tool_index < 0:
            return [history[latest_user_index]]

        prior_tool = history[previous_music_tool_index]
        prior_turn_id = prior_tool.get("turn_id")
        prior_user_index = next(
            (
                i
                for i in range(previous_music_tool_index - 1, -1, -1)
                if history[i].get("role") == "user"
                and (prior_turn_id is None or history[i].get("turn_id") == prior_turn_id)
            ),
            -1,
        )
        selected = []
        if prior_user_index >= 0:
            selected.append(history[prior_user_index])
        selected.append(prior_tool)
        selected.extend(
            message
            for message in history[previous_music_tool_index + 1 : latest_user_index]
            if message.get("role") == "assistant"
            and prior_turn_id is not None
            and message.get("turn_id") == prior_turn_id
        )
        selected.append(history[latest_user_index])
        return selected

    async def respond(self, history):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_mock_light",
                    "description": "Set the simulated light state.",
                    "parameters": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["enabled"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_mock_light",
                    "description": "Read the simulated light state.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_device_action",
                    "description": "Run one supported light, display, timer, time, weather, or reminder action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "light_on",
                                    "light_off",
                                    "set_color",
                                    "set_brightness",
                                    "set_timer",
                                    "set_alarm",
                                    "cancel_schedule",
                                    "set_temperature",
                                    "create_reminder",
                                    "list_reminders",
                                    "get_time",
                                    "get_weather",
                                ],
                            },
                            "value": {
                                "type": "string",
                                "description": "Optional value for the selected action.",
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if not self.legacy_mock_tools:
            tools = tools[2:]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "control_music",
                    "description": (
                        "Control Spotify. Use play for a specific track or artist; use "
                        "play_playlist for a requested mood, genre, chart, or named playlist. "
                        "For play_playlist, send concise discovery terms and optional ISO country "
                        "market. Do not claim a chart ranking unless Spotify confirms the context. "
                        "Other actions control the current player."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "play",
                                    "play_playlist",
                                    "pause",
                                    "resume",
                                    "stop",
                                    "next",
                                    "previous",
                                    "volume_up",
                                    "volume_down",
                                    "volume_set",
                                ],
                            },
                            "query": {"type": "string", "maxLength": 120},
                            "market": {
                                "type": "string",
                                "pattern": "^[A-Z]{2}$",
                                "description": "Optional ISO 3166-1 alpha-2 market for playlist search.",
                            },
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
            }
        )
        if self.agent_enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "run_agent_task",
                        "description": (
                            "Run a confined Onit task. It may create local draft files"
                            + (
                                " or one new empty sibling test-project directory"
                                if self.agent_directory_enabled
                                else ""
                            )
                            + (
                                "; it may perform bounded Tavily web searches"
                                if self.agent_tavily_enabled
                                else ""
                            )
                            + (
                                "; it may read GitHub repository metadata only"
                                if self.agent_github_readonly
                                else ""
                            )
                            + "; it cannot write GitHub repositories, run shell commands, deploy, publish, or control hardware."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "The user's requested task, up to 1000 characters.",
                                }
                            },
                            "required": ["task"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        selected_tool = self._request_focus(history)
        available = {tool["function"]["name"]: tool for tool in tools}
        tools = [available[selected_tool]] if selected_tool in available else []
        prompt_history = (
            self._music_prompt_history(history) if selected_tool == "control_music" else history
        )
        messages = self.messages(prompt_history)
        if tools:
            tool_instructions = {
                "control_music": (
                    "You are Swift. Call the provided control_music tool for music requests. "
                    "Use play only for a concrete track or artist. For mood, genre, chart or "
                    "playlist requests, use play_playlist and preserve the user's wording as "
                    "search terms. Never claim Spotify's ranking proves a top chart. If the "
                    "requested playlist cannot be found, ask for an artist or track."
                ),
                "run_device_action": "You are Swift. Call the provided run_device_action tool for this supported device request. Preserve the requested action and values. For RGB lights, use a standard color name or #RRGGBB and keep brightness percentages from 0 to 100 exact.",
                "run_agent_task": "You are Swift. Call the provided run_agent_task tool for this user-requested task within its configured limits.",
                "set_mock_light": "You are Swift. Call the provided set_mock_light tool for this simulated light request.",
                "read_mock_light": "You are Swift. Call the provided read_mock_light tool for this simulated light status request.",
            }[selected_tool]
            messages[0]["content"] = tool_instructions + " " + SWIFT_ENGLISH_OUTPUT_POLICY
        body = {
            "model": self.spec.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": selected_tool},
            }
        if self.spec.endpoint == "https://openrouter.ai/api/v1/chat/completions":
            if not (self.spec.model.endswith(":free") or self.spec.model == "openrouter/free"):
                raise ProviderError("OpenRouter reasoning requires an explicit free model")
            body["provider"] = {
                "max_price": {"prompt": 0, "completion": 0},
                "require_parameters": True,
                "allow_fallbacks": True,
            }
        # An active voice turn remains pending until the model responds, the
        # user interrupts, or the connection fails. The session's mic stays
        # available for barge-in, so an arbitrary total response deadline is
        # not needed here. Connection establishment remains bounded.
        stream = self.transport.events(body, total_timeout=False)
        pending, total, calls, reason = "", 0, {}, None
        first_delta = True
        first_tool_delta = True
        last_text_character = ""
        try:
            async for event in stream:
                error_detail = "event_not_object"
                try:
                    if not isinstance(event, dict):
                        raise ValueError
                    if "choices" not in event:
                        error_detail = (
                            "provider_error_event"
                            if "error" in event or event.get("type") == "error"
                            else "missing_choices"
                        )
                        raise ValueError
                    error_detail = "choices_not_array"
                    choices = event["choices"]
                    if not isinstance(choices, list):
                        raise ValueError
                    if not choices:
                        continue
                    if len(choices) != 1:
                        error_detail = "unexpected_choice_count"
                        raise ValueError
                    if not isinstance(choices[0], dict):
                        error_detail = "choice_not_object"
                        raise ValueError
                    if choices[0].get("index", 0) != 0:
                        error_detail = "invalid_choice_index"
                        raise ValueError
                    choice = choices[0]
                    reason = choice.get("finish_reason") or reason
                    error_detail = "invalid_delta_shape"
                    delta = choice["delta"]
                    if not isinstance(delta, dict):
                        raise ValueError
                    content = delta.get("content") or ""
                    if not isinstance(content, str):
                        raise ValueError
                    if content and first_delta:
                        self.on_event("llm_first_delta")
                        first_delta = False
                    if content.strip():
                        last_text_character = content.rstrip()[-1]
                    total += len(content)
                    if total > 8000:
                        raise ValueError
                    pending += content
                    while pending:
                        match = re.search(r"[.!?](?:\s|$)", pending)
                        cut = (
                            match.end()
                            if match
                            else (pending.rfind(" ", 0, 400) if len(pending) >= 400 else 0)
                        )
                        if cut <= 0:
                            if len(pending) >= 500:
                                raise ValueError
                            break
                        segment, pending = pending[:cut].strip(), pending[cut:]
                        if segment:
                            yield segment
                    error_detail = "invalid_tool_call_shape"
                    tool_values = delta.get("tool_calls", [])
                    if not isinstance(tool_values, list):
                        raise ValueError
                    for value in tool_values:
                        if not isinstance(value, dict) or not isinstance(
                            value.get("function", {}), dict
                        ):
                            raise ValueError
                        if first_tool_delta:
                            self.on_event("llm_first_tool_delta")
                            first_tool_delta = False
                        index = value["index"]
                        if type(index) is not int or not 0 <= index < 8:
                            raise ValueError
                        target = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        target["id"] += value.get("id", "")
                        for field in ("name", "arguments"):
                            target[field] += value.get("function", {}).get(field, "")
                        if any(len(v) > 4096 for v in target.values()):
                            raise ValueError
                except (KeyError, TypeError, ValueError, IndexError):
                    failure = ProviderError(
                        "invalid or oversized reasoning stream",
                        code="provider_invalid_response",
                        role="llm",
                        error_stage="response_parse",
                        error_detail=error_detail,
                    )
                    raise failure from None
            # Some compatible vLLM/Qwen deployments terminate a complete function
            # call with finish_reason="stop" rather than "tool_calls". The SSE
            # stream must still finish normally, and all call arguments are
            # validated below before any task can execute.
            if reason not in {"stop", "tool_calls"} or (not calls and reason != "stop"):
                raise ProviderError(
                    "reasoning stream incomplete or truncated",
                    code="provider_invalid_response",
                    role="llm",
                    error_stage="response_parse",
                    error_detail="invalid_finish_reason",
                )
            if not total and not calls:
                self.on_event("llm_empty_response")
            if last_text_character:
                self.on_event("llm_text_stream_completed")
                if last_text_character not in ".!?…。！？":
                    self.on_event("llm_text_unpunctuated_finish")
            if pending.strip():
                yield pending.strip()
            ready = []
            if calls:
                self.on_event("llm_tool_parse_started")
            allowed_tools = {
                "set_mock_light",
                "read_mock_light",
                "run_device_action",
                "run_mock_action",
                "control_music",
            }
            if self.agent_enabled:
                allowed_tools.add("run_agent_task")
            advertised_tools = {tool["function"]["name"] for tool in tools}
            for value in calls.values():
                try:
                    call_id, tool_name, raw_arguments = (
                        value["id"],
                        value["name"],
                        value["arguments"],
                    )
                except (KeyError, TypeError):
                    raise ProviderError(
                        "incomplete reasoning tool call",
                        code="provider_invalid_response",
                        role="llm",
                        error_stage="response_parse",
                        error_detail="invalid_tool_call_shape",
                    ) from None
                if not isinstance(call_id, str) or not re.fullmatch(
                    r"[A-Za-z0-9_-]{1,80}", call_id
                ):
                    raise ProviderError(
                        "invalid reasoning tool identity",
                        code="provider_invalid_response",
                        role="llm",
                        error_stage="response_parse",
                        error_detail="invalid_tool_identity",
                    )
                if tool_name not in allowed_tools or tool_name not in advertised_tools:
                    raise ProviderError(
                        "unadvertised reasoning tool name",
                        code="provider_invalid_response",
                        role="llm",
                        error_stage="response_parse",
                        error_detail="unadvertised_tool_name",
                    )
                try:
                    arguments = json.loads(raw_arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError
                    ready.append(ToolCall(call_id, tool_name, arguments))
                except (TypeError, ValueError):
                    failure = ProviderError(
                        "invalid reasoning tool call",
                        code="provider_invalid_response",
                        role="llm",
                        error_stage="response_parse",
                        error_detail="invalid_tool_arguments",
                    )
                    raise failure from None
            if len({call.call_id for call in ready}) != len(ready):
                failure = ProviderError(
                    "duplicate reasoning tool identity",
                    code="provider_invalid_response",
                    role="llm",
                    error_stage="response_parse",
                    error_detail="duplicate_tool_identity",
                )
                raise failure
            if ready:
                self.on_event("llm_tool_parse_finished")
                self.on_event("llm_tool_call_ready")
            for call in ready:
                yield call
        except ProviderError as exc:
            if not hasattr(exc, "role"):
                exc.role = "llm"
            if exc.code == "provider_timeout" and exc.error_boundary == "sse_stream":
                exc.error_boundary = "sse_stream" if not first_delta else "first_text_delta"
            raise
        finally:
            await stream.aclose()

    async def aclose(self):
        await self.transport.aclose()


def make_provider(
    role,
    spec,
    *,
    transcript="Mock question",
    delay=0,
    transport=None,
    credentials=None,
    agent_enabled=False,
    agent_directory_enabled=False,
    agent_tavily_enabled=False,
    agent_github_readonly=False,
    legacy_mock_tools=True,
):
    spec.validate(role, runnable=True)
    if spec.backend == "mock":
        return (
            MockASR(transcript, delay)
            if role == "asr"
            else MockTTS(delay)
            if role == "tts"
            else MockLLM()
        )
    provider = {
        "asr": QwenAudioASR
        if spec.protocol == "vllm-qwen-audio-v1"
        else VLLMQwen3ASR
        if spec.protocol == "vllm-qwen3-asr-v1"
        else GroqWhisperASR
        if spec.protocol == "groq-whisper-v1"
        else DeepgramASR,
        "tts": QwenTTS if spec.protocol == "vllm-qwen-tts-v1" else DeepgramTTS,
        "llm": ChatReasoning,
    }[role]
    if role == "llm":
        return provider(
            spec,
            transport or LiveTransport(spec, credentials),
            agent_enabled=agent_enabled,
            agent_directory_enabled=agent_directory_enabled,
            agent_tavily_enabled=agent_tavily_enabled,
            agent_github_readonly=agent_github_readonly,
            legacy_mock_tools=legacy_mock_tools,
        )
    return provider(spec, transport or LiveTransport(spec, credentials))
