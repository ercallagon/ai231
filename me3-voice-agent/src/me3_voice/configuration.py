"""Explicit service contracts and fail-closed live-request authorization."""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from .research import ResearchOptions


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class BackendSpec:
    backend: str = "mock"
    protocol: str = ""
    endpoint: str = ""
    auth_env: str = ""
    model: str = ""
    sample_rate: int = 16000
    channels: int = 1
    audio_format: str = "pcm_s16le"
    streaming: bool = True
    concurrency: int = 1
    timeout_seconds: float = 5
    max_audio_seconds: int = 30
    allow_live: bool = False
    request_limit: int | None = 0
    # Required for live chat-completions. The first bounds one full serialized
    # request; the second is frozen as the run-scoped aggregate campaign cap.
    max_input_tokens: int = 0
    max_input_tokens_total: int = 0
    allow_private_http: bool = False
    voice: str = ""
    language: str = ""
    instructions: str = ""
    seed: int | None = None
    transcription_mode: str = "prompt"

    def validate(self, role, *, runnable=False):
        if self.transcription_mode not in ("prompt", "prefill") or (
            self.transcription_mode == "prefill"
            and (
                role != "asr" or self.backend != "external" or self.protocol != "vllm-qwen-audio-v1"
            )
        ):
            raise ConfigurationError(f"{role}: transcription prefill requires Qwen2-Audio ASR")
        if self.backend not in {"mock", "external"}:
            raise ConfigurationError(f"{role}: select backend mock or external")
        if (
            type(self.allow_private_http) is not bool
            or any(not isinstance(v, str) or len(v) > 80 for v in (self.voice, self.language))
            or not isinstance(self.instructions, str)
            or len(self.instructions) > 500
        ):
            raise ConfigurationError(
                f"{role}: invalid transport/voice/language/instructions configuration"
            )
        if (
            type(self.allow_live) is not bool
            or (self.request_limit is not None and type(self.request_limit) is not int)
            or (self.request_limit is not None and not 0 <= self.request_limit <= 10000)
            or type(self.max_input_tokens) is not int
            or not 0 <= self.max_input_tokens <= 1_000_000
            or type(self.max_input_tokens_total) is not int
            or not 0 <= self.max_input_tokens_total <= 1_000_000
            or (
                self.seed is not None
                and (type(self.seed) is not int or not 0 <= self.seed <= 2147483647)
            )
        ):
            raise ConfigurationError(f"{role}: invalid live authorization/request limit/seed")
        if (
            type(self.concurrency) is not int
            or not 1 <= self.concurrency <= 32
            or type(self.timeout_seconds) not in (float, int)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 120
            or type(self.max_audio_seconds) is not int
            or not 1 <= self.max_audio_seconds <= 30
            or type(self.streaming) is not bool
        ):
            raise ConfigurationError(
                f"{role}: invalid concurrency, timeout, streaming or audio limit"
            )
        if (
            type(self.sample_rate) is not int
            or self.sample_rate not in (8000, 16000, 24000, 32000, 44100, 48000)
            or type(self.channels) is not int
            or self.channels not in (1, 2)
            or self.audio_format != "pcm_s16le"
        ):
            raise ConfigurationError(
                f"{role}: declare supported PCM format/rate/channels; conversion belongs in the adapter"
            )
        if any(
            not isinstance(v, str) or len(v) > 500
            for v in (self.protocol, self.endpoint, self.auth_env, self.model)
        ):
            raise ConfigurationError(f"{role}: invalid backend metadata")
        if self.backend == "mock":
            if (
                self.allow_private_http
                or self.voice
                or self.language
                or self.instructions
                or self.seed is not None
            ):
                raise ConfigurationError(
                    f"{role}: mock backend does not use private transport/voice/language/seed"
                )
            if self.allow_live or self.request_limit not in (0, None):
                raise ConfigurationError(f"{role}: mock backend cannot authorize live requests")
            if runnable and (
                self.sample_rate != 16000
                or self.channels != 1
                or self.concurrency != 1
                or self.max_audio_seconds != 30
                or not self.streaming
            ):
                raise ConfigurationError(
                    f"{role}: current mock runtime requires mono PCM16 16000 Hz, concurrency 1, streaming true and 30 second limit"
                )
            if any((self.endpoint, self.auth_env, self.model, self.protocol)):
                raise ConfigurationError(
                    f"{role}: mock backend does not use endpoint/auth/model/protocol fields"
                )
        else:
            try:
                url = urlsplit(self.endpoint)
                valid = (
                    url.scheme in {"https", "http", "wss", "ws"}
                    and url.hostname
                    and not url.username
                    and not url.password
                    and not url.query
                    and not url.fragment
                    and (url.port is None or 1 <= url.port <= 65535)
                )
            except ValueError:
                valid = False
            if (
                not valid
                or not self.model.strip()
                or not self.protocol.strip()
                or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", self.auth_env)
            ):
                raise ConfigurationError(
                    f"{role}: supply endpoint without credentials/query, documented protocol, model and auth_env variable name"
                )
            if url.hostname == "openrouter.ai" and (
                role != "llm"
                or self.protocol != "chat-completions-v1"
                or self.endpoint != "https://openrouter.ai/api/v1/chat/completions"
                or not (self.model.endswith(":free") or self.model == "openrouter/free")
            ):
                raise ConfigurationError(
                    "OpenRouter requires its exact HTTPS chat endpoint and an explicit :free model"
                )
            if runnable:
                contracts = {
                    "asr": {
                        "deepgram-listen-v1",
                        "groq-whisper-v1",
                        "vllm-qwen-audio-v1",
                        "vllm-qwen3-asr-v1",
                    },
                    "tts": {"deepgram-speak-v1", "vllm-qwen-tts-v1"},
                    "llm": {"chat-completions-v1"},
                }
                if self.protocol not in contracts.get(role, set()):
                    raise ConfigurationError(
                        f"{role}: external adapter is not implemented for this documented protocol; no request was made"
                    )
                if (
                    role == "llm"
                    and self.allow_live
                    and (not 1 <= self.max_input_tokens <= self.max_input_tokens_total <= 1_000_000)
                ):
                    raise ConfigurationError(
                        "llm: live chat requires positive per-request and aggregate input-token ceilings"
                    )
                if role != "llm" and (self.max_input_tokens or self.max_input_tokens_total):
                    raise ConfigurationError(
                        f"{role}: input-token ceilings are only supported for chat-completions"
                    )
                dgx_paths = {
                    "asr": {
                        (8000, "/v1/chat/completions", "vllm-qwen-audio-v1"),
                        (8000, "/v1/audio/transcriptions", "vllm-qwen3-asr-v1"),
                    },
                    "tts": {(8001, "/v1/audio/speech", "vllm-qwen-tts-v1")},
                    "llm": {
                        (8000, "/v1/chat/completions", "chat-completions-v1"),
                        (8001, "/v1/chat/completions", "chat-completions-v1"),
                    },
                }
                approved_http = (
                    self.allow_private_http
                    and role in dgx_paths
                    and url.scheme == "http"
                    and (url.hostname, url.port)
                    in {
                        ("10.153.60.233", 8000),
                        ("10.153.60.233", 8001),
                        ("202.92.159.240", 8000),
                        ("202.92.159.240", 8001),
                    }
                    and (url.port, url.path, self.protocol) in dgx_paths[role]
                )
                if self.allow_private_http and not approved_http:
                    raise ConfigurationError(
                        f"{role}: private HTTP exception is restricted to the supplied service endpoints"
                    )
                if url.scheme not in {"https", "wss"} and not approved_http:
                    raise ConfigurationError(f"{role}: live adapters require TLS endpoints")
                http_contract = role == "llm" or self.protocol in {
                    "groq-whisper-v1",
                    "vllm-qwen-audio-v1",
                    "vllm-qwen3-asr-v1",
                    "vllm-qwen-tts-v1",
                }
                if http_contract != (url.scheme in {"https", "http"}):
                    raise ConfigurationError(
                        f"{role}: select HTTPS for chat/file-ASR or WSS for streaming speech"
                    )
                if (
                    self.sample_rate != 16000
                    or self.channels != 1
                    or self.concurrency != 1
                    or self.streaming
                    != (
                        self.protocol
                        not in {
                            "groq-whisper-v1",
                            "vllm-qwen-audio-v1",
                            "vllm-qwen3-asr-v1",
                        }
                    )
                    or (
                        self.max_audio_seconds != 30
                        and not (role == "asr" and self.protocol == "deepgram-listen-v1")
                    )
                ):
                    raise ConfigurationError(
                        f"{role}: require mono PCM16 16000 Hz, concurrency 1 and supported audio limit; file/Qwen-audio ASR is final-only (streaming=false)"
                    )
                if self.protocol == "vllm-qwen-tts-v1":
                    if not self.voice.strip() or not self.language.strip():
                        raise ConfigurationError(
                            "tts: configure a documented built-in voice and language"
                        )
                elif self.voice or self.language or self.instructions:
                    raise ConfigurationError(
                        f"{role}: voice/language/instructions are only supported by the Qwen TTS adapter"
                    )
            if self.seed is not None and (role != "tts" or self.protocol != "vllm-qwen-tts-v1"):
                raise ConfigurationError(f"{role}: seed is only supported by the Qwen TTS adapter")
        return self


def load_backends(path: Path | None = None, *, runnable=True):
    if path is None:
        return {role: BackendSpec() for role in ("asr", "llm", "tts")}
    try:
        if path.stat().st_size > 16384:
            raise ConfigurationError("backend configuration exceeds 16 KiB")
        raw = json.loads(path.read_text())
        if set(raw) != {"asr", "llm", "tts"}:
            raise ValueError
        result = {
            role: BackendSpec(**values).validate(role, runnable=runnable)
            for role, values in raw.items()
        }
    except ConfigurationError:
        raise
    except OSError:
        raise ConfigurationError(
            "backend configuration is not readable; supply an existing local JSON file, or omit --backend-config for mocks"
        ) from None
    except (TypeError, ValueError, AttributeError):
        raise ConfigurationError(
            "configuration requires asr/llm/tts objects with documented fields; credentials are environment-only"
        ) from None
    return result


@dataclass(frozen=True)
class RuntimeOptions:
    """Application controller selection; independent of inference provider config."""

    engine: str = "legacy"
    silence_seconds: float = 0.5
    research: ResearchOptions = field(default_factory=lambda: ResearchOptions())
    max_model_calls_per_turn: int = 6
    max_tool_calls_per_turn: int = 8
    max_repeated_tool_calls: int = 2


def load_runtime(path=None, *, engine=None):
    values = {}
    if path is not None:
        path = Path(path)
        try:
            if path.stat().st_size > 4096:
                raise ConfigurationError("runtime configuration exceeds4KiB")
            values = json.loads(path.read_text())
        except (OSError, ValueError):
            raise ConfigurationError("invalid runtime configuration") from None
        if not isinstance(values, dict) or set(values) - {
            "engine",
            "silence_seconds",
            "research",
            "max_model_calls_per_turn",
            "max_tool_calls_per_turn",
            "max_repeated_tool_calls",
        }:
            raise ConfigurationError(
                "runtime accepts engine, silence_seconds, research and bounded turn safeguards"
            )
    if engine is not None:
        values["engine"] = engine
    try:
        if "research" in values:
            values["research"] = ResearchOptions(**values["research"])
        result = RuntimeOptions(**values)
        if (
            result.engine not in {"legacy", "native"}
            or type(result.silence_seconds) not in (int, float)
            or not math.isfinite(result.silence_seconds)
            or not 0 <= result.silence_seconds <= 3
            or type(result.max_model_calls_per_turn) is not int
            or not 1 <= result.max_model_calls_per_turn <= 12
            or type(result.max_tool_calls_per_turn) is not int
            or not 1 <= result.max_tool_calls_per_turn <= 8
            or type(result.max_repeated_tool_calls) is not int
            or not 1 <= result.max_repeated_tool_calls <= 4
        ):
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise ConfigurationError(
            "invalid engine, silence_seconds0..3, research permissions/budget or turn safeguards"
        ) from None
