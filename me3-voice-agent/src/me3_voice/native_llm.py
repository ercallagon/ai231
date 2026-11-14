"""Native framework request/tool loop; existing mock upstream and live budget.

No Session/ChatReasoning loop. Cloud endpoint/model comes unchanged from the
selected BackendSpec. SDK and framework retries are disabled, not hidden calls.
"""

import asyncio
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
import re
import time
from urllib.parse import urlsplit
import uuid
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.chat import ChatCompletionChunk
from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from .providers import MockLLM, ToolCall
from .remote import LiveTransport, ProviderError
from .conversation_context import prepare_context
from .request_budget import RequestBudgetError, conservative_input_token_estimate
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class NativeLLM(OpenAILLMService):
    def __init__(
        self,
        spec,
        *,
        credentials=None,
        budget=None,
        upstream=None,
        on_event=None,
        clock=None,
        source_ids=None,
        research_remaining=None,
        max_model_calls_per_turn=6,
        max_tool_calls_per_turn=8,
        max_repeated_tool_calls=2,
        max_retrieval_failures_per_turn=2,
        fast_action=None,
        refresh_tools=None,
    ):
        self.epoch = self.calls = self.tool_calls = 0
        self.fast_action_checked_epoch = None
        self.dispatch_epoch = ContextVar("me3_tool_dispatch_epoch", default=None)
        self.response_epoch = 0
        self.response_has_tools = False
        self.response_tool_indices = set()
        self.clock = clock
        self.max_model_calls_per_turn = max_model_calls_per_turn
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.max_retrieval_failures_per_turn = max_retrieval_failures_per_turn
        self.tool_operation_counts = {}
        self.research_tool_calls = 0
        self.retrieval_failures = 0
        self.response_failure_reason = None
        self._last_request_started = 0.0
        self._last_prompt_tokens = 0
        self._token_limit = 0
        self._remaining_tokens = None
        self._token_reset_seconds = None
        self._rate_headers_at = 0.0
        self.source_ids = source_ids or (lambda: set())
        self.research_remaining = research_remaining or (lambda: None)
        self.turn_deadline = None
        self.turn_seconds = 110.0
        self.answer_reserve_seconds = 12.0
        self.spec = spec
        self.upstream = upstream or MockLLM()
        self.budget = budget or LiveTransport(spec, credentials)
        self.on_event = on_event or (lambda *_args, **_kw: None)
        self.fast_action = fast_action
        self.refresh_tools = refresh_tools
        token = "offline-not-a-provider-key"
        base_url = None
        if spec.backend == "external":
            spec.validate("llm", runnable=True)
            # Validate credentials without reserving a request; reservation
            # occurs only when the actual provider request is dispatched.
            token = (credentials or {}).get(spec.auth_env) or os.environ.get(spec.auth_env, "")
            if not token:
                from .configuration import ConfigurationError

                raise ConfigurationError("configured project credential is missing")
            if not spec.endpoint.endswith("/chat/completions"):
                from .configuration import ConfigurationError

                raise ConfigurationError("native LLM requires a standard chat/completions endpoint")
            base_url = spec.endpoint.removesuffix("/chat/completions")
        groq = urlsplit(spec.endpoint).hostname == "api.groq.com"
        extra = (
            {
                "reasoning_effort": "low",
                "parallel_tool_calls": False,
                "extra_body": {"include_reasoning": False},
            }
            if groq and spec.model == "openai/gpt-oss-120b"
            else {}
        )
        super().__init__(
            api_key=token,
            base_url=base_url,
            settings=self.Settings(
                model=spec.model or "mock",
                # Eight approved GPT-OSS requests at 2,000 completion tokens
                # each cannot exceed the campaign's 16k aggregate ceiling.
                max_completion_tokens=(
                    2000 if groq and spec.model == "openai/gpt-oss-120b" else 2048
                ),
                extra=extra,
            ),
            retry_on_timeout=False,
        )

    def create_client(self, api_key=None, base_url=None, **_kwargs):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=self.spec.timeout_seconds,
            http_client=DefaultAsyncHttpxClient(trust_env=False, timeout=self.spec.timeout_seconds),
        )

    def new_turn(self):
        self.epoch += 1
        self.calls = self.tool_calls = 0
        self.tool_operation_counts.clear()
        self.research_tool_calls = 0
        self.retrieval_failures = 0
        self.response_failure_reason = None
        self.turn_deadline = time.monotonic() + self.turn_seconds

    def claim_tool(self):
        if self.tool_calls >= self.max_tool_calls_per_turn:
            raise ValueError("tool_budget")
        self.tool_calls += 1

    def record_retrieval_result(self, *, failed, error_code=None):
        if failed:
            self.retrieval_failures += 1
            self.on_event(
                "research_retrieval_failure",
                error_code=error_code or "retrieval_failed",
                failure_count=self.retrieval_failures,
                failure_limit=self.max_retrieval_failures_per_turn,
            )

    def claim_tool_operation(self, name, arguments):
        canonical = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        operation_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        count = self.tool_operation_counts.get(operation_hash, 0) + 1
        self.tool_operation_counts[operation_hash] = count
        if name in {"web_search", "retrieve_source"}:
            self.research_tool_calls += 1
        if self.tool_calls >= self.max_tool_calls_per_turn:
            return operation_hash, count, "tool_budget_exhausted"
        self.tool_calls += 1
        if (
            self.turn_deadline
            and time.monotonic() >= self.turn_deadline - self.answer_reserve_seconds
        ):
            return operation_hash, count, "turn_deadline_reached"
        if name == "web_search" and not self.source_ids():
            remaining = self.research_remaining()
            if remaining is not None and remaining <= 1:
                return operation_hash, count, "retrieval_budget_reserved"
        if count > self.max_repeated_tool_calls:
            return operation_hash, count, "repeated_tool_operation_blocked"
        return operation_hash, count, None

    async def run_function_calls(self, function_calls):
        if self.response_tool_indices and len(function_calls) != len(self.response_tool_indices):
            raise ProviderError(
                "invalid tool argument JSON rejected", code="invalid_tool_arguments"
            )
        if self.response_epoch != self.epoch:
            return
        if function_calls:
            self.response_has_tools = True
        # Framework-created tasks inherit immutable dispatch context, even when
        # their handlers start after an interruption/new user turn.
        token = self.dispatch_epoch.set(self.response_epoch)
        try:
            await super().run_function_calls(function_calls)
        finally:
            self.dispatch_epoch.reset(token)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self.response_epoch = self.epoch
                self.response_has_tools = False
                self.response_tool_indices.clear()
                self.response_failure_reason = None
            elif isinstance(frame, LLMFullResponseEndFrame):
                frame.metadata["me3_response_epoch"] = self.response_epoch
                frame.metadata["me3_response_has_tools"] = self.response_has_tools
                frame.metadata["me3_failure_reason"] = self.response_failure_reason
        await super().push_frame(frame, direction)

    async def process_frame(self, frame, direction):
        if isinstance(frame, InterruptionFrame):
            self.epoch += 1
        await super().process_frame(frame, direction)

    async def get_chat_completions(self, context):
        if self.fast_action is not None and self.fast_action_checked_epoch != self.epoch:
            # One parse attempt per user turn. Tool continuation requests in
            # the same Pipecat turn must never replay an already-run action.
            self.fast_action_checked_epoch = self.epoch
            epoch = self.epoch
            started = time.monotonic()
            answer = await self.fast_action(context)
            if answer is not None:
                self.on_event("fast_action_answer", response_chars=len(answer))
                return self.observe(self._local_chunks(answer), epoch, started)
        if self.retrieval_failures >= self.max_retrieval_failures_per_turn:
            return self._terminal_response("research_retrieval_failures_exhausted")
        if (
            self.turn_deadline
            and time.monotonic() >= self.turn_deadline - self.answer_reserve_seconds
        ):
            return self._terminal_response("turn_deadline_reached")
        if self.calls >= self.max_model_calls_per_turn:
            return self._terminal_response("model_call_budget_exhausted")
        if self.refresh_tools is not None:
            self.refresh_tools(context)
        dropped, instruction = prepare_context(context, clock=self.clock)
        self.on_event("llm_context_ready", **self.context_diagnostics(context, dropped))
        await self._update_settings(self.Settings(system_instruction=instruction))
        self.calls += 1
        epoch = self.epoch
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        self.on_event("route_selected", route="llm", request_id=request_id)
        self.on_event("llm_request", context_groups_dropped=dropped, request_id=request_id)
        if self.spec.backend == "external":
            try:
                params = await self._prepare_provider_params(context)
                estimated_tokens, serialized_bytes = conservative_input_token_estimate(params)
            except (RequestBudgetError, TypeError, ValueError):
                self.on_event("llm_input_budget_unavailable")
                return self._terminal_response("input_token_budget_unavailable")
            self.on_event(
                "llm_input_budget_estimated",
                estimated_input_tokens=estimated_tokens,
                serialized_request_bytes=serialized_bytes,
                request_limit=self.spec.max_input_tokens,
                campaign_limit=self.spec.max_input_tokens_total,
            )
            provider_attempts = 1 if os.environ.get("ME3_DISABLE_PROVIDER_RETRIES") == "1" else 2
            for attempt in range(provider_attempts):
                delay = self.provider_request_delay()
                if (
                    self.turn_deadline
                    and time.monotonic() + delay + self.answer_reserve_seconds >= self.turn_deadline
                ):
                    return self._terminal_response("pacing_exceeds_turn_deadline")
                await self._pace_provider_request()
                if (
                    self.turn_deadline
                    and time.monotonic() >= self.turn_deadline - self.answer_reserve_seconds
                ):
                    return self._terminal_response("turn_deadline_reached")
                try:
                    _, request_id = self.budget.authorize(
                        input_tokens=estimated_tokens,
                        input_token_limit=self.spec.max_input_tokens,
                        input_tokens_total_limit=self.spec.max_input_tokens_total,
                    )
                except ProviderError as exc:
                    if exc.code in {
                        "input_token_budget_exhausted",
                        "input_token_request_limit_reached",
                        "campaign_cap_increase_rejected",
                        "campaign_caps_unregistered",
                    }:
                        self.on_event(
                            "llm_input_budget_rejected",
                            estimated_input_tokens=estimated_tokens,
                            campaign_limit=self.spec.max_input_tokens_total,
                            reason=exc.code,
                        )
                        return self._terminal_response("input_token_budget_exhausted")
                    raise
                self._last_request_started = time.monotonic()
                try:
                    stream = await self._client.chat.completions.create(**params)
                    self._observe_rate_headers(getattr(stream, "response", None))
                    stream = self._track_ledger_stream(stream, request_id)
                    break
                except Exception as exc:
                    error = self.safe_error(exc)
                    self.budget.finish_request(request_id, "failure", error.code)
                    self.on_event(
                        "llm_provider_error",
                        code=error.code,
                        rate_limit=error.rate_limit,
                    )
                    retry_after = (error.rate_limit or {}).get("retry_after_seconds")
                    if (
                        attempt + 1 < provider_attempts
                        and error.code == "provider_rate_limited"
                        and isinstance(retry_after, (int, float))
                        and 0 <= retry_after <= 30
                        and (
                            self.spec.request_limit is None
                            or self.budget.requests < self.spec.request_limit
                        )
                    ):
                        await asyncio.sleep(retry_after)
                        continue
                    raise error from None
        else:
            stream = self.mock_chunks(context)
        return self.observe(self._bounded_stream(stream), epoch, started, request_id=request_id)

    async def _prepare_provider_params(self, context):
        """Build the same request parameters used by Pipecat before reserving.

        Pipecat owns context-to-provider formatting. Reuse its adapter and
        parameter builder, then pass this exact object to the SDK after the
        durable request/token reservation succeeds.
        """
        self._log_llm_response(context)
        adapter = self.get_llm_adapter()
        params_from_context = await adapter.get_llm_invocation_params(
            context,
            system_instruction=self._settings.system_instruction,
            convert_developer_to_user=not self.supports_developer_role,
        )
        params = self.build_chat_completion_params(params_from_context)
        if not isinstance(params, dict):
            raise RequestBudgetError("framework request parameters are unavailable")
        return params

    def _log_llm_response(self, context):
        """Keep Pipecat's request log useful without dumping user conversation text."""
        try:
            message_count = len(context.get_messages())
        except (AttributeError, TypeError):
            message_count = 0
        logger.debug("{}: Generating LLM response ({} context messages)", self, message_count)

    async def _track_ledger_stream(self, stream, request_id):
        """Finalize only after complete consumption; cancellation stays unknown."""
        try:
            async for chunk in stream:
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = self.safe_error(exc)
            self.budget.finish_request(request_id, "failure", error.code)
            raise
        else:
            self.budget.finish_request(request_id, "success")

    @staticmethod
    def _local_chunks(answer):
        async def chunks():
            yield ChatCompletionChunk(
                id="me3-local-action",
                object="chat.completion.chunk",
                created=0,
                model="me3-local-action",
                choices=[{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
            )
            yield ChatCompletionChunk(
                id="me3-local-action",
                object="chat.completion.chunk",
                created=0,
                model="me3-local-action",
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )

        return chunks()

    def _terminal_response(self, reason):
        self.response_failure_reason = reason
        self.on_event("llm_guardrail_termination", reason=reason)
        return self.observe(self.terminal_failure_chunks(), self.epoch, time.monotonic())

    async def _bounded_stream(self, stream):
        iterator = stream.__aiter__()
        while True:
            try:
                if self.turn_deadline:
                    remaining = self.turn_deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    chunk = await asyncio.wait_for(anext(iterator), remaining)
                else:
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                self.response_failure_reason = "turn_deadline_reached"
                self.on_event("llm_guardrail_termination", reason=self.response_failure_reason)
                async for failure in self.terminal_failure_chunks():
                    yield failure
                return
            yield chunk

    def context_diagnostics(self, context, dropped):
        tool_calls = tool_results = 0
        tool_errors = {}
        for message in context.get_messages():
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                tool_calls += len(message.get("tool_calls") or [])
            elif message.get("role") == "tool":
                tool_results += 1
                try:
                    result = json.loads(message.get("content") or "null")
                except (TypeError, ValueError):
                    continue
                if isinstance(result, dict) and result.get("ok") is False:
                    code = result.get("error")
                    if isinstance(code, str):
                        tool_errors[code] = tool_errors.get(code, 0) + 1
        return {
            "dropped_groups": dropped,
            "message_count": len(context.get_messages()),
            "assistant_tool_call_count": tool_calls,
            "tool_result_count": tool_results,
            "tool_error_counts": tool_errors,
            "retained_source_count": len(self.source_ids()),
        }

    @staticmethod
    def terminal_failure_chunks():
        message = "I couldn't complete that request reliably, so I can't give a confident answer."

        async def chunks():
            yield ChatCompletionChunk(
                id="me3-terminal-failure",
                object="chat.completion.chunk",
                created=0,
                model="me3-terminal",
                choices=[{"index": 0, "delta": {"content": message}, "finish_reason": "stop"}],
            )

        return chunks()

    @staticmethod
    def safe_error(exc):
        from openai import APITimeoutError, APIStatusError

        if isinstance(exc, ProviderError):
            return exc
        if isinstance(exc, (APITimeoutError, TimeoutError)):
            return ProviderError("reasoning service timed out", code="provider_timeout")
        if isinstance(exc, APIStatusError):
            rate_limit = NativeLLM.safe_rate_limit_headers(exc.response.headers)
            return LiveTransport.status_error(exc.status_code, "llm", rate_limit=rate_limit)
        return ProviderError("reasoning stream failed; details excluded", code="provider_error")

    @staticmethod
    def safe_rate_limit_headers(headers):
        """Keep only documented, non-secret quota headers and parsed retry guidance."""
        allowed = {
            "retry-after",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        }
        selected = {
            str(key).lower(): str(value)[:64]
            for key, value in headers.items()
            if str(key).lower() in allowed and re.fullmatch(r"[0-9A-Za-z.:+-]{1,64}", str(value))
        }
        result = {"headers": selected}
        retry = selected.get("retry-after")
        if retry:
            seconds = NativeLLM._parse_retry_seconds(retry)
            if seconds is not None and 0 <= seconds <= 86400:
                result["retry_after_seconds"] = seconds
        remaining_tokens = selected.get("x-ratelimit-remaining-tokens")
        remaining_requests = selected.get("x-ratelimit-remaining-requests")
        if remaining_tokens == "0":
            result["likely_limit"] = "token_per_minute_or_input_output_token_limit"
            seconds = NativeLLM._parse_retry_seconds(selected.get("x-ratelimit-reset-tokens", ""))
            if seconds is not None:
                result.setdefault("retry_after_seconds", seconds)
        elif remaining_requests == "0":
            result["likely_limit"] = "requests_per_day"
            seconds = NativeLLM._parse_retry_seconds(selected.get("x-ratelimit-reset-requests", ""))
            if seconds is not None:
                result.setdefault("retry_after_seconds", seconds)
        elif result.get("retry_after_seconds") is not None:
            result["likely_limit"] = "other_short_window_or_account_limit"
        return result

    @staticmethod
    def _parse_retry_seconds(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                if "," in value:
                    return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
                parts = re.findall(r"(\d+(?:\.\d+)?)([hms])", value)
                if not parts or "".join(number + unit for number, unit in parts) != value:
                    return None
                scale = {"h": 3600, "m": 60, "s": 1}
                return sum(float(number) * scale[unit] for number, unit in parts)
            except (TypeError, ValueError, OverflowError):
                return None

    def _observe_rate_headers(self, response):
        headers = getattr(response, "headers", {}) or {}
        safe = self.safe_rate_limit_headers(headers)
        try:
            self._token_limit = int(safe["headers"].get("x-ratelimit-limit-tokens", 0))
        except (TypeError, ValueError):
            self._token_limit = 0
        try:
            self._remaining_tokens = int(safe["headers"]["x-ratelimit-remaining-tokens"])
        except (KeyError, TypeError, ValueError):
            self._remaining_tokens = None
        self._token_reset_seconds = self._parse_retry_seconds(
            safe["headers"].get("x-ratelimit-reset-tokens", "")
        )
        self._rate_headers_at = time.monotonic()
        if safe["headers"]:
            self.on_event("llm_rate_limit_headers", rate_limit=safe)

    def provider_request_delay(self, now=None):
        """Return a conservative delay from the latest provider quota headers."""
        now = time.monotonic() if now is None else now
        delay = max(0.0, 2.1 - (now - self._last_request_started))
        if self._token_limit and self._last_prompt_tokens:
            expected = self._last_prompt_tokens * 1.10
            if self._remaining_tokens is not None:
                if self._remaining_tokens < expected and self._token_reset_seconds is not None:
                    delay = max(
                        delay,
                        self._token_reset_seconds - (now - self._rate_headers_at),
                    )
            else:
                # Headers may be absent on an intermediary or older endpoint.
                delay = max(delay, 60.0 * expected / self._token_limit)
        return max(0.0, delay)

    async def _pace_provider_request(self):
        """Keep below request minimum and wait for token reset when headers require it."""
        delay = self.provider_request_delay()
        if delay:
            await asyncio.sleep(delay)

    async def observe(self, stream, epoch, started, *, request_id=None):
        first = True
        total_chars = 0
        completed = False
        named_indices = set()
        tail = ""
        research_text = []
        research_citations = set()
        invalid_research_citation = False
        research_finalized = False
        validate_research_answer = bool(
            self.research_tool_calls and not self.response_failure_reason
        )
        try:
            async for chunk in stream:
                if epoch != self.epoch:
                    self.on_event("llm_stale_rejected")
                    return
                if chunk.usage:
                    self._last_prompt_tokens = max(0, chunk.usage.prompt_tokens or 0)
                    self.on_event(
                        "llm_usage",
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                    )
                for choice in chunk.choices:
                    if choice.finish_reason:
                        completed = True
                    if choice.finish_reason == "length":
                        raise ProviderError(
                            "response token budget exhausted", code="response_budget_exhausted"
                        )
                    is_research_final = (
                        choice.finish_reason == "stop"
                        and not self.response_tool_indices
                        and validate_research_answer
                    )
                    if research_finalized and is_research_final:
                        # Drain duplicate terminal choices without rendering or
                        # speaking the already-validated research answer again.
                        choice.delta.content = ""
                        self.on_event("llm_duplicate_finalization_suppressed")
                        continue
                    total_chars += len(choice.delta.content or "")
                    for call in choice.delta.tool_calls or []:
                        self.response_has_tools = True
                        self.response_tool_indices.add(call.index)
                        if len(self.response_tool_indices) > 8:
                            raise ProviderError("too many tool calls", code="tool_budget_exhausted")
                        if call.function:
                            if call.function.name:
                                named_indices.add(call.index)
                            total_chars += len(call.function.arguments or "") + len(
                                call.function.name or ""
                            )
                    if total_chars > 32000:
                        raise ProviderError(
                            "response exceeds character budget", code="response_budget_exhausted"
                        )
                    text = choice.delta.content
                    if text:
                        text = tail + text
                        tail = ""
                        # Buffer only a possible split citation, never whole sentences.
                        bracket = text.rfind("[")
                        if (
                            bracket >= 0
                            and re.fullmatch(r"\[S?\d*", text[bracket:])
                            and len(text[bracket:]) < 24
                        ):
                            text, tail = text[:bracket], text[bracket:]
                        for source in re.findall(r"\[(S\d+)\]", text):
                            if source not in self.source_ids():
                                if validate_research_answer:
                                    invalid_research_citation = True
                                else:
                                    raise ProviderError(
                                        "unknown citation rejected", code="unsupported_source"
                                    )
                            else:
                                research_citations.add(source)
                        if validate_research_answer:
                            research_text.append(text)
                            text = ""
                        choice.delta.content = text
                    if (
                        validate_research_answer
                        and choice.finish_reason == "stop"
                        and not self.response_tool_indices
                    ):
                        research_finalized = True
                        if invalid_research_citation:
                            self.response_failure_reason = "unsupported_source"
                        elif not research_citations or not self.source_ids():
                            self.response_failure_reason = "insufficient_verified_evidence"
                        if self.response_failure_reason:
                            choice.delta.content = "I couldn't verify this answer from a retrieved source, so I won't guess."
                            self.on_event(
                                "llm_guardrail_termination",
                                reason=self.response_failure_reason,
                                retained_source_count=len(self.source_ids()),
                            )
                        else:
                            choice.delta.content = "".join(research_text)
                            self.on_event(
                                "llm_grounding_check",
                                citation_count=len(research_citations),
                                retained_source_count=len(self.source_ids()),
                            )
                    if first and (choice.delta.content or choice.delta.tool_calls):
                        first = False
                        self.on_event(
                            "llm_first_token",
                            seconds=time.monotonic() - started,
                            request_id=request_id,
                        )
                if tail and any(c.finish_reason for c in chunk.choices):
                    raise ProviderError("incomplete citation rejected", code="unsupported_source")
                yield chunk
            if self.response_tool_indices - named_indices:
                raise ProviderError("tool name missing", code="invalid_tool_arguments")
            if first or not completed:
                raise ProviderError(
                    "empty or incomplete reasoning stream", code="provider_incomplete_response"
                )
            if tail:
                raise ProviderError("incomplete citation rejected", code="unsupported_source")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self.safe_error(exc) from None
        finally:
            if hasattr(stream, "aclose"):
                await stream.aclose()
            elif hasattr(stream, "close"):
                await stream.close()

    def mock_chunks(self, context):
        async def chunks():
            async for item in self.upstream.respond(context.get_messages()):
                if isinstance(item, ToolCall):
                    delta = {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": item.call_id + "-" + uuid.uuid4().hex,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": json.dumps(item.arguments),
                                },
                            }
                        ]
                    }
                else:
                    delta = {"content": item}
                yield ChatCompletionChunk(
                    id="mock",
                    object="chat.completion.chunk",
                    created=0,
                    model="mock",
                    choices=[{"index": 0, "delta": delta, "finish_reason": None}],
                )
            yield ChatCompletionChunk(
                id="mock",
                object="chat.completion.chunk",
                created=0,
                model="mock",
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )

        return chunks()
