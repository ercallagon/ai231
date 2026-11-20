"""Native framework request/tool loop; existing mock upstream and live budget.

No Session/ChatReasoning loop. Cloud endpoint/model comes unchanged from the
selected BackendSpec. SDK and framework retries are disabled, not hidden calls.
"""

import asyncio
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import replace
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import parse_qsl, urlsplit
import uuid
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.chat import ChatCompletionChunk
from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from .providers import MockLLM, ToolCall
from .remote import LiveTransport, ProviderError
from .conversation_context import prepare_context, runtime_time
from .conversation_tools import validate_arguments
from .request_budget import RequestBudgetError, conservative_input_token_estimate
from .research import public_url


from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection


MAX_RESEARCH_SEARCHES_PER_TURN = 3
MAX_RESEARCH_RETRIEVALS_PER_TURN = 5
MAX_RESEARCH_EVIDENCE_TOKENS_PER_TURN = 5000
MAX_MODEL_CALLS_PER_TURN = 7
_RESEARCH_CLAIM_STOP_WORDS = {
    "according",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_RESEARCH_HIGH_RISK_QUERY = re.compile(
    r"\b(?:medical|medicine|medication|drug|dose|dosage|diagnos\w*|treat\w*|"
    r"symptom\w*|prescription\w*|legal|lawyer|lawsuit|liable|liability|"
    r"guilty|court|contract|financial|investment|invest\w*|stock|tax|loan|"
    r"mortgage|credit|safety|hazard|emergency|poison)\b",
    re.IGNORECASE,
)
_RESEARCH_NEGATION = re.compile(
    r"\b(?:not|never|neither|nor|without|absent|absence|denied|refused|"
    r"failed|cannot|can't|couldn't|didn't|doesn't|isn't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_RESEARCH_NUMBER = re.compile(r"\b\d+(?:[,.]\d+)*(?:%|st|nd|rd|th)?\b", re.IGNORECASE)
_RESEARCH_ENTITY = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ELECTION_EVENT = re.compile(r"\b(?:elections?|races?|primar(?:y|ies)|runoffs?)\b", re.I)
_ELECTION_OFFICE = re.compile(r"\b(?:senate|house|governor|presiden\w*|mayor)\b", re.I)
_ELECTION_WIN = re.compile(
    r"\b(?:won|wins?|was the winner of|is the winner of|defeats?|defeated|beat|beats)\b",
    re.I,
)
_ELECTION_DEFEAT = re.compile(r"\b(?:defeat(?:s|ed|ing)?|beat(?:s|ing)?)\b", re.I)
_RESEARCH_NON_ENTITY = {
    "a",
    "an",
    "and",
    "after",
    "according",
    "another",
    "as",
    "at",
    "before",
    "by",
    "during",
    "for",
    "from",
    "in",
    "into",
    "it",
    "its",
    "later",
    "near",
    "of",
    "on",
    "or",
    "over",
    "since",
    "than",
    "the",
    "their",
    "then",
    "this",
    "those",
    "through",
    "to",
    "under",
    "until",
    "upon",
    "with",
    "within",
    "without",
    "prime",
    "minister",
    "president",
    "senator",
    "senate",
    "governor",
    "secretary",
    "official",
    "office",
    "government",
    "meeting",
    "report",
    "result",
    "results",
    "election",
    "elections",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}


def _research_entities(text):
    return {
        token.casefold()
        for token in _RESEARCH_ENTITY.findall(text)
        if token.casefold() not in _RESEARCH_NON_ENTITY
    }


def _research_content_terms(text):
    return {
        word.casefold()
        for word in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        if word.casefold() not in _RESEARCH_CLAIM_STOP_WORDS
    }


def _research_sentences(passage):
    # Keep an office name such as U.S. Senate in the result sentence.
    return re.split(r"(?<!\b[A-Z]\.[A-Z]\.)(?<=[.!?])\s+|\n+", passage)


def _relevant_research_sentences(claim, passage):
    """Return the passage sentences most directly tied to a claim's wording."""
    claim_terms = _research_content_terms(claim)
    claim_entities = _research_entities(claim)
    scored = []
    for sentence in _research_sentences(passage):
        shared_terms = len(claim_terms & _research_content_terms(sentence))
        shared_entities = len(claim_entities & _research_entities(sentence))
        if shared_terms or shared_entities:
            scored.append(((shared_entities, shared_terms), sentence))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return [sentence for score, sentence in scored if score == best]


def _election_result_supported(claim, passage):
    """Require a positive election result to anchor its participants and race together."""
    win = _ELECTION_WIN.search(claim)
    if (
        not win
        or not _ELECTION_EVENT.search(claim)
        or (not _ELECTION_OFFICE.search(claim) and not re.search(r"\belections?\b", claim, re.I))
    ):
        return True

    def names(text):
        return _research_entities(text) - {"democrat", "democratic", "republican", "party"}

    def subject(text):
        participants = names(text)
        return [
            word.casefold()
            for word in _RESEARCH_ENTITY.findall(text)
            if word.casefold() in participants
        ]

    def opponent(text):
        defeat = _ELECTION_DEFEAT.search(text)
        return (
            names(text[defeat.end() :].split(" in ", 1)[0].split(" for ", 1)[0])
            if defeat
            else set()
        )

    winner = subject(claim[: win.start()])
    loser = opponent(claim)
    entities = names(claim)
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", claim))
    offices = set(_ELECTION_OFFICE.findall(claim.casefold()))
    election_round = set(re.findall(r"\b(?:primary|runoff|general)\b", claim.casefold()))
    for sentence in _research_sentences(passage):
        result = _ELECTION_WIN.search(sentence)
        if (
            not result
            or not _ELECTION_EVENT.search(sentence)
            or _RESEARCH_NEGATION.search(sentence)
        ):
            continue
        if (
            winner
            and winner == subject(sentence[: result.start()])[-len(winner) :]
            and loser <= opponent(sentence)
            and entities <= names(sentence)
            and years <= set(re.findall(r"\b(?:19|20)\d{2}\b", sentence))
            and offices <= set(_ELECTION_OFFICE.findall(sentence.casefold()))
            and election_round
            <= set(re.findall(r"\b(?:primary|runoff|general)\b", sentence.casefold()))
        ):
            return True
    return False


def _research_claim_conflicts(claim, passage, question=""):
    """Flag only explicit lexical conflicts; this is not semantic entailment."""
    conflicts = []
    claim_entities = _research_entities(claim)
    passage_entities = _research_entities(passage)
    question_entities = _research_entities(question)
    unsupported_entities = claim_entities - passage_entities - question_entities
    if passage_entities and unsupported_entities and not (claim_entities & passage_entities):
        conflicts.append("entity_anchor_conflict")

    # A retrieved passage can contain several unrelated facts. Compare dates
    # and polarity only against the sentence(s) most relevant to this claim;
    # a stray date or negation elsewhere is not evidence of contradiction.
    relevant = _relevant_research_sentences(claim, passage)
    relevant_text = " ".join(relevant)
    claim_numbers = {value.casefold() for value in _RESEARCH_NUMBER.findall(claim)}
    passage_numbers = {value.casefold() for value in _RESEARCH_NUMBER.findall(relevant_text)}

    # Years, ceremony ordinals and quantities are different anchors. A year
    # absent from an ordinal-only sentence is not a contradictory year.
    def number_kind(value):
        if re.fullmatch(r"\d+(?:st|nd|rd|th)", value):
            return "ordinal"
        if re.fullmatch(r"(?:19|20)\d{2}", value):
            return "year"
        if value.endswith("%"):
            return "percent"
        return "quantity"

    for kind in {number_kind(value) for value in claim_numbers}:
        claimed = {value for value in claim_numbers if number_kind(value) == kind}
        retained = {value for value in passage_numbers if number_kind(value) == kind}
        if retained and not claimed <= retained:
            conflicts.append("numeric_anchor_conflict")
            break

    claim_negated = bool(_RESEARCH_NEGATION.search(claim))
    relevant_polarities = {bool(_RESEARCH_NEGATION.search(value)) for value in relevant}
    if len(relevant_polarities) == 1 and claim_negated not in relevant_polarities:
        conflicts.append("polarity_conflict")
    return conflicts


RESEARCH_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "passage_id": {"type": "string"},
                },
                "required": ["text", "passage_id"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


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
        timezone_name="UTC",
        source_ids=None,
        source_lookup=None,
        evidence_passage_ids=None,
        evidence_passage_lookup=None,
        research_remaining=None,
        max_model_calls_per_turn=MAX_MODEL_CALLS_PER_TURN,
        max_tool_calls_per_turn=8,
        max_repeated_tool_calls=2,
        fast_action=None,
        refresh_tools=None,
        research_alternatives=None,
        tavily_quick_research=None,
    ):
        self.epoch = self.calls = self.tool_calls = 0
        self.turn_number = 0
        self.fast_action_checked_epoch = None
        self.dispatch_epoch = ContextVar("me3_tool_dispatch_epoch", default=None)
        self.response_epoch = 0
        self.response_has_tools = False
        self.response_tool_indices = set()
        self.clock = clock
        self.timezone_name = timezone_name
        self.max_model_calls_per_turn = min(max_model_calls_per_turn, MAX_MODEL_CALLS_PER_TURN)
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.tool_operation_counts = {}
        self.research_tool_calls = 0
        self.research_searches = 0
        self.automatic_research_result = None
        self.research_retrievals = 0
        self.research_evidence_texts = []
        self.research_evidence_tokens = 0
        self.retrieval_failures = 0
        self.response_failure_reason = None
        self.pending_tavily_answer = None
        self.tavily_quick_research_epoch = None
        self.tavily_quick_research = tavily_quick_research
        self.research_format_repair_attempted = False
        self._research_evidence_repair_active = False
        self._research_repair_contexts = set()
        self._research_repair_tool_rounds = {}
        self._rejected_research_passage = None
        self._repair_excluded_passages = set()
        self._last_request_started = 0.0
        self._last_prompt_tokens = 0
        self._token_limit = 0
        self._remaining_tokens = None
        self._token_reset_seconds = None
        self._rate_headers_at = 0.0
        self.source_ids = source_ids or (lambda: set())
        self.source_lookup = source_lookup or (lambda _source_id: None)
        self.evidence_passage_ids = evidence_passage_ids or (lambda: set())
        self.evidence_passage_lookup = evidence_passage_lookup or (lambda _passage_id: None)
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
        self.research_alternatives = research_alternatives or (lambda: [])
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
        endpoint_host = urlsplit(spec.endpoint).hostname
        groq = endpoint_host == "api.groq.com"
        local_qwen = endpoint_host in {
            "localhost",
            "127.0.0.1",
            "::1",
        } and spec.model.casefold().startswith("qwen3:")
        extra = (
            {
                "reasoning_effort": "low",
                "parallel_tool_calls": False,
                "extra_body": {"include_reasoning": False},
            }
            if groq and spec.model == "openai/gpt-oss-120b"
            else {"extra_body": {"think": False}}
            if local_qwen
            else {}
        )
        super().__init__(
            api_key=token,
            base_url=base_url,
            settings=self.Settings(
                model=spec.model or "mock",
                # Bound completion work per call; session request allowances
                # are configured independently in the selected profile.
                max_completion_tokens=(
                    2000
                    if groq and spec.model == "openai/gpt-oss-120b"
                    else 512
                    if local_qwen
                    else 2048
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
        self.turn_number += 1
        self.calls = self.tool_calls = 0
        self.pending_tavily_answer = None
        self.tavily_quick_research_epoch = None
        self.tool_operation_counts.clear()
        self.research_tool_calls = 0
        self.research_searches = 0
        self.automatic_research_result = None
        self.research_retrievals = 0
        self.research_evidence_texts.clear()
        self.research_evidence_tokens = 0
        self.retrieval_failures = 0
        self.response_failure_reason = None
        self.pending_tavily_answer = None
        self.research_format_repair_attempted = False
        self._research_evidence_repair_active = False
        self._research_repair_contexts.clear()
        self._research_repair_tool_rounds.clear()
        self._rejected_research_passage = None
        self._repair_excluded_passages.clear()
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
                failure_limit=MAX_RESEARCH_RETRIEVALS_PER_TURN,
            )

    def _finish_research_evidence_repair(self, failure_code=None):
        if not self._research_evidence_repair_active:
            return
        self._research_evidence_repair_active = False
        self.on_event(
            "llm_research_evidence_repair_finished",
            completed=failure_code is None,
            failure_code=failure_code,
        )

    def reserve_alternate_retrieval(self):
        """Count a deterministic source failover against the turn retrieval cap."""
        if self.research_retrievals >= MAX_RESEARCH_RETRIEVALS_PER_TURN:
            return False
        self.research_retrievals += 1
        self.on_event(
            "research_turn_budget_reserved",
            operation="retrieve_source_alternate",
            searches=self.research_searches,
            search_limit=MAX_RESEARCH_SEARCHES_PER_TURN,
            retrievals=self.research_retrievals,
            retrieval_limit=MAX_RESEARCH_RETRIEVALS_PER_TURN,
            evidence_tokens=self.research_evidence_tokens,
            evidence_token_limit=MAX_RESEARCH_EVIDENCE_TOKENS_PER_TURN,
        )
        return True

    def claim_tool_operation(self, name, arguments):
        if name == "web_search" and isinstance(arguments, dict):
            arguments = dict(arguments)
            query = arguments.get("query")
            if isinstance(query, str):
                arguments["query"] = " ".join(query.casefold().split())
            arguments.setdefault("max_results", 5)
        canonical = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        operation_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        count = self.tool_operation_counts.get(operation_hash, 0) + 1
        self.tool_operation_counts[operation_hash] = count
        if name == "retrieve_source" and self._research_evidence_repair_active:
            repair = next(iter(self._research_repair_tool_rounds.values()), None)
            if repair and (
                not isinstance(arguments, dict) or arguments.get("url") != repair["url"]
            ):
                return operation_hash, count, "research_repair_source_not_selected"
        if name in {"web_search", "retrieve_source"}:
            self.research_tool_calls += 1
        if name == "web_search" and self.research_searches >= MAX_RESEARCH_SEARCHES_PER_TURN:
            return operation_hash, count, "research_search_budget_exhausted"
        if (
            name == "retrieve_source"
            and self.research_retrievals >= MAX_RESEARCH_RETRIEVALS_PER_TURN
        ):
            return operation_hash, count, "research_retrieval_budget_exhausted"
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
        # Reserve before network I/O; failed/uncertain requests consume capacity.
        if name == "web_search":
            self.research_searches += 1
        elif name == "retrieve_source":
            self.research_retrievals += 1
        if name in {"web_search", "retrieve_source"}:
            self.on_event(
                "research_turn_budget_reserved",
                operation=name,
                searches=self.research_searches,
                search_limit=MAX_RESEARCH_SEARCHES_PER_TURN,
                retrievals=self.research_retrievals,
                retrieval_limit=MAX_RESEARCH_RETRIEVALS_PER_TURN,
                evidence_tokens=self.research_evidence_tokens,
                evidence_token_limit=MAX_RESEARCH_EVIDENCE_TOKENS_PER_TURN,
            )
        return operation_hash, count, None

    def evidence_for_model(self, research, source, query):
        """Project relevant text under the cumulative turn cap.

        Full source text remains in ResearchClient for citation verification.
        """
        if self.research_evidence_tokens >= MAX_RESEARCH_EVIDENCE_TOKENS_PER_TURN:
            raise ValueError("research_evidence_budget_exhausted")

        def estimate(text):
            # Evidence-only byte upper bound plus 25% headroom. Whole provider
            # requests retain their independent serialized-request accounting.
            return math.ceil(len(text.encode("utf-8")) * 1.25)

        low, high, best = 1, 2400, None
        while low <= high:
            limit = (low + high) // 2
            candidate = research.evidence_for_model(source, query, max_chars=limit)
            text = candidate.get("text", "") if isinstance(candidate, dict) else ""
            needed = estimate("\n".join([*self.research_evidence_texts, text]))
            if text and needed <= MAX_RESEARCH_EVIDENCE_TOKENS_PER_TURN:
                best = (candidate, text, needed)
                low = limit + 1
            else:
                high = limit - 1
        if best is None:
            raise ValueError("research_evidence_budget_exhausted")
        candidate, text, needed = best
        self.research_evidence_texts.append(text)
        self.research_evidence_tokens = needed
        self.on_event(
            "research_evidence_projected",
            source_id=candidate.get("source_id"),
            source_type=candidate.get("kind"),
            extraction_method=candidate.get("extraction_method"),
            evidence_status=candidate.get("evidence_status"),
            evidence_characters=len(text),
            estimated_evidence_tokens=needed,
            remaining_evidence_tokens=MAX_RESEARCH_EVIDENCE_TOKENS_PER_TURN - needed,
        )
        return candidate

    def citation_sources(self, source_ids):
        """Resolve accepted citation IDs to retained, validated source records."""
        resolved = []
        valid_ids = {
            value for value in source_ids if isinstance(value, str) and re.fullmatch(r"S\d+", value)
        }
        for source_id in sorted(valid_ids, key=lambda value: int(value[1:])):
            source = self.source_lookup(source_id)
            if not isinstance(source, dict) or source.get("source_id") != source_id:
                continue
            # Search-result snippets are discovery hints, not fetched evidence
            # suitable for a user-facing research citation.
            if source.get("fetched") is not True:
                continue
            try:
                url = public_url(source.get("url"))
            except (TypeError, ValueError):
                continue
            # URLs with credential-like query parameters are useful internally
            # for fetching, but must not be copied into browser-visible traces.
            try:
                query_keys = [key for key, _ in parse_qsl(urlsplit(url).query, max_num_fields=50)]
            except ValueError:
                continue
            if any(
                re.search(
                    r"(?:key|token|secret|auth|signature|session|password|credential)", key, re.I
                )
                for key in query_keys
            ):
                continue
            resolved.append(
                {
                    "source_id": source_id,
                    "title": str(source.get("title", ""))[:250],
                    "url": url,
                    "evidence_status": str(source.get("evidence_status", ""))[:48],
                    "source_type": str(source.get("kind", ""))[:48],
                    "fetched": True,
                    "truncated": source.get("truncated") is True,
                }
            )
        return resolved

    def project_search_snippets(self, research, result, query, *, limit=2):
        """Project only short, source-owned Tavily snippets under the evidence cap."""
        if not isinstance(result, dict) or result.get("kind") != "untrusted_search_leads":
            return result
        projected = []
        evidence_count = 0
        for lead in result.get("results", []):
            if not isinstance(lead, dict):
                continue
            source = research.retain_search_snippet(lead) if evidence_count < limit else None
            if source is None:
                discovery = dict(lead)
                if isinstance(discovery.get("snippet"), str):
                    discovery["snippet"] = discovery["snippet"][:160]
                projected.append(discovery)
                continue
            try:
                evidence = self.evidence_for_model(research, source, query)
            except ValueError:
                research.discard_search_snippet(source["source_id"])
                projected.append(dict(lead))
                continue
            evidence["evidence_status"] = "search_result_snippet"
            evidence["source_type"] = "search_result"
            evidence["fetched"] = False
            projected.append(evidence)
            evidence_count += 1
        return {**result, "results": projected}

    async def run_function_calls(self, function_calls):
        if self.response_tool_indices and len(function_calls) != len(self.response_tool_indices):
            raise ProviderError(
                "invalid tool argument JSON rejected", code="invalid_tool_arguments"
            )
        if self.response_epoch != self.epoch:
            return
        if function_calls:
            self.response_has_tools = True
        if not self.research_searches:
            function_calls = [self._ground_yearless_senate_search(call) for call in function_calls]
        # Framework-created tasks inherit immutable dispatch context, even when
        # their handlers start after an interruption/new user turn.
        token = self.dispatch_epoch.set(self.response_epoch)
        try:
            await super().run_function_calls(function_calls)
        finally:
            self.dispatch_epoch.reset(token)

    def _ground_yearless_senate_search(self, call):
        """Keep an unrequested year out of a narrowly scoped election lookup."""
        if call.function_name != "web_search":
            return call
        question = self._latest_user_question(call.context).strip()
        match = re.fullmatch(
            r"Who won the (?:U\.S\. |US |United States )?Senate election in "
            r"([A-Za-z]+(?:[ -][A-Za-z]+){0,3})[?.!]?",
            question,
            flags=re.I,
        )
        if not match or re.search(
            r"\b(?:in|on|during|for|after|before|last|previous|nineteen|twenty|thousand)\b",
            match[1],
            re.I,
        ):
            return call
        try:
            args = validate_arguments("web_search", call.arguments)
            day = runtime_time(self.timezone_name, self.clock)["local"][:10]
        except (TypeError, ValueError):
            return call
        query = f"latest completed U.S. Senate election result in {match[1]} as of {day}"
        return replace(call, arguments={**args, "query": query})

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

    def queue_tavily_answer(self, answer, source_id=None):
        """Schedule one provider-generated answer or a terminal safe abstention."""
        failure = None
        if isinstance(answer, str) and answer.strip() and re.fullmatch(r"S\d+", source_id or ""):
            text = answer.strip()
            # A qualifier can be a separate sentence. Keep the whole bounded
            # answer or abstain when it cannot fit the quick-answer limit.
            if len(text.split()) <= 35:
                answer = f"{text} [{source_id}]"
            else:
                answer = None
        else:
            answer = None
        if answer is None:
            answer = "I couldn't verify this answer from a retrieved source, so I won't guess."
            failure = "tavily_answer_unavailable"
        self.pending_tavily_answer = (self.epoch, answer, failure)
        return failure is None

    def designate_tavily_quick_research(self):
        """Route this already-identified user turn directly to Tavily."""
        self.tavily_quick_research_epoch = self.epoch

    async def get_chat_completions(self, context):
        acceptance = getattr(self, "pending_task_acceptance", None)
        self.pending_task_acceptance = None
        if acceptance is not None and acceptance[0] == self.epoch:
            self.on_event("task_acceptance_answer")
            return self.observe(
                self._local_chunks(
                    "I've started the task in the background. You can keep talking to me."
                ),
                self.epoch,
                time.monotonic(),
                owner_turn_id=self.turn_number,
            )
        pending_tavily = self.pending_tavily_answer
        if pending_tavily is not None:
            self.pending_tavily_answer = None
            epoch, answer, failure = pending_tavily
            if epoch == self.epoch:
                self.response_failure_reason = failure
                started = time.monotonic()
                self.on_event("route_selected", route="tavily_answer")
                self.on_event(
                    "tavily_answer_stream_started",
                    response_characters=len(answer),
                    available=failure is None,
                )
                return self.observe(
                    self._local_chunks(answer),
                    epoch,
                    started,
                    owner_turn_id=self.turn_number,
                    direct_research_answer=True,
                )
        if self.fast_action is not None and self.fast_action_checked_epoch != self.epoch:
            # One parse attempt per user turn. Tool continuation requests in
            # the same Pipecat turn must never replay an already-run action.
            self.fast_action_checked_epoch = self.epoch
            epoch = self.epoch
            started = time.monotonic()
            answer = await self.fast_action(context)
            if answer is not None:
                self.tavily_quick_research_epoch = None
                self.on_event("fast_action_answer", response_chars=len(answer))
                return self.observe(
                    self._local_chunks(answer),
                    epoch,
                    started,
                    owner_turn_id=self.turn_number,
                )
        # Experiment opt-in; fast actions retain precedence. Never redirect
        # task requests, contextual pronouns or multi-source analysis.
        question = self._latest_user_question(context).strip()
        if (
            os.environ.get("ME3_AUTOMATIC_QUICK_RESEARCH") == "1"
            and self.fast_action_checked_epoch == self.epoch
            and not self.response_has_tools
            and self.calls == 0
            and self.tavily_quick_research is not None
            and re.match(
                r"(?i)^(?:(?:please )?(?:look up|search for) )?(?:who won|which movie won|who is|when was)\b",
                question,
            )
            and not re.search(r"(?i)\b(that|it|they|compare|build|create|task|sources)\b", question)
        ):
            self.designate_tavily_quick_research()
        if self.tavily_quick_research_epoch == self.epoch:
            self.tavily_quick_research_epoch = None
            if (
                self.turn_deadline
                and time.monotonic() >= self.turn_deadline - self.answer_reserve_seconds
            ):
                return self._terminal_response("turn_deadline_reached")
            epoch = self.epoch
            started = time.monotonic()
            self.on_event("route_selected", route="tavily_quick_research")

            async def direct_answer_chunks():
                question = self._latest_user_question(context)
                answer, source_id = None, None
                self.on_event(
                    "tavily_quick_research_started",
                    question_characters=len(question),
                )
                if self.tavily_quick_research is not None:
                    try:
                        answer, source_id = await self.tavily_quick_research(question, epoch)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.on_event(
                            "tavily_quick_research_failed",
                            error_code=getattr(exc, "code", "research_unavailable"),
                        )
                if epoch != self.epoch:
                    return
                self.queue_tavily_answer(answer, source_id)
                pending = self.pending_tavily_answer
                self.pending_tavily_answer = None
                _, rendered, failure = pending
                self.response_failure_reason = failure
                citation_url = None
                if failure is None and source_id and self.research is not None:
                    source = self.research.sources.get(source_id, {})
                    citation_url = source.get("url") if isinstance(source, dict) else None
                self.on_event(
                    "tavily_quick_research_finished",
                    available=failure is None,
                    response_characters=len(rendered),
                    source_id=source_id if failure is None else None,
                    source_url=citation_url,
                )
                self.on_event(
                    "tavily_answer_stream_started",
                    response_characters=len(rendered),
                    available=failure is None,
                )
                async for chunk in self._local_chunks(rendered):
                    yield chunk

            return self.observe(
                direct_answer_chunks(),
                epoch,
                started,
                owner_turn_id=self.turn_number,
                direct_research_answer=True,
            )
        if self.retrieval_failures >= MAX_RESEARCH_RETRIEVALS_PER_TURN:
            return self._terminal_response("research_retrieval_failures_exhausted")
        if (
            self.turn_deadline
            and time.monotonic() >= self.turn_deadline - self.answer_reserve_seconds
        ):
            return self._terminal_response("turn_deadline_reached")
        if self.calls >= self.max_model_calls_per_turn:
            return self._terminal_response("model_call_budget_exhausted")
        automatic_result = self.automatic_research_result
        self.automatic_research_result = None
        auto_fetched = automatic_result == (self.epoch, True)
        if automatic_result == (self.epoch, False):
            return self._terminal_response("insufficient_verified_evidence")
        repair_context_id = id(context)
        if repair_context_id not in self._research_repair_contexts:
            # Pipecat may deliver the repair tool result in a different context
            # object. The one active repair still owns that continuation.
            repair_context_id = next(
                (
                    key
                    for key, state in self._research_repair_tool_rounds.items()
                    if state["retrieval_used"]
                ),
                repair_context_id,
            )
        repair_context = repair_context_id in self._research_repair_contexts
        repair_tool_round = self._research_repair_tool_rounds.get(repair_context_id)
        if repair_tool_round and repair_tool_round.get("retrieval_used"):
            from pipecat.adapters.schemas.tools_schema import ToolsSchema
            from pipecat.processors.aggregators.llm_context import LLMContext

            context = LLMContext(
                messages=context.get_messages(), tools=ToolsSchema(standard_tools=[])
            )
            context.set_tool_choice("none")
            if not self._structured_research_passage_ids():
                self._research_repair_tool_rounds.pop(repair_context_id, None)
                self._research_repair_contexts.discard(repair_context_id)
                self._finish_research_evidence_repair("insufficient_verified_evidence")
                return self._terminal_response("insufficient_verified_evidence")
        if self.refresh_tools is not None and not repair_context:
            self.refresh_tools(context)
        dropped, instruction, request_context = prepare_context(
            context, clock=self.clock, timezone_name=self.timezone_name
        )
        if auto_fetched:
            request_context.set_tool_choice("none")
        # The first post-search call should fetch a lead while there is
        # room for both retrieval and a grounded final.
        if (
            not repair_context
            and self._supports_structured_research_output()
            and self.research_searches
            and not self.research_retrievals
            and self.tool_calls < self.max_tool_calls_per_turn
            and self.calls + 2 <= self.max_model_calls_per_turn
            and self.research_alternatives()
        ):
            from pipecat.adapters.schemas.tools_schema import ToolsSchema

            retrieve_tools = [
                tool
                for tool in getattr(request_context.tools, "standard_tools", ())
                if tool.name == "retrieve_source"
            ]
            if retrieve_tools:
                request_context.set_tools(ToolsSchema(standard_tools=retrieve_tools))
                request_context.set_tool_choice("required")
                self.on_event("llm_research_retrieval_required")
        self.on_event("llm_context_ready", **self.context_diagnostics(request_context, dropped))
        await self._update_settings(self.Settings(system_instruction=instruction))
        self.calls += 1
        epoch = self.epoch
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        self.on_event("route_selected", route="llm", request_id=request_id)
        self.on_event("llm_request", context_groups_dropped=dropped, request_id=request_id)
        research_question = self._latest_user_question(request_context)
        structured_research = self._uses_structured_research_output() and (
            request_context.tool_choice != "required"
        )
        strict_research_support = structured_research and self._requires_strict_research_support(
            request_context
        )
        if self.spec.backend == "external":
            try:
                (
                    params,
                    estimated_tokens,
                    serialized_bytes,
                ) = await self._prepare_provider_params_within_budget(request_context)
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
                self.on_event("llm_provider_request_started", request_id=request_id)
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
        validate_research_answer = bool(
            self.research_tool_calls
            and not self.response_failure_reason
            and not structured_research
        )
        observed = self.observe(
            self._bounded_stream(stream),
            epoch,
            started,
            request_id=request_id,
            structured_research=structured_research,
            research_question=research_question,
            strict_research_support=strict_research_support,
            owner_turn_id=self.turn_number,
            research_repair_context_id=(repair_context_id if repair_context else None),
        )
        if repair_context or not (structured_research or validate_research_answer):
            return observed
        return self._research_answer_with_one_format_repair(
            observed,
            epoch=epoch,
            started=started,
            research_question=research_question,
            strict_research_support=strict_research_support,
            request_context=request_context,
            structured_research=structured_research,
        )

    def _research_answer_with_one_format_repair(
        self,
        first_stream,
        *,
        epoch,
        started,
        research_question,
        strict_research_support,
        request_context,
        structured_research=True,
    ):
        """Correct one missing-citation/format failure without weakening evidence checks.

        The failed structured payload is never spoken or added to canonical
        history. A correction uses the same retained evidence, has no tools,
        and goes through the ordinary request ledger and model-call budget.
        """

        async def chunks():
            held_failure = []
            repairable_failures = {"invalid_structured_answer", "missing_research_citation"}
            evidence_failures = {
                "contradictory_claim",
                "unsupported_claim",
                "insufficient_verified_evidence",
            }
            if not structured_research:
                repairable_failures = {"missing_research_citation"}
            else:
                repairable_failures.update(evidence_failures)
            async for item in first_stream:
                if self.response_failure_reason in repairable_failures:
                    # Research text is withheld until validation; discard its
                    # failure fallback while trying the single bounded correction.
                    held_failure.append(item)
                    continue
                yield item

            if not held_failure:
                return
            repair_reason = self.response_failure_reason
            evidence_failure = structured_research and repair_reason in evidence_failures
            if evidence_failure and self._rejected_research_passage:
                self._repair_excluded_passages.add(self._rejected_research_passage)
            rejected_source = (
                self.source_lookup(self._rejected_research_passage.split("-")[0])
                if evidence_failure and self._rejected_research_passage
                else None
            )
            rejected_url = rejected_source.get("url") if isinstance(rejected_source, dict) else None
            alternatives = [
                lead for lead in self.research_alternatives() if lead["url"] != rejected_url
            ]
            evidence_recovery = (
                evidence_failure
                and bool(alternatives)
                and self.tool_calls < self.max_tool_calls_per_turn
                and self.research_retrievals < MAX_RESEARCH_RETRIEVALS_PER_TURN
            )
            required_calls = 2 if evidence_recovery else 1
            can_repair = (
                not self.research_format_repair_attempted
                and epoch == self.epoch
                and (
                    not evidence_failure
                    or evidence_recovery
                    or self._structured_research_passage_ids()
                )
                and self.calls + required_calls <= self.max_model_calls_per_turn
                and (
                    self.turn_deadline is None
                    or time.monotonic() < self.turn_deadline - self.answer_reserve_seconds
                )
            )
            if can_repair:
                from pipecat.processors.aggregators.llm_context import LLMContext
                from pipecat.adapters.schemas.tools_schema import ToolsSchema

                self.research_format_repair_attempted = True
                self.response_failure_reason = None
                messages = deepcopy(request_context.get_messages())
                if evidence_recovery:
                    self._research_evidence_repair_active = True
                    tools = getattr(request_context.tools, "standard_tools", ())
                    retrieve_tools = [
                        tool for tool in tools if getattr(tool, "name", None) == "retrieve_source"
                    ]
                    if not retrieve_tools:
                        self.response_failure_reason = repair_reason
                        self._finish_research_evidence_repair(repair_reason)
                        for item in held_failure:
                            yield item
                        return
                    correction = (
                        "The verifier found that the retained passages do not support your draft. "
                        f"Use exactly this different, untried URL with retrieve_source: {alternatives[0]['url']}. "
                        "Do not choose a different URL. "
                        "Do not search again or reuse a retained/failed URL. Search snippets "
                        "are discovery only and cannot support a final answer. "
                        "After the tool result, give one concise grounded "
                        "answer in the required JSON schema using only a retained passage ID. If no "
                        "available source supports the answer, return an empty claims array."
                    )
                    self.on_event(
                        "llm_research_evidence_repair_started",
                        reason=repair_reason,
                        retrievals_available=len(alternatives),
                        remaining_model_calls=max(0, self.max_model_calls_per_turn - self.calls),
                    )
                else:
                    correction = (
                        "The draft was rejected. Use only other retained passages that support "
                        "the answer and return the required JSON. Do not call tools. "
                        "If no eligible passage supports the answer, return an empty claims array."
                        if evidence_failure
                        else "Your previous research final did not match the required structured "
                        "answer format. Make one format correction using only the already "
                        "retained evidence in this conversation. Return only the required "
                        "JSON schema, with one or two concise claims and their exact retained "
                        "passage IDs. Do not call tools or add unsupported facts. If the "
                        "evidence does not support an answer, return an empty claims array."
                        if structured_research
                        else "Your previous answer omitted a verifiable source citation. "
                        "Using only the retained evidence already in this conversation, "
                        "write one concise answer with an exact citation such as [S1]. "
                        "Do not call tools, invent a source, or add unsupported facts. "
                        "If the evidence does not support an answer, say so briefly."
                    )
                    self.on_event(
                        (
                            "llm_structured_answer_repair_started"
                            if structured_research
                            else "llm_citation_answer_repair_started"
                        ),
                        reason=repair_reason,
                        remaining_model_calls=max(0, self.max_model_calls_per_turn - self.calls),
                    )
                messages.append(
                    {
                        "role": "developer",
                        "content": correction,
                    }
                )
                repair_context = LLMContext(
                    messages=messages,
                    tools=(ToolsSchema(standard_tools=retrieve_tools) if evidence_recovery else []),
                    # The correction has only one eligible operation. Require it
                    # so the model cannot spend the reserved recovery turn on
                    # another unsupported synthesis from the same passage.
                    tool_choice=("required" if evidence_recovery else "none"),
                )
                self._research_repair_contexts.add(id(repair_context))
                if evidence_recovery:
                    self._research_repair_tool_rounds[id(repair_context)] = {
                        "retrieval_used": False,
                        "url": alternatives[0]["url"],
                    }
                try:
                    repair_stream = await self.get_chat_completions(repair_context)
                except Exception:
                    self.response_failure_reason = "invalid_structured_answer"
                    repair_stream = None
                finally:
                    if not evidence_recovery:
                        self._research_repair_contexts.discard(id(repair_context))
                repaired = False
                repair_tool_requested = False
                if repair_stream is not None and epoch == self.epoch:
                    async for item in repair_stream:
                        choices = getattr(item, "choices", None) or []
                        delta = getattr(choices[0], "delta", None) if choices else None
                        repaired = repaired or bool(getattr(delta, "content", None))
                        repair_tool_requested = repair_tool_requested or bool(
                            getattr(delta, "tool_calls", None)
                        )
                        yield item
                elif repair_stream is None:
                    for item in held_failure:
                        yield item
                if not (evidence_recovery and repair_tool_requested):
                    self.on_event(
                        (
                            "llm_structured_answer_repair_finished"
                            if structured_research
                            else "llm_citation_answer_repair_finished"
                        ),
                        completed=repaired and self.response_failure_reason is None,
                        failure_code=self.response_failure_reason,
                    )
                return

            # No spare request allowance, deadline, or live turn: preserve the
            # original safe abstention rather than consuming outside the cap.
            for item in held_failure:
                yield item

        return chunks()

    async def _prepare_provider_params_within_budget(self, context):
        """Fit the actual framework request to its configured hard input ceiling.

        Omit discovery text for sources fetched within the same turn. Under
        budget pressure, drop old turn groups before compacting tool results.
        """
        from pipecat.processors.aggregators.llm_context import LLMContext

        # This second defensive copy makes the helper safe for direct callers
        # and tests too. No request compaction can mutate a live Pipecat history.
        context = LLMContext(
            messages=deepcopy(context.get_messages()),
            tools=context.tools,
            tool_choice=context.tool_choice,
        )
        self._omit_fetched_lead_snippets(context)
        params = await self._prepare_provider_params(context)
        estimated, size = conservative_input_token_estimate(params)
        limit = self.spec.max_input_tokens
        initial_estimate = estimated
        dropped_groups = 0
        compacted_search_results = 0
        if not limit or estimated <= limit:
            return params, estimated, size

        while estimated > limit and self._drop_oldest_context_group(context):
            dropped_groups += 1
            params = await self._prepare_provider_params(context)
            estimated, size = conservative_input_token_estimate(params)

        for max_results, snippet_chars in (
            (3, 120),
            (3, 60),
            (2, 60),
            (1, 0),
        ):
            if estimated <= limit:
                break
            if self._compact_search_discovery(context, max_results, snippet_chars):
                compacted_search_results += 1
                params = await self._prepare_provider_params(context)
                estimated, size = conservative_input_token_estimate(params)

        for evidence_chars in (1600, 1000, 600, 300):
            if estimated <= limit:
                break
            if self._compact_retrieved_evidence(context, evidence_chars):
                compacted_search_results += 1
                params = await self._prepare_provider_params(context)
                estimated, size = conservative_input_token_estimate(params)

        if dropped_groups or compacted_search_results:
            self.on_event(
                "llm_context_compacted",
                reason="per_request_input_budget",
                groups_dropped=dropped_groups,
                context_compaction_steps=compacted_search_results,
                estimated_tokens_before=initial_estimate,
                estimated_tokens_after=estimated,
                request_limit=limit,
            )
        return params, estimated, size

    @staticmethod
    def _omit_fetched_lead_snippets(context):
        """Omit search-only text once same-turn fetched evidence is available."""
        fetched_urls = set()
        for message in reversed(context.get_messages()):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                fetched_urls.clear()
                continue
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, str):
                try:
                    result = json.loads(content)
                except ValueError:
                    continue
            else:
                result = content
            if not isinstance(result, dict):
                continue
            kind = result.get("kind")
            if (
                kind == "untrusted_retrieved_source"
                and result.get("fetched") is True
                and isinstance(result.get("url"), str)
                and result.get("passage_ids")
                and isinstance(result.get("text"), str)
                and result["text"].strip()
            ):
                fetched_urls.add(result["url"])
            elif kind == "untrusted_search_leads" and isinstance(result.get("results"), list):
                changed = False
                for lead in result["results"]:
                    if (
                        isinstance(lead, dict)
                        and isinstance(lead.get("url"), str)
                        and lead["url"] in fetched_urls
                    ):
                        status = lead.get("evidence_status")
                        if status == "discovery_only" and "snippet" in lead:
                            del lead["snippet"]
                            changed = True
                        elif status == "search_result_snippet" and lead.get("fetched") is False:
                            for key in ("text", "passage_ids", "citation", "evidence_chunks"):
                                if key in lead:
                                    del lead[key]
                                    changed = True
                if changed and isinstance(content, str):
                    message["content"] = json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _drop_oldest_context_group(context):
        messages = deepcopy(context.get_messages())
        groups = []
        prefix = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                groups.append([])
            if groups:
                groups[-1].append(message)
            else:
                prefix.append(message)
        if len(groups) <= 1:
            return False
        context.set_messages(prefix + [message for group in groups[1:] for message in group])
        return True

    @staticmethod
    def _compact_search_discovery(context, max_results, snippet_chars):
        messages = deepcopy(context.get_messages())
        changed = False
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = message.get("content")
            was_json = isinstance(content, str)
            if was_json:
                try:
                    result = json.loads(content)
                except (TypeError, ValueError):
                    continue
            elif isinstance(content, dict):
                result = content
            else:
                continue
            if not isinstance(result, dict) or result.get("kind") != "untrusted_search_leads":
                continue
            rows = result.get("results")
            if not isinstance(rows, list):
                continue
            compacted = []
            for row in rows[:max_results]:
                if not isinstance(row, dict) or not isinstance(row.get("url"), str):
                    continue
                lead = {
                    key: row[key]
                    for key in (
                        "url",
                        "title",
                        "snippet",
                        "evidence_status",
                        "published_date_reported",
                        "ranked_for_query",
                    )
                    if key in row
                }
                if isinstance(lead.get("title"), str):
                    lead["title"] = lead["title"][:160]
                if isinstance(lead.get("snippet"), str):
                    lead["snippet"] = lead["snippet"][:snippet_chars]
                compacted.append(lead)
            if compacted == rows:
                continue
            result = {**result, "results": compacted}
            message["content"] = (
                json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                if was_json
                else result
            )
            changed = True
        if changed:
            context.set_messages(messages)
        return changed

    @staticmethod
    def _compact_retrieved_evidence(context, max_chars):
        """Shorten model-visible excerpts, retaining evidence identity/provenance."""
        messages = deepcopy(context.get_messages())
        changed = False
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = message.get("content")
            was_json = isinstance(content, str)
            if was_json:
                try:
                    result = json.loads(content)
                except (TypeError, ValueError):
                    continue
            elif isinstance(content, dict):
                result = content
            else:
                continue
            text = result.get("text") if isinstance(result, dict) else None
            if not (
                isinstance(result, dict)
                and isinstance(result.get("source_id"), str)
                and isinstance(result.get("url"), str)
                and isinstance(text, str)
                and len(text) > max_chars
            ):
                continue
            excerpt = text[:max_chars]
            boundary = max(excerpt.rfind(mark) for mark in (". ", "? ", "! "))
            if boundary >= max_chars // 2:
                excerpt = excerpt[: boundary + 1]
            message["content"] = (
                json.dumps({**result, "text": excerpt}, ensure_ascii=False, separators=(",", ":"))
                if was_json
                else {**result, "text": excerpt}
            )
            changed = True
        if changed:
            context.set_messages(messages)
        return changed

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
        if self._uses_structured_research_output() and context.tool_choice != "required":
            passage_ids = self._structured_research_passage_ids()
            schema = deepcopy(RESEARCH_ANSWER_SCHEMA)
            schema["properties"]["claims"]["items"]["properties"]["passage_id"]["enum"] = (
                passage_ids
            )
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "swift_grounded_research_answer",
                    "strict": True,
                    "schema": schema,
                },
            }
            # OpenRouter's structured-output support varies by routed endpoint.
            # Require the selected provider to accept every request parameter.
            extra_body = dict(params.get("extra_body") or {})
            provider = dict(extra_body.get("provider") or {})
            provider["require_parameters"] = True
            extra_body["provider"] = provider
            params["extra_body"] = extra_body
        return params

    def _supports_structured_research_output(self):
        return (
            self.spec.backend == "external"
            and urlsplit(self.spec.endpoint).hostname == "openrouter.ai"
            and self.spec.model == "openai/gpt-oss-120b"
        )

    def _uses_structured_research_output(self):
        return (
            self._supports_structured_research_output()
            and self.research_tool_calls > 0
            and bool(self._structured_research_passage_ids() or self._repair_excluded_passages)
        )

    @staticmethod
    def _latest_user_question(context):
        try:
            messages = context.get_messages()
        except (AttributeError, TypeError):
            return ""
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ):
                return message["content"]
        return ""

    @classmethod
    def _requires_strict_research_support(cls, context):
        return bool(_RESEARCH_HIGH_RISK_QUERY.search(cls._latest_user_question(context)))

    def _structured_research_source_ids(self):
        """Return currently retained citation IDs in stable numeric order."""
        values = self.source_ids()
        if not isinstance(values, (set, list, tuple, frozenset)):
            return []
        valid = {
            value for value in values if isinstance(value, str) and re.fullmatch(r"S\d+", value)
        }
        return sorted(valid, key=lambda value: (int(value[1:]), value))

    def _structured_research_passage_ids(self):
        """Only expose projected passage IDs with a fetched, citable owner."""
        source_ids = {
            source["source_id"]
            for source in self.citation_sources(self._structured_research_source_ids())
        }
        values = self.evidence_passage_ids()
        if not isinstance(values, (set, list, tuple, frozenset)):
            return []
        valid = set()
        for value in values:
            if not isinstance(value, str):
                continue
            match = re.fullmatch(r"(S\d+)-P\d+", value)
            if (
                match
                and match.group(1) in source_ids
                and value not in self._repair_excluded_passages
            ):
                valid.add(value)
        return sorted(
            valid,
            key=lambda value: tuple(int(part) for part in re.findall(r"\d+", value)),
        )

    def _capture_unsupported_claim(self, *, draft, claim, passage, question, decision):
        if os.environ.get("ME3_CAPTURE_REJECTED_RESEARCH") != "1":
            return
        if len(draft) > 32000 or len(passage["text"]) > 8000 or len(question) > 4000:
            return
        directory = Path("var/results/research-rejections").absolute()
        try:
            if directory.resolve() != directory:
                return
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = directory.stat()
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                return
            run_id = os.environ.get("ME3_REQUEST_RUN_ID", "")
            payload = {
                "run_id": run_id if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) else None,
                "turn": self.turn_number,
                "recorded_monotonic": time.monotonic(),
                "question": question,
                "draft": draft,
                "claim": claim,
                "passage": passage,
                "source": next(iter(self.citation_sources([passage["source_id"]])), None),
                "decision": decision,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="unsupported-",
                suffix=".json",
                dir=directory,
                delete=False,
            ) as stream:
                json.dump(payload, stream, ensure_ascii=False)
                stream.write("\n")
        except (OSError, RuntimeError, UnicodeError):
            # Diagnostics are optional and must never affect the response.
            return

    def _validate_structured_research_answer(
        self, raw, *, question="", strict_support=False, diagnostic_draft=None
    ):
        """Resolve source-owned IDs, screen explicit conflicts, and report lexical fit."""
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return self._validate_cited_research_prose(
                raw, question=question, strict_support=strict_support
            )
        if not isinstance(payload, dict) or set(payload) != {"claims"}:
            return None, "invalid_structured_answer", [], False
        claims = payload.get("claims")
        # An explicit abstention follows the answer contract. It is terminal,
        # not a rejected draft that warrants another model or retrieval call.
        if isinstance(claims, list) and not claims:
            return None, "research_abstained", [], False
        if not isinstance(claims, list) or not 1 <= len(claims) <= 2:
            return None, "insufficient_verified_evidence", [], False

        available = self.source_ids()
        used = []
        rendered = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"text", "passage_id"}:
                return None, "invalid_structured_answer", used, False
            text = claim["text"].strip() if isinstance(claim["text"], str) else ""
            passage_id = claim["passage_id"]
            if not text or len(text) > 500:
                return None, "insufficient_verified_evidence", used, False
            if re.search(r"\[S\d+\]", text, flags=re.I):
                return None, "unsupported_source", used, True
            if not isinstance(passage_id, str) or not re.fullmatch(r"S\d+-P\d+", passage_id):
                return None, "unsupported_evidence_passage", used, True
            if passage_id not in self._structured_research_passage_ids():
                return None, "unsupported_evidence_passage", used, True
            passage = self.evidence_passage_lookup(passage_id)
            source_id = passage.get("source_id") if isinstance(passage, dict) else None
            source = self.source_lookup(source_id) if source_id in available else None
            original = (
                source.get("evidence_passages", {}).get(passage_id)
                if isinstance(source, dict) and isinstance(source.get("evidence_passages"), dict)
                else None
            )
            if (
                not isinstance(original, str)
                or not isinstance(passage, dict)
                or passage.get("text") != original
            ):
                failed_sources = used + ([source_id] if source_id in available else [])
                return None, "insufficient_verified_evidence", failed_sources, False
            claim_terms = _research_content_terms(text)
            passage_terms = _research_content_terms(original)
            matched_terms = claim_terms & passage_terms
            lexical_mismatch = not claim_terms <= passage_terms
            conflicts = _research_claim_conflicts(text, original, question)
            self.on_event(
                "llm_claim_support_diagnostic",
                source_id=source_id,
                passage_id=passage_id,
                lexical_mismatch=lexical_mismatch,
                claim_term_count=len(claim_terms),
                matched_term_count=len(matched_terms),
                strict_support_required=bool(strict_support),
                contradiction_codes=conflicts,
            )
            if not claim_terms:
                return None, "insufficient_verified_evidence", used + [source_id], False
            if conflicts:
                self._rejected_research_passage = passage_id
                return None, "contradictory_claim", used + [source_id], False
            # Ordinary research may use natural paraphrases. Lexical fit is
            # diagnostic there; high-risk queries retain the stricter existing
            # token-coverage rule. Neither branch claims semantic entailment.
            if (strict_support and lexical_mismatch) or not _election_result_supported(
                text, original
            ):
                self._rejected_research_passage = passage_id
                self._capture_unsupported_claim(
                    draft=raw if diagnostic_draft is None else diagnostic_draft,
                    claim=claim,
                    passage={"source_id": source_id, "passage_id": passage_id, "text": original},
                    question=question,
                    decision={
                        "code": "unsupported_claim",
                        "rule": "strict_lexical_support"
                        if strict_support and lexical_mismatch
                        else "election_result_support",
                        "strict_support_required": bool(strict_support),
                        "claim_term_count": len(claim_terms),
                        "matched_term_count": len(matched_terms),
                        "contradiction_codes": conflicts,
                    },
                )
                return None, "unsupported_claim", used + [source_id], False
            used.append(source_id)
            # Source IDs are rendered by the application, not supplied inside
            # model text. The speech boundary strips these labels for TTS.
            punctuation = text[-1:] if text[-1:] in ".!?" else ""
            body = text[:-1].rstrip() if punctuation else text
            rendered.append(f"{body} [{source_id}]{punctuation}")
        citations = sorted(set(used))
        if {item["source_id"] for item in self.citation_sources(citations)} != set(citations):
            return None, "unavailable_citation_url", citations, False
        return " ".join(rendered), None, citations, False

    def _validate_cited_research_prose(self, raw, *, question="", strict_support=False):
        """Accept ordinary prose only when each claim names a retained source.

        OpenRouter-compatible endpoints may return a readable, explicitly
        cited answer despite the requested JSON schema. Resolve that explicit
        source citation to the best retained projected passage, then run the
        unchanged claim validator. This never invents a citation or treats an
        ID alone as proof of support.
        """
        if not isinstance(raw, str) or "{" in raw or "}" in raw:
            return None, "invalid_structured_answer", [], False
        sentences = [part.strip() for part in _research_sentences(raw) if part.strip()]
        if not 1 <= len(sentences) <= 2:
            return None, "invalid_structured_answer", [], False

        source_ids = set(self._structured_research_source_ids())
        rendered = []
        used_sources = []
        for sentence in sentences:
            citations = re.findall(r"\[(S\d+)\]", sentence)
            if not citations:
                return None, "missing_research_citation", used_sources, False
            if len(citations) != 1:
                return None, "invalid_structured_answer", used_sources, False
            source_id = citations[0]
            if source_id not in source_ids:
                return None, "unsupported_source", used_sources, True
            claim = re.sub(r"\s*\[S\d+\]", "", sentence).strip()
            if not claim or re.search(r"\[S\d+\]", claim, flags=re.I):
                return None, "invalid_structured_answer", used_sources, False

            passage_ids = [
                passage_id
                for passage_id in self._structured_research_passage_ids()
                if passage_id.startswith(source_id + "-")
            ]
            candidates = []
            claim_terms = _research_content_terms(claim)
            for passage_id in passage_ids:
                passage = self.evidence_passage_lookup(passage_id)
                if not isinstance(passage, dict) or passage.get("source_id") != source_id:
                    continue
                passage_terms = _research_content_terms(passage.get("text", ""))
                overlap = len(claim_terms & passage_terms)
                candidates.append((overlap, passage_id))
            if not candidates:
                return None, "unsupported_evidence_passage", used_sources, True

            # A source with one projected passage is unambiguous. With several,
            # require at least one shared content term to resolve the cited
            # source to evidence rather than picking an arbitrary passage.
            candidates.sort(key=lambda row: (-row[0], row[1]))
            if len(candidates) > 1 and candidates[0][0] == 0:
                return None, "unsupported_evidence_passage", used_sources, True
            passage_id = candidates[0][1]
            checked, failure, citations, invalid = self._validate_structured_research_answer(
                json.dumps({"claims": [{"text": claim, "passage_id": passage_id}]}),
                question=question,
                strict_support=strict_support,
                diagnostic_draft=raw,
            )
            if failure:
                return None, failure, used_sources + citations, invalid
            punctuation = sentence[-1:] if sentence[-1:] in ".!?" else ""
            body = claim[:-1].rstrip() if claim[-1:] in ".!?" else claim
            rendered.append(f"{body} [{source_id}]{punctuation}")
            used_sources.append(source_id)

        citations = sorted(set(used_sources))
        if {item["source_id"] for item in self.citation_sources(citations)} != set(citations):
            return None, "unavailable_citation_url", citations, False
        self.on_event(
            "llm_cited_prose_accepted",
            source_ids=citations,
            claim_count=len(rendered),
        )
        return " ".join(rendered), None, citations, False

    def _log_llm_response(self, context):
        """Keep Pipecat's request log useful without dumping user conversation text."""
        try:
            message_count = len(context.get_messages())
        except (AttributeError, TypeError):
            message_count = 0
        logger.debug("{}: Generating LLM response ({} context messages)", self, message_count)

    async def _track_ledger_stream(self, stream, request_id):
        """Finalize only after complete consumption; cancellation stays unknown."""
        actual_input_tokens = None
        actual_output_tokens = None
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    if callable(getattr(usage, "model_dump", None)):
                        usage = usage.model_dump()
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                        completion_tokens = usage.get(
                            "completion_tokens", usage.get("output_tokens")
                        )
                        actual_input_tokens = (
                            prompt_tokens
                            if type(prompt_tokens) is int and 0 <= prompt_tokens <= 2**31 - 1
                            else None
                        )
                        actual_output_tokens = (
                            completion_tokens
                            if type(completion_tokens) is int
                            and 0 <= completion_tokens <= 2**31 - 1
                            else None
                        )
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = self.safe_error(exc)
            self.budget.finish_request(
                request_id,
                "failure",
                error.code,
                input_tokens_actual=actual_input_tokens,
                output_tokens_actual=actual_output_tokens,
            )
            raise
        else:
            self.budget.finish_request(
                request_id,
                "success",
                input_tokens_actual=actual_input_tokens,
                output_tokens_actual=actual_output_tokens,
            )

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

    async def observe(
        self,
        stream,
        epoch,
        started,
        *,
        request_id=None,
        structured_research=False,
        research_question="",
        strict_research_support=False,
        owner_turn_id=None,
        research_repair_context_id=None,
        direct_research_answer=False,
    ):
        first = True
        total_chars = 0
        completed = False
        named_indices = set()
        response_tool_names = set()
        tail = ""
        research_text = []
        structured_text = []
        research_citations = set()
        invalid_research_citation = False
        research_finalized = False
        request_terminal = False

        def terminal(outcome, error_code=None):
            nonlocal request_terminal
            if request_terminal:
                return
            request_terminal = True
            fields = {
                "request_id": request_id,
                "outcome": outcome,
                "elapsed_seconds": max(0.0, time.monotonic() - started),
            }
            if owner_turn_id is not None:
                fields["owner_turn_id"] = owner_turn_id
            if error_code:
                fields["error_code"] = error_code
            self.on_event("llm_stream_terminal", **fields)

        validate_research_answer = bool(
            self.research_tool_calls
            and not self.response_failure_reason
            and not structured_research
            and not direct_research_answer
        )
        try:
            async for chunk in stream:
                if epoch != self.epoch:
                    self.on_event(
                        "llm_stale_rejected",
                        request_id=request_id,
                        owner_turn_id=owner_turn_id,
                    )
                    terminal("superseded", "turn_epoch_changed")
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
                        and (
                            structured_research
                            or validate_research_answer
                            or direct_research_answer
                        )
                    )
                    if research_finalized and is_research_final:
                        # Some compatible streaming endpoints emit another
                        # terminal choice (often alongside usage metadata).
                        # The verified answer has already been sent downstream;
                        # never render or speak it a second time.
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
                                response_tool_names.add(call.function.name)
                            total_chars += len(call.function.arguments or "") + len(
                                call.function.name or ""
                            )
                    if total_chars > 32000:
                        raise ProviderError(
                            "response exceeds character budget", code="response_budget_exhausted"
                        )
                    text = choice.delta.content
                    if text:
                        if structured_research:
                            structured_text.append(text)
                            text = ""
                        else:
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
                        direct_research_answer
                        and choice.finish_reason == "stop"
                        and not self.response_tool_indices
                    ):
                        research_finalized = True
                        self.on_event(
                            "tavily_answer_stream_finished",
                            completed=True,
                            citation_count=1,
                        )
                    if (
                        structured_research
                        and choice.finish_reason == "stop"
                        and not self.response_tool_indices
                    ):
                        research_finalized = True
                        raw_answer = "".join(structured_text)
                        self._rejected_research_passage = None
                        rendered, failure, citations, invalid_citation = (
                            self._validate_structured_research_answer(
                                raw_answer,
                                question=research_question,
                                strict_support=strict_research_support,
                            )
                        )
                        self.on_event(
                            "llm_citation_check",
                            response_characters=len(raw_answer),
                            citation_ids=citations,
                            invalid_citation=invalid_citation,
                            retained_source_ids=sorted(self.source_ids()),
                            evidence_valid=failure is None,
                            citation_sources=(
                                self.citation_sources(citations) if failure is None else []
                            ),
                        )
                        if failure:
                            self.response_failure_reason = failure
                            choice.delta.content = "I couldn't verify this answer from a retrieved source, so I won't guess."
                            self.on_event(
                                "llm_guardrail_termination",
                                reason=failure,
                                retained_source_count=len(self.source_ids()),
                            )
                            self._finish_research_evidence_repair(failure)
                        else:
                            choice.delta.content = rendered
                            self.on_event(
                                "llm_grounding_check",
                                citation_count=len(citations),
                                retained_source_count=len(self.source_ids()),
                            )
                            self._finish_research_evidence_repair()
                    if (
                        validate_research_answer
                        and choice.finish_reason == "stop"
                        and not self.response_tool_indices
                    ):
                        research_finalized = True
                        if invalid_research_citation:
                            self.response_failure_reason = "unsupported_source"
                        elif not self.source_ids():
                            self.response_failure_reason = "insufficient_verified_evidence"
                        elif not research_citations:
                            self.response_failure_reason = "missing_research_citation"
                        elif {
                            item["source_id"] for item in self.citation_sources(research_citations)
                        } != research_citations:
                            self.response_failure_reason = "unavailable_citation_url"
                        evidence_valid = self.response_failure_reason is None
                        citation_sources = (
                            self.citation_sources(research_citations) if evidence_valid else []
                        )
                        self.on_event(
                            "llm_citation_check",
                            response_characters=sum(map(len, research_text)) + len(text or ""),
                            citation_ids=sorted(research_citations),
                            invalid_citation=invalid_research_citation,
                            retained_source_ids=sorted(self.source_ids()),
                            evidence_valid=evidence_valid,
                            citation_sources=citation_sources,
                        )
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
            if epoch != self.epoch:
                terminal("superseded", "turn_epoch_changed")
                return
            if first or not completed:
                raise ProviderError(
                    "empty or incomplete reasoning stream", code="provider_incomplete_response"
                )
            if tail:
                raise ProviderError("incomplete citation rejected", code="unsupported_source")
            if research_repair_context_id is not None:
                repair_state = self._research_repair_tool_rounds.get(research_repair_context_id)
                if repair_state is not None and self.response_tool_indices:
                    if len(self.response_tool_indices) != 1 or response_tool_names != {
                        "retrieve_source"
                    }:
                        raise ProviderError(
                            "research evidence repair permits one source retrieval",
                            code="tool_budget_exhausted",
                        )
                    repair_state["retrieval_used"] = True
                    self.on_event(
                        "llm_research_evidence_repair_retrieval_selected",
                        tool_name="retrieve_source",
                    )
                elif repair_state is not None:
                    self._research_repair_tool_rounds.pop(research_repair_context_id, None)
                    self._research_repair_contexts.discard(research_repair_context_id)
                    self._finish_research_evidence_repair(self.response_failure_reason)
            terminal(
                "rejected" if self.response_failure_reason else "completed",
                self.response_failure_reason,
            )
        except asyncio.CancelledError:
            if research_repair_context_id is not None:
                self._research_repair_tool_rounds.pop(research_repair_context_id, None)
                self._research_repair_contexts.discard(research_repair_context_id)
            self._finish_research_evidence_repair("interrupted")
            terminal("cancelled", "pipecat_interruption")
            raise
        except Exception as exc:
            if research_repair_context_id is not None:
                self._research_repair_tool_rounds.pop(research_repair_context_id, None)
                self._research_repair_contexts.discard(research_repair_context_id)
            error = self.safe_error(exc)
            self._finish_research_evidence_repair(error.code)
            terminal("failed", error.code)
            raise error from None
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
