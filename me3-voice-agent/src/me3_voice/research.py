"""Bounded read-only research, transport permission and original-source provenance.

No agent loop. aiohttp resolver checks the actual connection addresses, including
redirect destinations; no proxy/cookies, credentials in URLs or private IP access.
"""

from collections import OrderedDict
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
from html.parser import HTMLParser
import ipaddress
import json
import os
import re
import socket
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit, urljoin
import aiohttp


def research_sentences(text):
    """Keep abbreviations like U.S. attached to the office that follows."""
    return re.split(r"(?<!\b[A-Z]\.[A-Z]\.)(?<=[.!?])\s+|\n+", text)


class ResearchError(ValueError):
    def __init__(
        self,
        code,
        *,
        status=None,
        host=None,
        stage=None,
        failure_class=None,
        redirect_count=None,
        local_extraction=None,
        fallback_status=None,
        fallback_error=None,
        alternate_source_attempt=None,
    ):
        super().__init__(code)
        self.code = code
        self.status = status
        self.host_hash = hashlib.sha256(host.encode()).hexdigest()[:12] if host else None
        self.stage = stage
        self.failure_class = failure_class
        self.redirect_count = redirect_count
        self.local_extraction = local_extraction
        self.fallback_status = fallback_status
        self.fallback_error = fallback_error
        self.alternate_source_attempt = alternate_source_attempt

    def result(self):
        result = {"ok": False, "error": self.code}
        if self.status is not None:
            result["http_status"] = self.status
        if self.host_hash:
            result["host_hash"] = self.host_hash
        for key, value in (
            ("stage", self.stage),
            ("failure_class", self.failure_class),
            ("redirect_count", self.redirect_count),
            ("local_extraction", self.local_extraction),
            ("fallback_status", self.fallback_status),
        ):
            if value is not None:
                result[key] = value
        if self.fallback_error:
            result["fallback_error"] = self.fallback_error
        if self.alternate_source_attempt:
            result["alternate_source_attempt"] = self.alternate_source_attempt
        return result


@dataclass(frozen=True)
class ResearchOptions:
    allow_live: bool = False
    request_limit: int | None = 0
    answer_mode: str = "llm"

    def __post_init__(self):
        if (
            type(self.allow_live) is not bool
            or (
                self.request_limit is not None
                and (
                    type(self.request_limit) is not int
                    or not 0 <= self.request_limit <= 100
                )
            )
            or self.answer_mode not in {"llm", "tavily"}
        ):
            raise ValueError("research allow_live, request_limit, or answer_mode is invalid")


def public_url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or any(ord(c) < 33 for c in value):
        raise ResearchError("invalid_url")
    try:
        u = urlsplit(value)
        if (
            u.scheme not in {"https", "http"}
            or not u.hostname
            or u.username is not None
            or u.password is not None
            or u.fragment
            or u.port not in (None, 80 if u.scheme == "http" else 443)
        ):
            raise ValueError
        host = u.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            addr = None
        if addr is not None and not addr.is_global:
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise ResearchError("forbidden_url") from None


class PublicResolver(aiohttp.abc.AbstractResolver):
    def __init__(self):
        self.resolver = aiohttp.resolver.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await self.resolver.resolve(host, port, family)
        if not records or any(not ipaddress.ip_address(r["host"]).is_global for r in records):
            raise ResearchError(
                "forbidden_address",
                host=host,
                stage="dns_policy",
                failure_class="non_public_address",
            )
        return records

    async def close(self):
        await self.resolver.close()


class PageText(HTMLParser):
    """Prefer semantic article content and discard common page chrome."""

    BOILERPLATE = {
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "script",
        "style",
        "noscript",
        "template",
        "svg",
    }
    CONTENT = {"main", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.all_parts = []
        self.content_parts = []
        self.titles = []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        # HTML from the public web is often malformed; unwind through the
        # nearest matching open element instead of leaving suppression stuck.
        if tag in self.stack:
            del self.stack[len(self.stack) - 1 - self.stack[::-1].index(tag) :]

    def handle_data(self, text):
        if "title" in self.stack:
            self.titles.append(text)
        if any(tag in self.BOILERPLATE for tag in self.stack):
            return
        self.all_parts.append(text)
        if any(tag in self.CONTENT for tag in self.stack):
            self.content_parts.append(text)

    @staticmethod
    def normalize(parts):
        return " ".join(" ".join(parts).split())

    def extract(self):
        primary = self.normalize(self.content_parts)
        fallback = self.normalize(self.all_parts)
        # Prefer main/article text, but retain readable non-semantic pages.
        # Reject title-only shells and short JS placeholders as non-evidence.
        text = primary if len(primary) >= 80 else fallback
        if len(text.split()) < 12 or any(
            phrase in text.casefold()
            for phrase in (
                "enable javascript to view",
                "requires javascript",
                "rendered by javascript",
            )
        ):
            raise ResearchError("insufficient_source_content")
        return text, self.normalize(self.titles)[:250]


class ResearchClient:
    def __init__(self, options=None, credentials=None):
        self.options = options or ResearchOptions()
        self.credentials = credentials or {}
        self.requests = 0
        self.search_requests = 0
        self.client = None
        self.sources = OrderedDict()
        self.next_id = 0
        self.cache = OrderedDict()
        self.cache_ttl_seconds = 300
        self.cache_limit = 32
        self.search_leads = OrderedDict()
        self.search_snippet_source_ids = {}
        self.failed_retrieval_urls = set()
        self.projected_passage_ids = set()
        self.provider_state = "NOT_PROBED"
        self.operation_state = "NOT_PROBED"
        self.last_checked_at = None

    def usage_count(self):
        """Return process count plus any durable run-scoped Tavily reservation count."""
        ledger = os.environ.get("ME3_RESEARCH_LEDGER")
        campaign = os.environ.get("ME3_REQUEST_RUN_ID")
        if not ledger or not campaign or not Path(ledger).is_file():
            return self.requests
        try:
            with sqlite3.connect(ledger, timeout=1) as db:
                exists = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_campaign'"
                ).fetchone()
                if not exists:
                    return self.requests
                row = db.execute(
                    "SELECT used FROM research_campaign WHERE run_id=?", (campaign,)
                ).fetchone()
        except sqlite3.Error:
            return self.requests
        return max(self.requests, row[0] if row else 0)

    @staticmethod
    def _ensure_evidence_passages(source):
        """Attach stable sentence/word-boundary passage IDs to one retained source."""
        passages = source.get("evidence_passages")
        if isinstance(passages, dict) and passages:
            return passages
        source_id = source.get("source_id")
        text = source.get("text")
        if not isinstance(source_id, str) or not re.fullmatch(r"S\d+", source_id):
            return {}
        if not isinstance(text, str) or not text.strip():
            return {}

        passages = OrderedDict()
        ordinal = 0
        for sentence in research_sentences(text):
            remaining = sentence.strip()
            while remaining:
                ordinal += 1
                if len(remaining) <= 1000:
                    piece, remaining = remaining, ""
                else:
                    boundary = remaining.rfind(" ", 0, 1001)
                    if boundary < 1:
                        boundary = 1000
                    piece, remaining = remaining[:boundary].strip(), remaining[boundary:].strip()
                if piece:
                    passages[f"{source_id}-P{ordinal}"] = piece
        source["evidence_passages"] = passages
        return passages

    def evidence_passage_ids(self):
        """Passages actually projected to model context and still source-retained."""
        available = {
            passage_id
            for source in self.sources.values()
            for passage_id in self._ensure_evidence_passages(source)
        }
        return sorted(self.projected_passage_ids & available, key=self._passage_order)

    @staticmethod
    def _passage_order(passage_id):
        match = re.fullmatch(r"S(\d+)-P(\d+)", passage_id)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    def lookup_evidence_passage(self, passage_id):
        if passage_id not in self.projected_passage_ids:
            return None
        for source_id, source in self.sources.items():
            text = self._ensure_evidence_passages(source).get(passage_id)
            if isinstance(text, str):
                return {"passage_id": passage_id, "source_id": source_id, "text": text}
        return None

    def _cached(self, key):
        row = self.cache.get(key)
        if not row:
            return None
        expires, value = row
        if expires <= time.monotonic():
            self.cache.pop(key, None)
            return None
        self.cache.move_to_end(key)
        return value

    def _remember(self, key, value):
        self.cache[key] = (time.monotonic() + self.cache_ttl_seconds, value)
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

    @staticmethod
    def _query_terms(query):
        return {
            w
            for w in re.findall(r"[a-z0-9]{3,}", query.casefold())
            if w
            not in {
                "the",
                "and",
                "for",
                "who",
                "what",
                "when",
                "where",
                "with",
                "from",
                "current",
                "currently",
                "official",
                "source",
            }
        }

    @classmethod
    def _rank_leads(cls, query, rows):
        terms = cls._query_terms(query)

        def rank(row):
            host = urlsplit(row["url"]).hostname or ""
            host_terms = set(re.findall(r"[a-z0-9]{3,}", host.casefold()))
            text_terms = set(
                re.findall(r"[a-z0-9]{3,}", (row["title"] + " " + row["snippet"]).casefold())
            )
            relevance = len(terms & (host_terms | text_terms))
            authoritative_host = host.endswith((".gov", ".edu", ".int")) or ".gov." in host
            return (relevance, authoritative_host, row.get("provider_score", 0.0))

        return sorted(rows, key=rank, reverse=True)

    @classmethod
    def rank_answer_support(cls, answer, rows):
        """Order already-returned search leads by overlap with a provider answer."""
        return cls._rank_leads(answer, rows)

    def authorize(self):
        limit = self.options.request_limit
        if not self.options.allow_live or (limit is not None and self.requests >= limit):
            raise ResearchError("research_disabled_or_budget_exhausted")
        # The optional demo campaign ledger survives reconnect/client replacement.
        import os
        import sqlite3
        ledger = os.environ.get("ME3_RESEARCH_LEDGER")
        campaign = os.environ.get("ME3_REQUEST_RUN_ID")
        if ledger:
            if not campaign:
                raise ResearchError("research_campaign_missing")
            with sqlite3.connect(ledger, timeout=1) as db:
                db.execute("CREATE TABLE IF NOT EXISTS research_campaign (run_id TEXT PRIMARY KEY, used INTEGER NOT NULL, cap INTEGER NOT NULL)")
                db.execute("BEGIN IMMEDIATE")
                cap = -1 if limit is None else limit
                db.execute("INSERT OR IGNORE INTO research_campaign VALUES (?,0,?)", (campaign, cap))
                used, cap = db.execute("SELECT used,cap FROM research_campaign WHERE run_id=?", (campaign,)).fetchone()
                requested_cap = -1 if limit is None else limit
                if cap != requested_cap or (cap != -1 and used >= cap):
                    raise ResearchError("research_disabled_or_budget_exhausted")
                db.execute("UPDATE research_campaign SET used=used+1 WHERE run_id=?", (campaign,))
        self.requests += 1  # Reserve before I/O; timeouts do not refund/retry.

    def session(self):
        if self.client is None:
            self.client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False),
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
                trust_env=False,
                cookie_jar=aiohttp.DummyCookieJar(),
                timeout=aiohttp.ClientTimeout(total=8),
            )
        return self.client

    async def read(self, method, url, **kwargs):
        self.authorize()
        safe_url = public_url(url)
        host = urlsplit(safe_url).hostname
        try:
            response_context = self.session().request(
                method, safe_url, allow_redirects=False, **kwargs
            )
            async with response_context as response:
                return await self._read_response(response, url, host)
        except ResearchError:
            raise
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
            raise ResearchError(
                "research_timeout", host=host, stage="transport", failure_class="timeout"
            ) from None
        except aiohttp.ClientConnectorDNSError:
            raise ResearchError(
                "research_dns_failure", host=host, stage="dns", failure_class="resolution"
            ) from None
        except aiohttp.ClientConnectorCertificateError:
            raise ResearchError(
                "research_tls_failure", host=host, stage="tls", failure_class="certificate"
            ) from None
        except aiohttp.ClientSSLError:
            raise ResearchError(
                "research_tls_failure", host=host, stage="tls", failure_class="handshake"
            ) from None
        except aiohttp.ClientConnectorError as exc:
            error_number = getattr(getattr(exc, "os_error", None), "errno", None)
            if error_number in {errno.EACCES, errno.EPERM}:
                code, failure = "research_network_restricted", "permission_denied"
            elif error_number in {errno.ENETUNREACH, errno.EHOSTUNREACH}:
                code, failure = "research_network_unreachable", "route_unavailable"
            elif error_number == errno.ECONNREFUSED:
                code, failure = "research_connection_refused", "connection_refused"
            else:
                code, failure = "research_connect_failure", "connection_failed"
            raise ResearchError(code, host=host, stage="connect", failure_class=failure) from None
        except aiohttp.ClientPayloadError:
            raise ResearchError(
                "research_response_read_failure",
                host=host,
                stage="response_body",
                failure_class="incomplete_or_invalid_body",
            ) from None
        except aiohttp.ClientError:
            raise ResearchError(
                "research_transport_failure",
                host=host,
                stage="transport",
                failure_class="client_error",
            ) from None

    async def _read_response(self, response, url, host):
        if response.status in {301, 302, 303, 307, 308}:
            return {
                "redirect": urljoin(url, response.headers.get("Location", "")),
                "redirect_status": response.status,
            }
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ResearchError(
                "unsupported_source_encoding",
                host=host,
                stage="response",
                failure_class="unsupported_encoding",
            )
        if response.status != 200:
            raise ResearchError(
                "research_http_failure",
                status=response.status,
                host=host,
                stage="http_status",
                failure_class="non_success_status",
            )
        data = bytearray()
        async for chunk in response.content.iter_chunked(8192):
            data.extend(chunk)
            if len(data) > 131072:
                raise ResearchError(
                    "source_too_large", host=host, stage="response_body", failure_class="body_limit"
                )
        return {
            "data": bytes(data),
            "type": response.content_type,
            "last_modified": response.headers.get("Last-Modified"),
        }

    async def search(self, query, max_results=5):
        normalized_query = " ".join(query.casefold().split()) + f"\0{max_results}"
        key_cache = "search:" + hashlib.sha256(normalized_query.encode()).hexdigest()
        cached = self._cached(key_cache)
        if cached is not None:
            self._remember_search_leads(cached)
            return {**cached, "cache_hit": True}
        limit = self.options.request_limit
        if not self.options.allow_live or (limit is not None and self.requests >= limit):
            raise ResearchError("research_disabled_or_budget_exhausted")
        key = self.credentials.get("ME3_RESEARCH_TAVILY_API_KEY") or os.environ.get(
            "ME3_RESEARCH_TAVILY_API_KEY", ""
        )
        if not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            self.provider_state = "NOT_CONFIGURED"
            self.operation_state = "NOT_PROBED"
            raise ResearchError("search_credential_missing")
        self.search_requests += 1  # Includes failed Tavily API attempts.
        try:
            row = await self.read(
                "POST",
                "https://api.tavily.com/search",
                headers={"Authorization": "Bearer " + key},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": "basic" if self.options.answer_mode == "tavily" else False,
                    "include_raw_content": False,
                    "auto_parameters": False,
                },
            )
        except ResearchError as exc:
            self.provider_state = (
                "AUTH_REJECTED"
                if exc.status in {401, 403}
                else "REACHABLE_RATE_LIMITED"
                if exc.status == 429
                else "UNAVAILABLE"
            )
            self.operation_state = "FAILED"
            self.last_checked_at = datetime.now(timezone.utc).isoformat()
            raise
        self.provider_state = "AUTHENTICATED_REACHABLE"
        self.last_checked_at = datetime.now(timezone.utc).isoformat()
        if "redirect" in row:
            self.operation_state = "REDIRECT_REJECTED"
            raise ResearchError(
                "search_redirect_rejected",
                status=row.get("redirect_status"),
                host="api.tavily.com",
                stage="redirect",
                failure_class="redirect_not_followed",
            )
        try:
            result_payload = json.loads(row["data"])
            results = result_payload["results"]
            if not isinstance(results, list):
                raise ValueError
            leads = []
            for item in results[:max_results]:
                try:
                    url = public_url(item["url"])
                except (ResearchError, KeyError, TypeError):
                    continue
                leads.append(
                    {
                        "url": url,
                        "title": str(item.get("title", ""))[:250],
                        "snippet": str(item.get("content", ""))[:320],
                        "evidence_status": "discovery_only",
                        "provider_score": float(item.get("score", 0.0))
                        if isinstance(item.get("score", 0.0), (int, float))
                        else 0.0,
                        "published_date_reported": str(item.get("published_date", ""))[:80],
                    }
                )
            leads = self._rank_leads(query, leads)
            for lead in leads:
                lead.pop("provider_score", None)
                lead["ranked_for_query"] = True
            result = {
                "kind": "untrusted_search_leads",
                "query": query,
                "results": leads,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
            if self.options.answer_mode == "tavily":
                answer = result_payload.get("answer")
                if isinstance(answer, str) and answer.strip():
                    result["answer"] = answer
            self._remember(key_cache, result)
            self._remember_search_leads(result)
            self.operation_state = "SEARCH_READY"
            return result
        except (ValueError, KeyError, TypeError):
            self.operation_state = "INVALID_RESPONSE"
            raise ResearchError("invalid_search_response") from None

    def _remember_search_leads(self, result):
        for lead in result.get("results", []) if isinstance(result, dict) else ():
            if isinstance(lead, dict) and isinstance(lead.get("url"), str):
                self.search_leads[lead["url"]] = lead
                self.search_leads.move_to_end(lead["url"])
        while len(self.search_leads) > 20:
            self.search_leads.popitem(last=False)

    def retain_search_snippet(self, lead):
        """Give a bounded Tavily snippet explicit, non-fetched provenance."""
        if not isinstance(lead, dict) or not isinstance(lead.get("url"), str):
            return None
        url = public_url(lead["url"])
        text = " ".join(str(lead.get("snippet", "")).split())[:320]
        if len(text) < 40:
            return None
        for source in self.sources.values():
            if source.get("url") == url:
                return source if source.get("extraction_method") == "tavily_search_snippet" else None
        source_id = self.search_snippet_source_ids.get(url)
        if source_id and source_id in self.sources:
            return self.sources[source_id]
        self.next_id += 1
        source_id = f"S{self.next_id}"
        source = {
            "source_id": source_id,
            "url": url,
            "title": str(lead.get("title", ""))[:250],
            "text": text,
            "kind": "untrusted_search_result",
            "truncated": len(str(lead.get("snippet", ""))) > len(text),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "last_modified_reported": "",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "extraction_method": "tavily_search_snippet",
            "evidence_status": "search_result_snippet",
            "fetched": False,
        }
        self.sources[source_id] = source
        self.search_snippet_source_ids[url] = source_id
        return source

    def discard_search_snippet(self, source_id):
        source = self.sources.get(source_id)
        if not isinstance(source, dict) or source.get("extraction_method") != "tavily_search_snippet":
            return
        self.sources.pop(source_id, None)
        self.projected_passage_ids.difference_update(self._ensure_evidence_passages(source))
        self.search_snippet_source_ids.pop(source.get("url"), None)

    def alternative_leads(self, *, exclude_url=None, limit=3):
        retrieved_urls = {
            row.get("url")
            for row in self.sources.values()
            if row.get("kind") == "untrusted_retrieved_source"
        }
        excluded = {exclude_url} | self.failed_retrieval_urls | retrieved_urls
        candidates = [
            dict(lead) for url, lead in self.search_leads.items() if url not in excluded
        ]
        # A host-level transport failure makes another URL on the same host a
        # poor immediate fallback. Prefer another publisher when Tavily already
        # returned one, while retaining same-host leads as a last resort.
        failed_hosts = {
            (urlsplit(url).hostname or "").casefold() for url in self.failed_retrieval_urls
        }
        candidates.sort(
            key=lambda lead: (urlsplit(lead["url"]).hostname or "").casefold() in failed_hosts
        )
        return candidates[:limit]

    async def retrieve(self, url, *, allow_extract_fallback=True):
        url = public_url(url)
        if url in self.failed_retrieval_urls:
            raise ResearchError(
                "source_previously_failed",
                host=urlsplit(url).hostname,
                stage="session_cache",
                failure_class="not_retried",
            )
        key_cache = "retrieve:" + hashlib.sha256(url.encode()).hexdigest()
        cached = self._cached(key_cache)
        if cached is not None:
            return cached
        redirect_count = 0
        fallback_text = None
        fallback_status = None
        fallback_error = None
        for _ in range(4):
            try:
                row = await self.read("GET", url)
            except ResearchError as exc:
                # Basic Extract is one bounded alternative for an explicitly
                # unavailable page. Unsafe URLs, malformed requests and local
                # execution restrictions are never bypassed.
                fallback_eligible = exc.code in {
                    "research_http_failure",
                    "research_timeout",
                    "research_connect_failure",
                    "research_connection_refused",
                    "source_too_large",
                }
                if not fallback_eligible or not allow_extract_fallback:
                    self.failed_retrieval_urls.add(url)
                    raise
                fallback_text, fallback_status, fallback_error = await self._tavily_extract(url)
                if not fallback_text:
                    self.failed_retrieval_urls.add(url)
                    raise ResearchError(
                        exc.code,
                        status=exc.status,
                        host=urlsplit(url).hostname,
                        stage=exc.stage,
                        failure_class=exc.failure_class,
                        fallback_status=fallback_status,
                        fallback_error=fallback_error or {"error": "no_extracted_content"},
                    ) from None
                row = {"data": b"", "type": "text/plain", "last_modified": None}
                break
            if "redirect" not in row:
                break
            redirect_count += 1
            try:
                url = public_url(row["redirect"])
            except ResearchError:
                raise ResearchError(
                    "redirect_target_rejected",
                    host=urlsplit(url).hostname,
                    stage="redirect",
                    failure_class="unsafe_target",
                    redirect_count=redirect_count,
                ) from None
        else:
            raise ResearchError(
                "redirect_limit",
                host=urlsplit(url).hostname,
                stage="redirect",
                failure_class="too_many_redirects",
                redirect_count=redirect_count,
            )
        decoded = row["data"].decode("utf-8", errors="replace")
        if fallback_text is not None:
            text, title, extraction_method = fallback_text, "", "tavily_basic_extract"
            local_extraction = "unavailable"
        elif row["type"] == "text/html":
            parser = PageText()
            parser.feed(decoded)
            try:
                text, title = parser.extract()
                extraction_method = "local_html"
            except ResearchError as exc:
                if exc.code != "insufficient_source_content":
                    raise
                text, title = "", " ".join(parser.titles)[:250]
        elif row["type"] == "text/plain":
            text, title, extraction_method = decoded, "", "plain_text"
        else:
            raise ResearchError(
                "unsupported_source_type",
                host=urlsplit(url).hostname,
                stage="extraction",
                failure_class="unsupported_content_type",
            )
        text = " ".join(text.split())
        text = "".join(c for c in text if ord(c) >= 32)
        if fallback_text is None:
            local_extraction = "sufficient" if len(text.split()) >= 12 else "insufficient"
        if (
            allow_extract_fallback
            and fallback_text is None
            and local_extraction == "insufficient"
            and row["type"] == "text/html"
        ):
            extracted, fallback_status, fallback_error = await self._tavily_extract(url)
            if extracted:
                text = extracted
                extraction_method = "tavily_basic_extract"
        if not text or (row["type"] == "text/html" and len(text.split()) < 12):
            self.failed_retrieval_urls.add(url)
            raise ResearchError(
                "insufficient_source_content",
                host=urlsplit(url).hostname,
                stage="extraction",
                failure_class="insufficient_content",
                redirect_count=redirect_count,
                local_extraction=local_extraction,
                fallback_status=fallback_status,
                fallback_error=fallback_error,
            )
        truncated = len(text) > 8000
        text = text[:8000]
        self.next_id += 1
        source_id = "S" + str(self.next_id)
        source = {
            "source_id": source_id,
            "url": url,
            "title": title,
            "text": text,
            "kind": "untrusted_retrieved_source",
            "truncated": truncated,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "last_modified_reported": str(row.get("last_modified") or "")[:80],
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "extraction_method": extraction_method,
            "evidence_status": "retrieved_unverified",
            # Retrieval records content availability, not factual verification.
            "fetched": True,
            "local_extraction": local_extraction,
            "fallback_status": fallback_status,
            "fallback_error": fallback_error,
        }
        self._ensure_evidence_passages(source)
        self.sources[source_id] = source
        if len(self.sources) > 16:
            self.sources.popitem(last=False)
        self._remember(key_cache, source)
        return source

    async def _tavily_extract(self, url):
        """One Basic Extract fallback, called only when local HTML lacks content."""
        if not self.options.allow_live:
            return "", "disabled", None
        if (
            self.options.request_limit is not None
            and self.requests >= self.options.request_limit
        ):
            return "", "budget_exhausted", None
        token = self.credentials.get("ME3_RESEARCH_TAVILY_API_KEY") or os.environ.get(
            "ME3_RESEARCH_TAVILY_API_KEY", ""
        )
        if not token:
            return "", "not_configured", None
        try:
            row = await self.read(
                "POST",
                "https://api.tavily.com/extract",
                headers={"Authorization": "Bearer " + token},
                json={"urls": [url], "extract_depth": "basic", "include_images": False},
            )
            if "redirect" in row:
                raise ResearchError(
                    "tavily_extract_redirect_rejected",
                    status=row.get("redirect_status"),
                    host="api.tavily.com",
                    stage="redirect",
                    failure_class="redirect_not_followed",
                )
            payload = json.loads(row["data"])
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not results or not isinstance(results[0], dict):
                return "", "empty_response", None
            text = results[0].get("raw_content") or results[0].get("content") or ""
            if isinstance(text, str) and len(text.split()) >= 12:
                return " ".join(text.split()), "succeeded", None
            return "", "insufficient_content", None
        except ResearchError as exc:
            return "", "failed", exc.result()
        except (ValueError, TypeError, KeyError):
            return "", "invalid_response", None

    def evidence_for_model(self, source, query, *, max_chars=2400):
        """Project source-owned passages with stable IDs; full text stays internal."""
        terms = self._query_terms(query)
        passages = self._ensure_evidence_passages(source)
        ranked = []
        for passage_id, passage_text in passages.items():
            words = set(re.findall(r"[a-z0-9]{3,}", passage_text.casefold()))
            score = len(terms & words)
            if score:
                ranked.append((score, passage_id, passage_text))
        selected = []
        used = 0
        for _, passage_id, passage_text in sorted(ranked, key=lambda row: (-row[0], row[1])):
            projected_size = len(passage_id) + 2 + len(passage_text)
            if used + projected_size > max_chars:
                continue
            selected.append((passage_id, passage_text))
            used += projected_size
        if not selected:
            first = next(iter(passages.items()), None)
            if first and len(first[1]) <= max_chars:
                selected = [first]
        result = {k: v for k, v in source.items() if k not in {"text", "evidence_passages"}}
        # Make the model-facing citation token explicit instead of requiring
        # the model to infer bracket syntax from the internal source ID.
        # This is a projection only; verification still checks the exact ID
        # against ResearchClient.sources and never trusts this display value.
        source_id = result.get("source_id")
        if isinstance(source_id, str) and re.fullmatch(r"S\d+", source_id):
            result["citation"] = f"[{source_id}]"
        result["passage_ids"] = [passage_id for passage_id, _ in selected]
        result["text"] = "\n".join(
            f"{passage_id}: {passage_text}" for passage_id, passage_text in selected
        )
        self.projected_passage_ids.update(result["passage_ids"])
        result["evidence_chunks"] = len(selected)
        return result

    def verify(self, source_id, quote):
        source = self.sources.get(source_id)
        if not source:
            raise ResearchError("unknown_source")
        found = " ".join(quote.split()) in source["text"]
        return {
            "source_id": source_id,
            "url": source["url"],
            "quote_found": found,
            "text_sha256": source["text_sha256"],
            "claim_truth_verified": False,
        }

    async def close(self):
        if self.client:
            await self.client.close()
