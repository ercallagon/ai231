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
import time
from urllib.parse import urlsplit, urljoin
import aiohttp


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
        return result


@dataclass(frozen=True)
class ResearchOptions:
    allow_live: bool = False
    request_limit: int = 0

    def __post_init__(self):
        if (
            type(self.allow_live) is not bool
            or type(self.request_limit) is not int
            or not 0 <= self.request_limit <= 100
        ):
            raise ValueError("research allow_live boolean and request_limit0..100 required")


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

    def authorize(self):
        if not self.options.allow_live or self.requests >= self.options.request_limit:
            raise ResearchError("research_disabled_or_budget_exhausted")
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
            return {**cached, "cache_hit": True}
        if not self.options.allow_live or self.requests >= self.options.request_limit:
            raise ResearchError("research_disabled_or_budget_exhausted")
        key = self.credentials.get("ME3_RESEARCH_TAVILY_API_KEY") or os.environ.get(
            "ME3_RESEARCH_TAVILY_API_KEY", ""
        )
        if not key or len(key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise ResearchError("search_credential_missing")
        self.search_requests += 1  # Includes failed Tavily API attempts.
        row = await self.read(
            "POST",
            "https://api.tavily.com/search",
            headers={"Authorization": "Bearer " + key},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "auto_parameters": False,
            },
        )
        if "redirect" in row:
            raise ResearchError(
                "search_redirect_rejected",
                status=row.get("redirect_status"),
                host="api.tavily.com",
                stage="redirect",
                failure_class="redirect_not_followed",
            )
        try:
            result = json.loads(row["data"])
            results = result["results"]
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
            self._remember(key_cache, result)
            return result
        except (ValueError, KeyError, TypeError):
            raise ResearchError("invalid_search_response") from None

    async def retrieve(self, url):
        url = public_url(url)
        key_cache = "retrieve:" + hashlib.sha256(url.encode()).hexdigest()
        cached = self._cached(key_cache)
        if cached is not None:
            return cached
        redirect_count = 0
        for _ in range(4):
            row = await self.read("GET", url)
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
        if row["type"] == "text/html":
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
        local_extraction = "sufficient" if len(text.split()) >= 12 else "insufficient"
        fallback_status = None
        fallback_error = None
        if local_extraction == "insufficient" and row["type"] == "text/html":
            extracted, fallback_status, fallback_error = await self._tavily_extract(url)
            if extracted:
                text = extracted
                extraction_method = "tavily_basic_extract"
        if not text or (row["type"] == "text/html" and len(text.split()) < 12):
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
        }
        self.sources[source_id] = source
        if len(self.sources) > 16:
            self.sources.popitem(last=False)
        self._remember(key_cache, source)
        return source

    async def _tavily_extract(self, url):
        """One Basic Extract fallback, called only when local HTML lacks content."""
        if not self.options.allow_live:
            return "", "disabled", None
        if self.requests >= self.options.request_limit:
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
        """Select lexical query-relevant sentences while retaining full source internally."""
        terms = self._query_terms(query)
        sentences = re.split(r"(?<=[.!?])\s+", source["text"])
        ranked = []
        for i, sentence in enumerate(sentences):
            words = set(re.findall(r"[a-z0-9]{3,}", sentence.casefold()))
            score = len(terms & words)
            if score:
                ranked.append((score, -i, sentence.strip()))
        selected = []
        used = 0
        for _, _, sentence in sorted(ranked, reverse=True):
            if used + len(sentence) > max_chars:
                continue
            selected.append(sentence)
            used += len(sentence) + 1
        if not selected:
            selected = [source["text"][: min(max_chars, 800)]]
        result = {k: v for k, v in source.items() if k != "text"}
        result["text"] = " ".join(selected)
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
