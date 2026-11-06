"""Restricted Onit chat worker. Run only through OnitTaskExecutor.

The worker offers isolated draft artifacts and an opt-in, bounded static-site
writer. It cannot invoke Onit's default MCP servers, shell, or deployment tools.
"""

import asyncio
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
import importlib

from .english import SWIFT_ENGLISH_OUTPUT_POLICY

_KEY = ""
_TAVILY_KEY = ""
_GITHUB_TOKEN = ""


def groq_tool_results(messages):
    """Remove Onit's local tool metadata unsupported by Groq's wire schema."""
    return [
        {k: v for k, v in message.items() if k != "parameters"}
        if isinstance(message, dict) and message.get("role") == "tool"
        else message
        for message in messages
    ]


def prewarm_sdk():
    """Load Onit and OpenAI's lazy chat resources before the task clock."""
    importlib.import_module("src.model.serving.chat")
    importlib.import_module("openai.resources")


class ArtifactTool:
    url = "swift-local-artifact"
    tool_item = {
        "type": "function",
        "function": {
            "name": "write_artifact",
            "description": "Create one draft website/document file in the isolated task workspace. No deployment, repository, shell or hardware action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "One filename ending in .html, .css, .js, .md, .txt or .json",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file content, at most 16 KiB",
                    },
                },
                "required": ["filename", "content"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, workspace, completed=None, on_completed=None):
        self.workspace = workspace
        self.completed = completed
        self.on_completed = on_completed
        self.artifacts = []
        self.bytes_written = 0

    def get_tool(self):
        return self.tool_item

    async def __call__(self, log_handler=None, **kwargs):
        if getattr(self, "on_invoked", None):
            self.on_invoked()
        if set(kwargs) != {"filename", "content"}:
            return "Rejected: provide only filename and content."
        filename, content = kwargs["filename"], kwargs["content"]
        if not isinstance(filename, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)", filename
        ):
            return "Rejected: unsupported filename."
        if not isinstance(content, str):
            return "Rejected: content must be text."
        encoded = content.encode("utf-8")
        if (
            len(encoded) > 16384
            or self.bytes_written + len(encoded) > 65536
            or len(self.artifacts) >= 8
        ):
            return "Rejected: task artifact size or count limit reached."
        if getattr(self, "on_started", None):
            self.on_started()
        try:
            fd = os.open(
                self.workspace / filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            return "Rejected: this filename already exists; choose another name."
        with os.fdopen(fd, "wb") as target:
            target.write(encoded)
        self.artifacts.append(filename)
        self.bytes_written += len(encoded)
        if self.completed:
            self.completed.set()
        if self.on_completed:
            self.on_completed()
        return f"Draft artifact {filename} created and verified in the isolated workspace."


class ProjectDirectoryTool:
    """Create one new project directory and optionally populate a static site."""

    url = "swift-project-directory"
    tool_item = {
        "type": "function",
        "function": {
            "name": "create_project_directory",
            "description": (
                "Create one new project directory under Swift's configured test-output root. "
                "For a website, include files with complete contents; index.html is required. "
                "Only flat static files are allowed. Existing directories/files are never changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Single new directory name"},
                    "files": {
                        "type": "array",
                        "description": "Optional static website files. Include index.html; use only flat filenames.",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "One root filename ending in .html, .css, .js, .md, .txt or .json",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Complete UTF-8 file content, at most 16 KiB",
                                },
                            },
                            "required": ["filename", "content"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, root, completed=None, on_completed=None):
        self.root = root
        self.completed = completed
        self.on_completed = on_completed
        self.created = []
        self.artifacts = []

    def get_tool(self):
        return self.tool_item

    async def __call__(self, log_handler=None, **kwargs):
        if getattr(self, "on_invoked", None):
            self.on_invoked()
        if set(kwargs) not in ({"name"}, {"name", "files"}) or not isinstance(
            kwargs.get("name"), str
        ):
            return "Rejected: provide one directory name and optional website files."
        name = kwargs["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name):
            return "Rejected: invalid project directory name."
        if self.created:
            return "Rejected: this task already created a project directory."
        files = kwargs.get("files")
        validated = []
        if files is not None:
            if not isinstance(files, list) or not 1 <= len(files) <= 8:
                return "Rejected: provide between one and eight website files."
            seen = set()
            total_bytes = 0
            for item in files:
                if not isinstance(item, dict) or set(item) != {"filename", "content"}:
                    return "Rejected: each website file needs only filename and content."
                filename, content = item["filename"], item["content"]
                if not isinstance(filename, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.(?:html|css|js|md|txt|json)",
                    filename,
                ):
                    return "Rejected: website filenames must be simple supported root filenames."
                if filename in seen:
                    return "Rejected: duplicate website filename."
                if not isinstance(content, str):
                    return "Rejected: website file content must be text."
                encoded = content.encode("utf-8")
                if len(encoded) > 16384:
                    return "Rejected: each website file is limited to 16 KiB."
                total_bytes += len(encoded)
                if total_bytes > 65536:
                    return "Rejected: website output is limited to 64 KiB total."
                seen.add(filename)
                validated.append((filename, encoded))
            if "index.html" not in seen:
                return "Rejected: a static website must include index.html."

        if getattr(self, "on_started", None):
            self.on_started()
        root_fd = -1
        project_fd = -1
        written = []
        try:
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.mkdir(name, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                return "Rejected: project directory already exists; no changes were made."
            if validated:
                project_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                for filename, encoded in validated:
                    descriptor = os.open(
                        filename,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=project_fd,
                    )
                    written.append(filename)
                    try:
                        with os.fdopen(descriptor, "wb") as target:
                            if target.write(encoded) != len(encoded):
                                raise OSError("short website file write")
                            target.flush()
                            os.fsync(target.fileno())
                    except BaseException:
                        # fdopen owns and closes the descriptor once entered.
                        raise
                self.artifacts = [filename for filename, _ in validated]
        except BaseException:
            if project_fd >= 0:
                for filename in written:
                    try:
                        os.unlink(filename, dir_fd=project_fd)
                    except OSError:
                        pass
                os.close(project_fd)
                project_fd = -1
            if root_fd >= 0:
                try:
                    os.rmdir(name, dir_fd=root_fd)
                except OSError:
                    pass
            raise
        finally:
            if project_fd >= 0:
                os.close(project_fd)
            if root_fd >= 0:
                os.close(root_fd)
        self.created.append(name)
        if self.completed:
            self.completed.set()
        if self.on_completed:
            self.on_completed()
        if self.artifacts:
            return (
                f"Created static website project {name} with {len(self.artifacts)} files. "
                "Files are present; the site was not run or deployed."
            )
        return f"Created empty project directory {name} under the configured test-output root."


class TavilySearchTool:
    """Basic Tavily search with an explicit per-task credit ceiling."""

    url = "swift-tavily-search"
    tool_item = {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": (
                "Search the web for current information. For weather, search the named city and "
                "say when the source was published. This is web search, not a dedicated forecast API. "
                "Use only when current information or sources are needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 3, "maxLength": 400},
                    "topic": {"type": "string", "enum": ["general", "news"]},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, api_key, credit_budget, *, client_factory=None):
        import httpx

        self.api_key = api_key
        self.credit_budget = credit_budget
        self.credits_reserved = 0
        self.client_factory = client_factory or httpx.AsyncClient

    def get_tool(self):
        return self.tool_item

    async def __call__(self, log_handler=None, **kwargs):
        import httpx

        if set(kwargs) - {"query", "topic"}:
            return "Rejected: unsupported web-search options."
        query, topic = kwargs.get("query"), kwargs.get("topic", "general")
        if (
            not isinstance(query, str)
            or not 3 <= len(query.strip()) <= 400
            or topic not in {"general", "news"}
        ):
            return "Rejected: provide a bounded query and supported search topic."
        if self.credits_reserved >= self.credit_budget:
            return "Tavily search allowance is exhausted for this bridge run."
        # Reserve before network I/O. Failures are conservatively charged so
        # retries cannot silently exceed the configured credit budget.
        self.credits_reserved += 1
        try:
            async with self.client_factory(
                timeout=15,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "query": query.strip(),
                        "search_depth": "basic",
                        "max_results": 3,
                        "topic": topic,
                        "include_answer": True,
                        "include_raw_content": False,
                        "include_usage": True,
                    },
                )
            if response.status_code != 200:
                return json.dumps(
                    {
                        "error": f"Tavily search failed with HTTP {response.status_code}.",
                        "credits_reserved": self.credits_reserved,
                    }
                )
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("invalid search response")
            rows = data.get("results", [])
            if not isinstance(rows, list):
                raise ValueError("invalid search results")
            results = []
            for row in rows[:3]:
                if not isinstance(row, dict):
                    continue
                title, url, content = row.get("title"), row.get("url"), row.get("content")
                if not all(isinstance(value, str) for value in (title, url, content)):
                    continue
                if not url.startswith(("https://", "http://")):
                    continue
                results.append(
                    {
                        "title": title[:180],
                        "url": url[:1000],
                        "content": content[:700],
                        "published_date": row.get("published_date")
                        if isinstance(row.get("published_date"), str)
                        else None,
                    }
                )
            answer = data.get("answer")
            usage = data.get("usage")
            credits = usage.get("credits") if isinstance(usage, dict) else None
            return json.dumps(
                {
                    "answer": answer[:1200] if isinstance(answer, str) else None,
                    "results": results,
                    "credits_used": credits if type(credits) is int and credits >= 0 else 1,
                },
                ensure_ascii=True,
            )
        except (httpx.RequestError, ValueError, TypeError):
            return json.dumps(
                {
                    "error": "Tavily search failed; credentials and response details were withheld.",
                    "credits_reserved": self.credits_reserved,
                }
            )


class GitHubReadTool:
    """Read-only subset of OnIt's GitHub repository tool."""

    url = "swift-github-read"
    tool_item = {
        "type": "function",
        "function": {
            "name": "github_repo_read",
            "description": "Read repository details or list repositories visible to the configured GitHub token. This tool cannot create, fork, modify, or delete repositories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["get", "list"]},
                    "name": {"type": "string", "maxLength": 200},
                    "org": {"type": "string", "maxLength": 100},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, token, *, client_factory=None, max_calls=3):
        import httpx

        self.token = token
        self.calls = 0
        self.max_calls = max_calls
        self.client_factory = client_factory or httpx.AsyncClient

    def get_tool(self):
        return self.tool_item

    async def __call__(self, log_handler=None, **kwargs):
        import httpx
        from urllib.parse import quote

        if set(kwargs) - {"action", "name", "org", "per_page"}:
            return "Rejected: unsupported GitHub options."
        action = kwargs.get("action")
        name, org = kwargs.get("name"), kwargs.get("org")
        if action == "get":
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name
            ):
                return "Rejected: get requires a repository in owner/name form."
            path = "/repos/" + quote(name, safe="/-_.")
            params = None
        elif action == "list":
            if org is not None and (
                not isinstance(org, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", org)
            ):
                return "Rejected: invalid organization name."
            path = f"/orgs/{quote(org, safe='-_.')}/repos" if org else "/user/repos"
            per_page = kwargs.get("per_page", 5)
            if type(per_page) is not int or not 1 <= per_page <= 10:
                return "Rejected: per_page must be between 1 and 10."
            params = {"per_page": per_page, "sort": "updated"}
        else:
            return "Rejected: this GitHub tool supports only get and list."
        if self.calls >= self.max_calls:
            return "GitHub read allowance is exhausted for this task."
        self.calls += 1
        try:
            async with self.client_factory(
                timeout=12,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.get(
                    "https://api.github.com" + path,
                    headers={
                        "Authorization": f"Bearer {self.token}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    params=params,
                )
            if response.status_code != 200:
                return f"GitHub read failed with HTTP {response.status_code}; details withheld."
            data = response.json()

            def summary(repo):
                if not isinstance(repo, dict):
                    return None
                return {
                    key: repo.get(source)
                    for key, source in (
                        ("name", "name"),
                        ("full_name", "full_name"),
                        ("description", "description"),
                        ("private", "private"),
                        ("default_branch", "default_branch"),
                        ("url", "html_url"),
                    )
                }

            if action == "list":
                if not isinstance(data, list):
                    raise ValueError("invalid repository list")
                payload = [summary(repo) for repo in data[:10]]
                return json.dumps({"repositories": payload, "count": len(payload)})
            repo = summary(data)
            if not repo or not isinstance(repo.get("full_name"), str):
                raise ValueError("invalid repository")
            return json.dumps({"repository": repo})
        except (httpx.RequestError, ValueError, TypeError):
            return "GitHub read failed; credentials and response details were withheld."


async def run(payload, protocol_stdout=None):
    import httpx
    from openai import AsyncOpenAI
    from src.type.tools import ToolRegistry

    chat_module = importlib.import_module("src.model.serving.chat")

    workspace = Path(payload["workspace"]).resolve(strict=True)
    if not workspace.is_dir() or not workspace.is_relative_to((Path.cwd() / "var").resolve()):
        raise ValueError("invalid task workspace")
    endpoint = payload["endpoint"]
    model = payload["model"]
    openrouter = endpoint == "https://openrouter.ai/api/v1/chat/completions" and (
        model.endswith(":free") or model == "openrouter/free"
    )
    dgx_qwen = (
        endpoint == "http://202.92.159.240:8001/v1/chat/completions" and model == "Qwen/Qwen3.8-27B"
    )
    groq = endpoint == "https://api.groq.com/openai/v1/chat/completions" and model in {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    }
    if not (openrouter or dgx_qwen or groq):
        raise ValueError(
            "Onit worker requires a documented university, allowlisted Groq, or free-only OpenRouter route"
        )
    effort = payload.get("reasoning_effort")
    streaming = payload.get("streaming", True)
    output_budget = payload.get("max_output_tokens", 512)
    if effort not in {None, "none", "low", "medium"} or (
        effort == "none" and model != "qwen/qwen3.8-27b"
    ):
        raise ValueError("unsupported reasoning effort")
    if (
        type(streaming) is not bool
        or type(output_budget) is not int
        or not 128 <= output_budget <= 4096
    ):
        raise ValueError("invalid streaming/output budget")
    key = _KEY or os.environ.pop("ME3_ONIT_KEY", "")
    if not key:
        raise ValueError("missing task credential")
    limit = payload["max_requests"]
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 8):
        raise ValueError("invalid request bound")
    timeout = payload["timeout"]
    if type(timeout) not in (int, float) or not 1 <= timeout <= 120:
        raise ValueError("invalid timeout")
    if not isinstance(payload["task"], str) or not 1 <= len(payload["task"]) <= 8000:
        raise ValueError("invalid task")
    tool_free = payload.get("tool_free", False)
    if type(tool_free) is not bool:
        raise ValueError("invalid tool-free setting")
    if tool_free and (
        payload.get("tavily_credit_budget")
        or payload.get("github_read_enabled")
        or payload.get("project_output_root")
    ):
        raise ValueError("tool-free probe cannot enable integrations")
    tavily_budget = payload.get("tavily_credit_budget", 0)
    if type(tavily_budget) is not int or not 0 <= tavily_budget <= 20:
        raise ValueError("invalid Tavily credit budget")
    github_read_enabled = payload.get("github_read_enabled", False)
    if type(github_read_enabled) is not bool:
        raise ValueError("invalid GitHub read setting")
    if tavily_budget and not _TAVILY_KEY:
        raise ValueError("Tavily is enabled but its server-side credential is missing")
    if github_read_enabled and not _GITHUB_TOKEN:
        raise ValueError("GitHub reading is enabled but its server-side credential is missing")
    from .onit_diagnostics import StageJournal, send_observed, ObservedBody

    class HTTPObservedBody(ObservedBody, httpx.AsyncByteStream):
        pass

    journal = StageJournal(workspace)
    journal.record("worker_received")
    receipt_checkpoints = payload.get("receipt_checkpoints", True)
    record_final_reply = payload.get("record_final_reply", False)
    if type(receipt_checkpoints) is not bool or type(record_final_reply) is not bool:
        raise ValueError("invalid result policy")
    requests = 0
    clients = []

    async def send_with_headers(*args, **kwargs):
        response = await send_observed(*args, **kwargs)
        meta = {
            "attempt": requests,
            "status": response.status_code,
            "rate_limits": {
                k: v
                for k, v in response.headers.items()
                if k.startswith("x-ratelimit-") or k == "retry-after"
            },
        }
        if not streaming and response.status_code == 200:
            try:
                completion = json.loads(await response.aread())
                meta["usage"] = completion.get("usage")
                meta["finish_reasons"] = [
                    c.get("finish_reason") for c in completion.get("choices", [])
                ]
            except (ValueError, TypeError):
                meta["completion_parse_error"] = True
        if response.status_code >= 400:
            try:
                error = json.loads(await response.aread()).get("error", {})
                meta["error"] = {
                    k: str(error.get(k, ""))[:1200].replace(key, "[redacted]")
                    for k in ("code", "type", "message")
                }
            except (ValueError, TypeError):
                meta["error"] = {"type": "non_json"}
        with (workspace / "response-metadata.jsonl").open("a") as stream:
            stream.write(json.dumps(meta) + "\n")
        return response

    class Guard(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request):
            nonlocal requests
            if (
                str(request.url) != endpoint
                or request.method != "POST"
                or (limit is not None and requests >= limit)
            ):
                raise RuntimeError("Onit task request bound or route rejected")
            body = json.loads(await request.aread())
            if body.get("model") != model:
                raise RuntimeError("Onit task model changed")
            if openrouter:
                body["provider"] = {
                    "max_price": {"prompt": 0, "completion": 0},
                    "require_parameters": True,
                    "allow_fallbacks": True,
                }
            if tool_free:
                body.pop("tools", None)
                body.pop("tool_choice", None)
            thinking = body.get("chat_template_kwargs", {}).get("enable_thinking")
            for field in (
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "chat_template_kwargs",
            ):
                body.pop(field, None)
            if dgx_qwen and type(thinking) is bool:
                # Documented Qwen/vLLM request control. Omitting it undoes think=False.
                body["chat_template_kwargs"] = {"enable_thinking": thinking}
            # Keep the established task-output budget. The 512-token probe
            # produced an incomplete tool argument and did not improve the
            # full project request; do not trade usable artifacts for a lower
            # but unproven latency setting.
            if groq:
                body["messages"] = groq_tool_results(body.get("messages", []))
                body["parallel_tool_calls"] = False
                if effort is not None:
                    body["reasoning_effort"] = effort
            if not streaming:
                body.pop("stream_options", None)
            token_cap = output_budget
            body["max_tokens"] = min(body.get("max_tokens", 512), token_cap)
            wire = json.dumps(body).encode()
            safe = httpx.Request(
                "POST",
                request.url,
                headers=request.headers,
                content=wire,
                extensions=dict(request.extensions),
            )
            safe.headers["Content-Length"] = str(len(wire))
            requests += 1
            # Payload-free request contract evidence; no message text or arguments.
            if requests <= 128:
                settings = {
                    "attempt": requests,
                    "tool_choice": body.get("tool_choice")
                    if isinstance(body.get("tool_choice"), str)
                    else "named"
                    if body.get("tool_choice")
                    else None,
                    "tool_names": [
                        t.get("function", {}).get("name") for t in body.get("tools", [])
                    ],
                    "reasoning_effort": body.get("reasoning_effort"),
                    "tool_schema_bytes": len(json.dumps(body.get("tools", [])).encode()),
                    "stream": body.get("stream"),
                    "max_tokens": body.get("max_tokens"),
                    "message_roles": [m.get("role") for m in body.get("messages", [])],
                    "message_characters": [
                        len(m.get("content", "")) if isinstance(m.get("content"), str) else None
                        for m in body.get("messages", [])
                    ],
                    "thinking_enabled": body.get("chat_template_kwargs", {}).get("enable_thinking"),
                    "read_timeout": safe.extensions.get("timeout", {}).get("read"),
                }
                settings_fd = os.open(
                    workspace / "request-settings.jsonl",
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                    0o600,
                )
                with os.fdopen(settings_fd, "w") as settings_file:
                    settings_file.write(json.dumps(settings) + "\n")
                    settings_file.flush()
                    os.fsync(settings_file.fileno())
            # Record attempts before transmission so a timeout/worker death does
            # not erase accounting. No prompt, response, headers or key is stored.
            import time

            audit = workspace / "request-attempts.jsonl"
            if requests > 128:
                return await send_with_headers(
                    self.inner,
                    safe,
                    journal,
                    requests,
                    wrapper=HTTPObservedBody,
                    tool_result=any(
                        m.get("role") == "tool"
                        for m in body.get("messages", [])
                        if isinstance(m, dict)
                    ),
                )
            descriptor = os.open(
                audit, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600
            )
            try:
                os.write(
                    descriptor,
                    (
                        json.dumps({"attempt": requests, "monotonic": time.monotonic()}) + "\n"
                    ).encode(),
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return await send_with_headers(
                self.inner,
                safe,
                journal,
                requests,
                wrapper=HTTPObservedBody,
                tool_result=any(
                    m.get("role") == "tool" for m in body.get("messages", []) if isinstance(m, dict)
                ),
            )

        async def aclose(self):
            await self.inner.aclose()

    def client(**kwargs):
        kwargs.update(
            max_retries=0,
            http_client=httpx.AsyncClient(transport=Guard(), trust_env=False, timeout=timeout),
        )
        instance = AsyncOpenAI(**kwargs)
        clients.append(instance)
        return instance

    registry = ToolRegistry()
    tool = ArtifactTool(workspace)
    project_output_root = payload.get("project_output_root")
    # With an explicitly approved project root, the project tool already
    # creates every requested artifact inside the new directory. Offering both
    # it and the isolated draft writer is redundant and makes the provider
    # compile a larger, nested tool schema for the common project task.
    if not tool_free and not project_output_root:
        registry.register(tool)
    tavily_tool = None
    if tavily_budget:
        tavily_tool = TavilySearchTool(_TAVILY_KEY, tavily_budget)
        registry.register(tavily_tool)
    github_tool = None
    if github_read_enabled:
        github_tool = GitHubReadTool(_GITHUB_TOKEN)
        registry.register(github_tool)
    project_tool = None
    if project_output_root:
        from .artifacts import approved_project_root

        output_root = approved_project_root(project_output_root)
        project_tool = ProjectDirectoryTool(output_root)
        registry.register(project_tool)
    if tool_free:
        instruction = "Reply without tools. User request: " + payload["task"]
    elif project_tool and not tavily_tool and not github_tool:
        instruction = (
            "You are a confined task agent. Use create_project_directory once for the requested "
            "project, with complete flat files in its files array. Do not run, test, or deploy it. "
            "User request: " + payload["task"]
        )
    else:
        instruction = (
            "You are Swift's confined task worker. "
            + SWIFT_ENGLISH_OUTPUT_POLICY
            + " Complete the user's request using only the available scoped tools. You have no shell, "
            "deployment, device, or general filesystem tools. Never claim those effects. "
            + (
                "You may use tavily_search for current facts. This is a basic web search; "
                "for weather, use the requested city and report the source/link and date. "
                "Do not describe it as a structured weather forecast API. "
                if tavily_tool
                else "Current web search is unavailable in this task. "
            )
            + (
                "You may use github_repo_read for repository get/list requests only; "
                "no repository writes are available. "
                if github_tool
                else "GitHub is unavailable in this task. "
            )
            + (
                "If asked to create a static website and the project tool is available, "
                "call create_project_directory once with the requested directory name and "
                "a files array containing index.html plus any small styles.css or script.js. "
                "Use only flat filenames and complete file contents. If asked only for an "
                "empty directory, call the same tool with name only. Do not claim the site "
                "was run, tested in a browser, or deployed. "
                if project_tool
                else "Project directory creation is unavailable in this task. "
            )
            + (
                "If asked to draft a website without a configured project output root, "
                "create a few files with write_artifact in the isolated task workspace and "
                "say they are drafts available to download, not a sandbox project. "
                if not project_tool
                else ""
            )
            + "User request: "
            + payload["task"]
        )
    original = chat_module.AsyncOpenAI
    chat_module.AsyncOpenAI = client
    started = time.monotonic()
    metrics = {}

    def emit_action_receipt():
        if protocol_stdout is None or not receipt_checkpoints:
            return
        is_project = bool(project_tool and project_tool.created)
        is_website = bool(is_project and project_tool.artifacts)
        artifacts = list(project_tool.artifacts) if is_website else list(tool.artifacts)
        projects = list(project_tool.created) if project_tool else []
        summary = (
            f"Created static website project {projects[0]} with {len(artifacts)} files; not run or deployed."
            if is_website
            else f"Created empty project directory {projects[0]} under the configured test-output root."
            if is_project
            else f"Created {len(artifacts)} draft file(s) in the isolated task workspace; nothing was deployed."
        )
        receipt = {
            "checkpoint": "action_completed",
            "mode": "local-project-site"
            if is_website
            else "local-project"
            if is_project
            else "local-artifact",
            "summary": summary,
            "artifacts": artifacts,
            "projects": projects,
            "answer": "The requested action completed and its result was verified.",
            "requests": requests,
            "tavily_credits_used": tavily_tool.credits_reserved if tavily_tool else 0,
            "github_reads_used": github_tool.calls if github_tool else 0,
            "worker_ms": round((time.monotonic() - started) * 1000, 3),
            "onit_metrics": {
                "turns": [
                    {
                        "ttft_ms": round(row["ttft_s"] * 1000, 3)
                        if isinstance(row.get("ttft_s"), (int, float))
                        else None,
                        "model_ms": round(row["model_s"] * 1000, 3)
                        if isinstance(row.get("model_s"), (int, float))
                        else None,
                        "completion_tokens": row["completion_tokens"]
                        if type(row.get("completion_tokens")) is int
                        and row["completion_tokens"] > 0
                        else None,
                    }
                    for row in metrics.get("turns", [])[:8]
                    if isinstance(row, dict)
                ]
            },
        }
        protocol_stdout.write(json.dumps(receipt, ensure_ascii=True) + "\n")
        protocol_stdout.flush()

    def action_completed():
        journal.record("tool_execution_complete")
        emit_action_receipt()

    tool.on_invoked = lambda: journal.record("tool_invoked", tool_name="write_artifact")
    tool.on_started = lambda: journal.record("tool_execution_started", tool_name="write_artifact")
    tool.on_completed = action_completed
    if project_tool:
        project_tool.on_invoked = lambda: journal.record(
            "tool_invoked", tool_name="create_project_directory"
        )
        project_tool.on_started = lambda: journal.record(
            "tool_execution_started", tool_name="create_project_directory"
        )
        project_tool.on_completed = action_completed
    try:
        with open(os.devnull, "w") as silent, contextlib.redirect_stdout(silent):
            answer = await chat_module.chat(
                host=endpoint.removesuffix("/chat/completions"),
                host_key=key,
                model=model,
                instruction=instruction,
                tool_registry=registry,
                timeout=timeout,
                safety_queue=asyncio.Queue(),
                stream=streaming,
                think=False,
                max_tokens=output_budget,
                max_context_tokens=16384,
                max_chat_iterations=limit or -1,
                max_api_retries=1,
                max_final_continuations=0,
                verify_answers=False,
                harness_tools=False,
                code_execution=False,
                result_store=False,
                data_path=str(workspace),
                metrics=metrics,
            )
    finally:
        chat_module.AsyncOpenAI = original
        for instance in clients:
            await instance.close()
    if record_final_reply:
        descriptor = os.open(
            workspace / "final-reply.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w") as output:
            json.dump({"answer": answer}, output, ensure_ascii=False)
    if (
        not tool.artifacts
        and not (project_tool and project_tool.created)
        and (not isinstance(answer, str) or not answer.strip())
    ):
        journal.record("worker_no_output")
        return {"error": "NoOutput", "requests": requests}
    journal.record("worker_complete")
    return {
        "final_reply_status": "received"
        if isinstance(answer, str) and answer.strip()
        else "missing",
        "mode": "local-project-site"
        if project_tool and project_tool.created and project_tool.artifacts
        else "local-project"
        if project_tool and project_tool.created
        else "local-artifact"
        if tool.artifacts
        else "local-answer",
        "summary": (
            f"Created static website project {project_tool.created[0]} with {len(project_tool.artifacts)} files; not run or deployed."
            if project_tool and project_tool.created and project_tool.artifacts
            else f"Created empty project directory {project_tool.created[0]} under the configured test-output root."
            if project_tool and project_tool.created
            else f"Created {len(tool.artifacts)} draft file(s) in the isolated task workspace; nothing was deployed."
            if tool.artifacts
            else "Read current web sources and returned a sourced answer; no local file or external state was changed."
            if tavily_tool and tavily_tool.credits_reserved
            else "Read GitHub repository metadata; no repository content was changed."
            if github_tool and github_tool.calls
            else "Onit answered the request; no external action was reported."
        ),
        "artifacts": project_tool.artifacts
        if project_tool and project_tool.created and project_tool.artifacts
        else tool.artifacts,
        "projects": project_tool.created if project_tool else [],
        "answer": answer[:1000] if isinstance(answer, str) else "",
        "requests": requests,
        "tavily_credits_used": tavily_tool.credits_reserved if tavily_tool else 0,
        "github_reads_used": github_tool.calls if github_tool else 0,
        "worker_ms": round((time.monotonic() - started) * 1000, 3),
        "onit_metrics": {
            "model_ms": round(metrics["model_s"] * 1000, 3)
            if isinstance(metrics.get("model_s"), (int, float))
            else None,
            "tool_ms": round(metrics["tool_s"] * 1000, 3)
            if isinstance(metrics.get("tool_s"), (int, float))
            else None,
            "turns": [
                {
                    "ttft_ms": round(row["ttft_s"] * 1000, 3)
                    if isinstance(row.get("ttft_s"), (int, float))
                    else None,
                    "model_ms": round(row["model_s"] * 1000, 3)
                    if isinstance(row.get("model_s"), (int, float))
                    else None,
                    "completion_tokens": row["completion_tokens"]
                    if type(row.get("completion_tokens")) is int and row["completion_tokens"] > 0
                    else None,
                }
                for row in metrics.get("turns", [])[:8]
                if isinstance(row, dict)
            ],
        },
    }


def main():
    global _KEY, _TAVILY_KEY, _GITHUB_TOKEN
    _KEY = os.environ.pop("ME3_ONIT_KEY", "")
    _TAVILY_KEY = os.environ.pop("ME3_ONIT_TAVILY_KEY", "")
    _GITHUB_TOKEN = os.environ.pop("ME3_ONIT_GITHUB_TOKEN", "")
    if not _KEY:
        raise SystemExit(1)
    # Pay the Onit/OpenAI import cost before any task is requested.
    prewarm_sdk()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for raw in sys.stdin.buffer:
        try:
            if len(raw) > 16384:
                raise ValueError("oversized task request")
            result = asyncio.run(run(json.loads(raw), protocol_stdout=sys.stdout))
        except Exception:
            result = {"error": "TaskFailed"}
        sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
