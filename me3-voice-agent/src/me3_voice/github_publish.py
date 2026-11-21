"""Opt-in, approval-gated publication of task artifacts to a configured GitHub repo."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo

import aiohttp


class GitHubPublishError(ValueError):
    def __init__(self, code, *, unknown=False):
        super().__init__(code)
        self.code = code
        self.unknown = unknown


OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")
SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,119}$")
MAX_FILES = 20
MAX_TOTAL_BYTES = 512 * 1024
ARTIFACT_BRANCH = "swift-artifacts"
GALLERY_TIMESTAMPS = "swift-gallery-published.json"


def _pages_published_at(value):
    """Use the matching successful build's GitHub timestamp when available."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.astimezone(ZoneInfo("Asia/Manila")).isoformat() if stamp.tzinfo else None
    except (ValueError, OverflowError):
        return None


def new_pages_artifact_paths(files, task_id):
    """Map a new local site to a unique Demo directory, retaining relative paths."""
    if not re.fullmatch(r"task_[a-f0-9]{24}", task_id):
        raise GitHubPublishError("github_pages_artifact_path_invalid")
    prefix = f"me3-voice-agent/artifacts/demos/{task_id[5:]}/"
    return {_path(prefix + _path(path)): content for path, content in files.items()}


def render_pages_index(site_paths, *, published_at=None):
    """Promote the newest verified Pages deployment and archive the rest."""
    paths = sorted({_path(path) for path in site_paths if isinstance(path, str)})
    verified = []
    for path in paths:
        if not path.startswith("me3-voice-agent/artifacts/"):
            continue
        stamp = _pages_published_at((published_at or {}).get(path))
        if stamp:
            verified.append((stamp, path))
    verified.sort(key=lambda entry: datetime.fromisoformat(entry[0]), reverse=True)

    def card(stamp, path):
        label = PurePosixPath(path).name.replace("-", " ").replace("_", " ").strip().title()
        href = "/".join(quote(part, safe="-._~") for part in path.split("/")) + "/"
        return (
            f'<a class="site-card" href="{html.escape(href, quote=True)}">'
            f'<strong>{html.escape(label)}</strong>'
            f'<time datetime="{stamp}">Published {stamp[:19].replace("T", " ")} PHT</time>'
            '<span>Visit site <span aria-hidden="true">↗</span></span></a>'
        )

    latest = archive = ""
    if verified:
        latest = '<section aria-labelledby="latest"><h2 id="latest">Latest</h2>' + card(*verified[0]) + "</section>"
    if len(verified) > 1:
        archive = (
            '<details><summary>Other Trials</summary>'
            + "".join(card(*entry) for entry in verified[1:])
            + "</details>"
        )
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark"><title>Published sites · Swift</title>'
        '<style>:root{color-scheme:light dark;--bg:#f4f7f3;--ink:#17231d;--card:#fff;--line:#dce5dc;--muted:#465b4d;--accent:#187a58}'
        '@media(prefers-color-scheme:dark){:root{--bg:#101713;--ink:#edf5ee;--card:#17211b;--line:#2b3a30;--muted:#adbbb1;--accent:#72d6a5}}'
        '*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:1rem/1.5 system-ui,sans-serif}'
        '.shell{max-width:940px;width:calc(100% - 40px);margin:auto}header{padding:24px 0;border-bottom:1px solid var(--line);font-weight:700}'
        'main{padding:clamp(36px,8vw,76px) 0}h1{max-width:760px;font-size:clamp(2.4rem,7vw,5rem);line-height:1;letter-spacing:-.045em;margin:0 0 1em}'
        'h2,summary{font-size:1.05rem;font-weight:700}h2{color:var(--accent)}'
        '.site-card{display:grid;gap:7px;color:inherit;text-decoration:none;padding:20px;margin:10px 0;border:1px solid var(--line);border-radius:14px;background:var(--card);overflow-wrap:anywhere}'
        'section .site-card strong{font-size:clamp(1.4rem,4vw,2.1rem)}time,.site-card span{color:var(--muted);font-size:.87rem}'
        'details{margin-top:28px;border-top:1px solid var(--line);padding-top:14px}summary{cursor:pointer}a:hover{border-color:var(--accent)}'
        ':focus-visible{outline:3px solid var(--accent);outline-offset:3px}</style></head>'
        '<body><header class="shell">Swift</header><main class="shell">'
        f'<h1>From conversation to live web projects</h1>{latest}{archive}</main></body></html>\n'
    )


def project_artifact_paths(files, project):
    """Place non-website files in one stable project directory on swift-artifacts."""
    if not isinstance(project, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", project):
        raise GitHubPublishError("github_project_path_invalid")
    output = {}
    for source, content in files.items():
        target = _path(f"projects/{project}/{PurePosixPath(_path(source)).name}")
        if target in output:
            raise GitHubPublishError("github_project_path_collision")
        output[target] = content
    return dict(sorted(output.items()))


def _branch(value):
    if (
        not isinstance(value, str)
        or not BRANCH_RE.fullmatch(value)
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or ".." in value
        or "@{" in value
        or any(part.endswith(".lock") for part in value.split("/"))
    ):
        raise GitHubPublishError("github_invalid_branch")
    return value


def _path(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or "\\" in value
        or value.startswith("/")
        or any(ord(c) < 32 for c in value)
    ):
        raise GitHubPublishError("github_invalid_artifact_path")
    path = PurePosixPath(value)
    if any(
        part in {"", ".", "..", ".git"} or not SEGMENT_RE.fullmatch(part) for part in path.parts
    ):
        raise GitHubPublishError("github_invalid_artifact_path")
    return path.as_posix()


def load_github_writer(credentials, *, project_root=None):
    """Read only this adapter's private config/key; absence disables the tool."""
    token = (credentials or {}).get("ME3_GITHUB_WRITE_TOKEN")
    root = Path(project_root or Path(__file__).resolve().parents[2])
    config_path = root / "var/cache/github-write.json"
    if not token or not config_path.exists():
        return None
    if config_path.is_symlink():
        raise GitHubPublishError("github_config_invalid")
    try:
        fd = os.open(config_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd) as stream:
            mode = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(mode.st_mode)
                or mode.st_uid != os.getuid()
                or mode.st_mode & 0o077
                or mode.st_size > 4096
            ):
                raise GitHubPublishError("github_config_permissions")
            config = json.load(stream)
    except GitHubPublishError:
        raise
    except (OSError, ValueError, UnicodeError):
        raise GitHubPublishError("github_config_invalid") from None
    if (
        not isinstance(config, dict)
        or set(config) != {"schema_version", "owner", "repository", "base_branch"}
        or config.get("schema_version") != 1
        or not isinstance(config.get("owner"), str)
        or not OWNER_RE.fullmatch(config["owner"])
        or not isinstance(config.get("repository"), str)
        or not REPO_RE.fullmatch(config["repository"])
    ):
        raise GitHubPublishError("github_config_invalid")
    base_branch = _branch(config.get("base_branch"))
    return GitHubPublisher(token, config["owner"], config["repository"], base_branch)


class GitHubPublisher:
    """Publish approved task artifacts on the persistent, isolated artifact branch."""

    def __init__(
        self,
        token,
        owner,
        repository,
        base_branch,
        *,
        session_factory=None,
        public_session_factory=None,
        pages_poll_attempts=12,
        pages_poll_interval=5,
    ):
        if not isinstance(token, str) or not 1 <= len(token) <= 4096:
            raise GitHubPublishError("github_credential_missing")
        self.token = token
        self.owner = owner
        self.repository = repository
        self.base_branch = _branch(base_branch)
        self.session_factory = session_factory
        self.public_session_factory = public_session_factory
        self.pages_poll_attempts = pages_poll_attempts
        self.pages_poll_interval = pages_poll_interval
        self.client_state = "INITIALIZED_UNVERIFIED"

    @property
    def repository_name(self):
        return f"{self.owner}/{self.repository}"

    async def repository_visibility(self):
        """Read verified repository visibility before asking for a write approval."""
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            status, repository = await self._request(client, "GET", "")
        if status != 200 or not isinstance(repository, dict) or type(repository.get("private")) is not bool:
            raise GitHubPublishError("github_repository_visibility_unavailable")
        return "private" if repository["private"] else "public"

    async def pages_configuration(self):
        """Read the current Pages source without changing repository settings."""
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            status, value = await self._request(client, "GET", "/pages")
        if status == 404:
            return {"configured": False, "branch": "gh-pages", "path": "/"}
        if status != 200 or not isinstance(value, dict):
            raise GitHubPublishError("github_pages_configuration_unavailable")
        source = value.get("source")
        if not isinstance(source, dict):
            raise GitHubPublishError("github_pages_source_unavailable")
        branch = _branch(source.get("branch"))
        path = source.get("path")
        if path not in {"/", "/docs"}:
            raise GitHubPublishError("github_pages_source_unsupported")
        return {"configured": True, "branch": branch, "path": path}

    async def list_repository_files(self, path="", *, ref=None):
        """Bounded, read-only path discovery in the configured repository."""
        if path:
            path = _path(path)
        branch = _branch(ref or self.base_branch)
        encoded = quote(path, safe="/")
        suffix = f"?ref={quote(branch, safe='/')}"
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            status, value = await self._request(client, "GET", "/contents/" + encoded + suffix)
        if status == 404:
            return {"path": path, "ref": branch, "exists": False, "entries": []}
        if status != 200:
            raise GitHubPublishError("github_repository_path_unavailable")
        entries = value if isinstance(value, list) else [value]
        if len(entries) > 100:
            raise GitHubPublishError("github_repository_listing_too_large")
        output = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") not in {"file", "dir"}:
                continue
            name = entry.get("name")
            item_path = entry.get("path")
            if not isinstance(name, str) or not isinstance(item_path, str):
                continue
            output.append(
                {"name": name[:120], "path": _path(item_path), "type": entry["type"]}
            )
        return {"path": path, "ref": branch, "exists": True, "entries": output}

    async def read_repository_file(self, path, *, ref=None):
        path = _path(path)
        branch = _branch(ref or self.base_branch)
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        target = "/contents/" + quote(path, safe="/") + "?ref=" + quote(branch, safe="/")
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            status, value = await self._request(client, "GET", target)
        if status == 404:
            raise GitHubPublishError("github_repository_file_missing")
        if status != 200 or not isinstance(value, dict) or value.get("type") != "file":
            raise GitHubPublishError("github_repository_file_unavailable")
        encoded = value.get("content")
        if not isinstance(encoded, str) or len(encoded) > MAX_TOTAL_BYTES * 2:
            raise GitHubPublishError("github_repository_file_too_large")
        try:
            content = base64.b64decode(encoded, validate=False).decode("utf-8")
        except (ValueError, UnicodeError):
            raise GitHubPublishError("github_repository_file_not_utf8") from None
        if len(content.encode("utf-8")) > MAX_TOTAL_BYTES:
            raise GitHubPublishError("github_repository_file_too_large")
        return {"path": path, "ref": branch, "sha": value.get("sha"), "content": content}

    async def existing_paths(self, paths, *, ref=None):
        branch = _branch(ref or self.base_branch)
        safe_paths = sorted({_path(path) for path in paths})
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        found = set()
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            for path in safe_paths:
                target = "/contents/" + quote(path, safe="/") + "?ref=" + quote(branch, safe="/")
                status, _ = await self._request(client, "GET", target)
                if status == 200:
                    found.add(path)
                elif status != 404:
                    raise GitHubPublishError("github_repository_path_unavailable")
        return sorted(found)

    def expected_pages_url(self, site_path=""):
        if site_path:
            site_path = _path(site_path)
        repository_slug = quote(self.repository.casefold(), safe="-._~")
        root = (
            f"https://{self.owner.casefold()}.github.io/"
            f"{repository_slug}/"
        )
        return root + (quote(site_path, safe="/-._~") + "/" if site_path else "")

    def _validate_files(self, files):
        if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
            raise GitHubPublishError("github_invalid_file_set")
        output = {}
        total = 0
        for name, content in files.items():
            safe_name = _path(name)
            if not isinstance(content, str):
                raise GitHubPublishError("github_invalid_file_content")
            encoded = content.encode("utf-8")
            total += len(encoded)
            if total > MAX_TOTAL_BYTES:
                raise GitHubPublishError("github_file_set_too_large")
            output[safe_name] = content
        return dict(sorted(output.items()))

    @staticmethod
    def branch_for(task_id, operation_id):
        if not re.fullmatch(r"task_[a-f0-9]{24}", task_id) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,80}", operation_id
        ):
            raise GitHubPublishError("github_invalid_operation")
        return _branch(f"swift/{task_id[5:]}-{operation_id[-12:]}")

    async def _request(self, client, method, path, payload=None):
        url = (
            "https://api.github.com/repos/"
            + quote(self.owner, safe="")
            + "/"
            + quote(self.repository, safe="")
            + path
        )
        try:
            async with client.request(method, url, json=payload) as response:
                try:
                    value = await response.json(content_type=None)
                except (ValueError, aiohttp.ContentTypeError):
                    value = None
                return response.status, value
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise GitHubPublishError(
                "github_outcome_unknown"
                if method in {"POST", "PATCH", "PUT"}
                else "github_unavailable",
                unknown=method in {"POST", "PATCH", "PUT"},
            ) from None

    async def _enable_and_verify_pages(
        self, client, branch, commit_sha, site_path="", *, configure=False, source_path="/"
    ):
        pages_source = {"branch": branch, "path": source_path}
        status, current = await self._request(client, "GET", "/pages")
        if status == 404:
            if not configure:
                raise GitHubPublishError("github_pages_not_configured")
            status, current = await self._request(
                client,
                "POST",
                "/pages",
                {"build_type": "legacy", "source": pages_source},
            )
        elif status == 200:
            source = current.get("source") if isinstance(current, dict) else None
            if source != pages_source:
                raise GitHubPublishError("github_pages_source_changed_refused")
        if status not in {200, 201, 202, 204}:
            raise GitHubPublishError("github_pages_configuration_failed", unknown=status >= 500)

        build = None
        for attempt in range(max(1, self.pages_poll_attempts)):
            status, build = await self._request(client, "GET", "/pages/builds/latest")
            if status != 200 or not isinstance(build, dict):
                raise GitHubPublishError("github_pages_build_unavailable")
            if build.get("status") == "errored":
                raise GitHubPublishError("github_pages_build_failed")
            if build.get("status") == "built" and build.get("commit") == commit_sha:
                break
            if attempt + 1 < max(1, self.pages_poll_attempts):
                await asyncio.sleep(self.pages_poll_interval)
        else:
            raise GitHubPublishError("github_pages_build_pending")

        status, site = await self._request(client, "GET", "/pages")
        source = site.get("source") if status == 200 and isinstance(site, dict) else None
        url = site.get("html_url") if isinstance(site, dict) else None
        parsed = urlsplit(url) if isinstance(url, str) else None
        if (
            source != pages_source
            or parsed is None
            or parsed.scheme != "https"
            or not parsed.hostname
            or not (parsed.hostname == "github.io" or parsed.hostname.endswith(".github.io"))
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise GitHubPublishError("github_pages_url_unverified")

        if site_path:
            try:
                site_path = _path(site_path)
            except GitHubPublishError:
                raise GitHubPublishError("github_pages_path_invalid") from None
            url = url.rstrip("/") + "/" + quote(site_path, safe="/-._~") + "/"

        factory = self.public_session_factory or aiohttp.ClientSession
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        try:
            async with factory(timeout=timeout, trust_env=False) as public_client:
                async with public_client.get(url, allow_redirects=True) as response:
                    final = urlsplit(str(response.url))
                    if (
                        response.status != 200
                        or response.content_type != "text/html"
                        or final.scheme != "https"
                        or not final.hostname
                        or not (
                            final.hostname == "github.io"
                            or final.hostname.endswith(".github.io")
                        )
                        or final.hostname != parsed.hostname
                        or final.path != urlsplit(url).path
                    ):
                        raise GitHubPublishError("github_pages_public_url_unverified")
        except GitHubPublishError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise GitHubPublishError("github_pages_public_url_unavailable") from None
        return url, _pages_published_at(build.get("updated_at"))

    async def enable_pages_for_published_branch(
        self,
        branch,
        commit_sha,
        site_path="",
        *,
        files=None,
        task_id=None,
        operation_id=None,
        operation="create",
        expected_source=None,
    ):
        """Deploy reviewed files into one stable Pages source; never switch an existing source."""
        branch = _branch(branch)
        if not isinstance(commit_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", commit_sha):
            raise GitHubPublishError("github_invalid_commit")
        if not files or not task_id or not operation_id:
            raise GitHubPublishError("github_pages_files_required")
        site_path = _path(site_path) if site_path else ""
        if not site_path.startswith("me3-voice-agent/artifacts/"):
            raise GitHubPublishError("github_pages_artifact_path_invalid")
        files = self._validate_files(files)
        if operation not in {"create", "update", "push"}:
            raise GitHubPublishError("github_invalid_operation")
        if not any(name == (site_path + "/" if site_path else "") + "index.html" for name in files):
            raise GitHubPublishError("github_pages_requires_one_index_html")
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            branch_path = quote(branch, safe="/")
            status, ref = await self._request(client, "GET", f"/git/ref/heads/{branch_path}")
            actual_commit = ref.get("object", {}).get("sha") if status == 200 and isinstance(ref, dict) else None
            if actual_commit != commit_sha:
                raise GitHubPublishError("github_pages_branch_commit_mismatch")
            status, pages = await self._request(client, "GET", "/pages")
            configure = status == 404
            if status not in {200, 404}:
                raise GitHubPublishError("github_pages_configuration_unavailable")
            source = pages.get("source") if status == 200 and isinstance(pages, dict) else None
            source_branch = _branch(source.get("branch")) if isinstance(source, dict) else "gh-pages"
            source_folder = source.get("path", "/") if isinstance(source, dict) else "/"
            if source_folder not in {"/", "/docs"}:
                raise GitHubPublishError("github_pages_source_unsupported")
            if configure:
                source_branch = "gh-pages"
                source_folder = "/"
            if expected_source is not None and expected_source != {
                "configured": not configure, "branch": source_branch, "path": source_folder
            }:
                raise GitHubPublishError("github_pages_source_changed")
            if source_branch == self.base_branch:
                raise GitHubPublishError("github_pages_source_is_base_branch")
            prefix = "docs/" if source_folder == "/docs" else ""
            deployed_files = {(prefix + name): content for name, content in files.items()}
            gallery_parent, site_paths, stamps = None, set(), {}
            if not configure:
                gallery_parent, site_paths, stamps = await self._pages_site_paths(
                    client, source_branch, prefix
                )
            site_deployed_paths = {prefix + name for name in files}
            existing_paths = set()
            for name in site_deployed_paths:
                target = (
                    "/contents/"
                    + quote(name, safe="/")
                    + "?ref="
                    + quote(source_branch, safe="/")
                )
                path_status, _ = await self._request(client, "GET", target)
                if path_status == 200:
                    existing_paths.add(name)
                elif path_status != 404:
                    raise GitHubPublishError("github_pages_path_unavailable")
            if operation == "create" and existing_paths:
                raise GitHubPublishError("github_pages_path_exists")
            if operation == "update":
                required_index = prefix + site_path + "/index.html"
                if required_index not in existing_paths or any(
                    not name.startswith(prefix + site_path + "/") for name in site_deployed_paths
                ):
                    raise GitHubPublishError("github_pages_update_target_missing")
            page_commit = await self._commit_files_to_branch(
                client,
                source_branch,
                deployed_files,
                f"Publish site {site_path or 'root'} from Swift",
                create_branch=configure,
                task_id=task_id,
                operation_id=operation_id,
                expected_parent=gallery_parent,
            )
            if configure:
                status, _ = await self._request(
                    client,
                    "POST",
                    "/pages",
                    {"build_type": "legacy", "source": {"branch": "gh-pages", "path": "/"}},
                )
                if status not in {200, 201, 202, 204}:
                    raise GitHubPublishError("github_pages_configuration_failed", unknown=status >= 500)
            url, published_at = await self._enable_and_verify_pages(
                client,
                source_branch,
                page_commit,
                site_path,
                configure=False,
                source_path=source_folder,
            )
            gallery_status = "timestamp_unavailable"
            if published_at:
                site_paths.add(site_path)
                # Refresh the index only after the site build and HTTPS checks pass.
                stamps[site_path] = published_at
                try:
                    gallery_commit = await self._commit_files_to_branch(
                        client,
                        source_branch,
                        {
                            prefix + "index.html": render_pages_index(site_paths, published_at=stamps),
                            prefix + GALLERY_TIMESTAMPS: json.dumps(stamps, sort_keys=True) + "\n",
                        },
                        "Promote verified Swift publication",
                        expected_parent=page_commit,
                        task_id=task_id,
                        operation_id=operation_id,
                    )
                    await self._enable_and_verify_pages(
                        client, source_branch, gallery_commit, source_path=source_folder
                    )
                    gallery_status = "verified"
                except GitHubPublishError as exc:
                    gallery_status = "unknown" if exc.unknown else exc.code
        return {
            "status": "deployed",
            "repository": self.repository_name,
            "branch": source_branch,
            "commit": page_commit,
            "review_branch": branch,
            "review_commit": commit_sha,
            "deployment_url": url,
            "deployment_status": "verified",
            "gallery_status": gallery_status,
            **({"published_at_pht": published_at} if published_at else {}),
        }

    async def _commit_files_to_branch(
        self, client, branch, files, message, *, create_branch=False, task_id=None, operation_id=None, expected_parent=None
    ):
        branch_path = quote(branch, safe="/")
        status, ref = await self._request(client, "GET", f"/git/ref/heads/{branch_path}")
        if status == 404:
            if not create_branch:
                raise GitHubPublishError("github_pages_source_branch_missing")
            parent = None
            base_tree = None
        elif status == 200 and isinstance(ref, dict):
            parent = ref.get("object", {}).get("sha")
            if not isinstance(parent, str) or not re.fullmatch(r"[a-f0-9]{40}", parent):
                raise GitHubPublishError("github_pages_source_commit_invalid")
            if expected_parent is not None and parent != expected_parent:
                raise GitHubPublishError("github_pages_gallery_source_changed")
            status, commit = await self._request(client, "GET", f"/git/commits/{parent}")
            if status != 200 or not isinstance(commit, dict):
                raise GitHubPublishError("github_pages_source_commit_unavailable")
            base_tree = commit.get("tree", {}).get("sha")
            if not isinstance(base_tree, str) or not re.fullmatch(r"[a-f0-9]{40}", base_tree):
                raise GitHubPublishError("github_pages_source_tree_invalid")
        else:
            raise GitHubPublishError("github_pages_source_branch_unavailable")
        tree = [
            {"path": name, "mode": "100644", "type": "blob", "content": content}
            for name, content in files.items()
        ]
        payload = {"tree": tree}
        if base_tree:
            payload["base_tree"] = base_tree
        status, result = await self._request(client, "POST", "/git/trees", payload)
        if status not in {200, 201} or not isinstance(result, dict):
            raise GitHubPublishError("github_pages_tree_create_failed", unknown=status >= 500)
        tree_sha = result.get("sha")
        if not isinstance(tree_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", tree_sha):
            raise GitHubPublishError("github_pages_tree_invalid")
        commit_payload = {"message": message[:120], "tree": tree_sha}
        if parent:
            commit_payload["parents"] = [parent]
        status, result = await self._request(client, "POST", "/git/commits", commit_payload)
        if status not in {200, 201} or not isinstance(result, dict):
            raise GitHubPublishError("github_pages_commit_create_failed", unknown=status >= 500)
        commit_sha = result.get("sha")
        if not isinstance(commit_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", commit_sha):
            raise GitHubPublishError("github_pages_commit_invalid")
        if parent:
            status, _ = await self._request(
                client,
                "PATCH",
                f"/git/refs/heads/{branch_path}",
                {"sha": commit_sha, "force": False},
            )
        else:
            status, _ = await self._request(
                client,
                "POST",
                "/git/refs",
                {"ref": "refs/heads/" + branch, "sha": commit_sha},
            )
        if status not in {200, 201}:
            raise GitHubPublishError("github_pages_publish_outcome_unknown" if status >= 500 else "github_pages_publish_failed", unknown=status >= 500)
        return commit_sha

    async def _pages_site_paths(self, client, branch, source_prefix):
        branch_path = quote(branch, safe="/")
        status, ref = await self._request(client, "GET", f"/git/ref/heads/{branch_path}")
        if status == 404:
            return None, set(), {}
        if status != 200 or not isinstance(ref, dict):
            raise GitHubPublishError("github_pages_source_unavailable")
        parent = ref.get("object", {}).get("sha")
        if not isinstance(parent, str) or not re.fullmatch(r"[a-f0-9]{40}", parent):
            raise GitHubPublishError("github_pages_source_commit_invalid")
        status, commit = await self._request(client, "GET", f"/git/commits/{parent}")
        tree_sha = commit.get("tree", {}).get("sha") if status == 200 and isinstance(commit, dict) else None
        if not isinstance(tree_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", tree_sha):
            raise GitHubPublishError("github_pages_source_tree_invalid")
        status, tree = await self._request(client, "GET", f"/git/trees/{tree_sha}?recursive=1")
        entries = tree.get("tree") if status == 200 and isinstance(tree, dict) else None
        if not isinstance(entries, list) or len(entries) > 10000 or tree.get("truncated") is True:
            raise GitHubPublishError("github_pages_source_tree_unavailable")
        prefix = source_prefix + "me3-voice-agent/artifacts/"
        output = set()
        stamps_blob = None
        for entry in entries:
            path = entry.get("path") if isinstance(entry, dict) else None
            if isinstance(path, str) and path.startswith(prefix) and path.endswith("/index.html"):
                output.add(path[len(source_prefix) : -len("/index.html")])
            elif path == source_prefix + GALLERY_TIMESTAMPS:
                stamps_blob = entry.get("sha")
        if stamps_blob is None:
            return parent, output, {}
        if not isinstance(stamps_blob, str) or not re.fullmatch(r"[a-f0-9]{40}", stamps_blob):
            raise GitHubPublishError("github_pages_gallery_timestamps_invalid")
        status, blob = await self._request(client, "GET", f"/git/blobs/{stamps_blob}")
        if status != 200 or not isinstance(blob, dict) or blob.get("encoding") != "base64":
            raise GitHubPublishError("github_pages_gallery_timestamps_unavailable")
        try:
            encoded = blob["content"]
            if not isinstance(encoded, str) or len(encoded) > 48_000:
                raise ValueError
            stamps = json.loads(base64.b64decode("".join(encoded.split()), validate=True))
            if not isinstance(stamps, dict):
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise GitHubPublishError("github_pages_gallery_timestamps_invalid") from None
        return parent, output, {path: stamps[path] for path in output if path in stamps}

    async def publish(
        self,
        task_id,
        operation_id,
        files,
        message,
        *,
        deploy_pages=False,
        pages_site_path="",
        operation="create",
    ):
        files = self._validate_files(files)
        if operation not in {"create", "update", "push"}:
            raise GitHubPublishError("github_invalid_operation")
        if deploy_pages:
            index_paths = sorted(
                name for name in files if name.rsplit("/", 1)[-1].lower() == "index.html"
            )
            expected_site_path = (
                index_paths[0].rsplit("/", 1)[0] if len(index_paths) == 1 and "/" in index_paths[0] else ""
            )
            if len(index_paths) != 1:
                raise GitHubPublishError("github_pages_requires_one_index_html")
            if pages_site_path != expected_site_path:
                raise GitHubPublishError("github_pages_path_mismatch")
            if not pages_site_path.startswith("me3-voice-agent/artifacts/"):
                raise GitHubPublishError("github_pages_artifact_path_invalid")
        branch = ARTIFACT_BRANCH
        if (
            not isinstance(message, str)
            or not 1 <= len(message.strip()) <= 120
            or any(ord(c) < 32 and c not in "\t" for c in message)
        ):
            raise GitHubPublishError("github_invalid_commit_message")
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        factory = self.session_factory or aiohttp.ClientSession
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with factory(timeout=timeout, headers=headers, trust_env=False) as client:
            branch_path = quote(branch, safe="/")
            branch_status, branch_ref = await self._request(
                client, "GET", f"/git/ref/heads/{branch_path}"
            )
            if branch_status not in {200, 404}:
                raise GitHubPublishError("github_artifact_branch_unavailable")
            branch_exists = branch_status == 200
            existing_paths = set()
            for name in files:
                if not branch_exists:
                    # The artifact branch is intentionally an orphan, independent of main.
                    path_status = 404
                else:
                    target = "/contents/" + quote(name, safe="/") + "?ref=" + quote(branch, safe="/")
                    path_status, _ = await self._request(client, "GET", target)
                if path_status not in {200, 404}:
                    raise GitHubPublishError("github_repository_path_unavailable")
                if path_status == 200:
                    existing_paths.add(name)
                if operation == "create" and path_status == 200:
                    raise GitHubPublishError("github_create_path_exists")
                if operation == "update" and path_status == 404 and not deploy_pages:
                    raise GitHubPublishError("github_update_path_missing")
            if operation == "update" and deploy_pages:
                index_path = pages_site_path + "/index.html"
                if index_path not in existing_paths or any(
                    not name.startswith(pages_site_path + "/") for name in files
                ):
                    raise GitHubPublishError("github_update_site_missing")
            if branch_exists:
                base_commit = branch_ref.get("object", {}).get("sha") if isinstance(branch_ref, dict) else None
                if not isinstance(base_commit, str) or not re.fullmatch(r"[a-f0-9]{40}", base_commit):
                    raise GitHubPublishError("github_invalid_base_commit")
                status, commit = await self._request(client, "GET", f"/git/commits/{base_commit}")
                if status != 200 or not isinstance(commit, dict):
                    raise GitHubPublishError("github_base_commit_unavailable")
                base_tree = commit.get("tree", {}).get("sha")
                if not isinstance(base_tree, str) or not re.fullmatch(r"[a-f0-9]{40}", base_tree):
                    raise GitHubPublishError("github_invalid_base_tree")
            else:
                # Keep generated files on an orphan branch; never copy or alter main.
                base_commit = None
                base_tree = None
            tree = [
                {"path": name, "mode": "100644", "type": "blob", "content": content}
                for name, content in files.items()
            ]
            status, created_tree = await self._request(
                client,
                "POST",
                "/git/trees",
                ({"base_tree": base_tree} if base_tree else {}) | {"tree": tree},
            )
            if status not in {200, 201} or not isinstance(created_tree, dict):
                raise GitHubPublishError("github_tree_create_failed")
            tree_sha = created_tree.get("sha")
            if not isinstance(tree_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", tree_sha):
                raise GitHubPublishError("github_invalid_tree")
            commit_payload = {"message": message.strip(), "tree": tree_sha}
            if base_commit:
                commit_payload["parents"] = [base_commit]
            status, created_commit = await self._request(client, "POST", "/git/commits", commit_payload)
            if status not in {200, 201} or not isinstance(created_commit, dict):
                raise GitHubPublishError("github_commit_create_failed")
            commit_sha = created_commit.get("sha")
            if not isinstance(commit_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", commit_sha):
                raise GitHubPublishError("github_invalid_commit")
            status, created_ref = await self._request(
                client,
                "PATCH" if branch_exists else "POST",
                f"/git/refs/heads/{branch_path}" if branch_exists else "/git/refs",
                {"sha": commit_sha, "force": False}
                if branch_exists
                else {"ref": "refs/heads/" + branch, "sha": commit_sha},
            )
            if status not in {200, 201} or not isinstance(created_ref, dict):
                raise GitHubPublishError(
                    "github_publish_outcome_unknown"
                    if status >= 500
                    else "github_publish_ref_failed",
                    unknown=status >= 500,
                )
            result = {
                "status": "published",
                "repository": self.repository_name,
                "branch": branch,
                "commit": commit_sha,
                "files": list(files),
                "url": "https://github.com/"
                + self.repository_name
                + "/tree/"
                + quote(branch, safe="/"),
            }
            if deploy_pages:
                deployment = await self.enable_pages_for_published_branch(
                    branch,
                    commit_sha,
                    pages_site_path,
                    files=files,
                    task_id=task_id,
                    operation_id=operation_id,
                    operation=operation,
                )
                result["deployment_url"] = deployment["deployment_url"]
                result["deployment_status"] = "verified"
            return result
