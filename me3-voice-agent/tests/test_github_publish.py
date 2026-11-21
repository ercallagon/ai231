"""Offline boundary checks for the optional GitHub task publisher."""

import json
import os
import importlib.util
import base64
from pathlib import Path
from urllib.parse import urljoin

import pytest

from me3_voice.github_publish import (
    ARTIFACT_BRANCH,
    GitHubPublishError,
    GitHubPublisher,
    GALLERY_TIMESTAMPS,
    load_github_writer,
    render_pages_index,
)


def test_incremental_github_setup_updates_token_and_repository_independently(tmp_path, monkeypatch):
    import me3_voice.credentials as credentials

    path = Path(__file__).resolve().parents[1] / "scripts/setup_github_write.py"
    spec = importlib.util.spec_from_file_location("swift_github_setup", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    cache = tmp_path / "var/cache"
    cache.mkdir(parents=True)
    monkeypatch.setattr(credentials, "CACHE_ROOT", cache)
    monkeypatch.setattr(setup, "CACHE", cache)
    monkeypatch.setattr(setup, "TOKEN_FILE", cache / "github-write-token.env")
    monkeypatch.setattr(setup, "CONFIG_FILE", cache / "github-write.json")
    monkeypatch.setattr(setup, "PROVIDER_FILE", cache / "selected-provider-credentials.env")

    setup._save_token("old-hidden-test-token")
    setup._save_repository(
        {
            "schema_version": 1,
            "owner": "demo-owner",
            "repository": "demo-site",
            "base_branch": "main",
        }
    )
    original_config = setup.CONFIG_FILE.read_bytes()

    setup._save_token("new-hidden-test-token", replace=True)
    assert setup.CONFIG_FILE.read_bytes() == original_config
    assert (
        credentials.load_secrets(setup.TOKEN_FILE, {"ME3_GITHUB_WRITE_TOKEN"})[
            "ME3_GITHUB_WRITE_TOKEN"
        ]
        == "new-hidden-test-token"
    )

    setup._save_repository(
        {
            "schema_version": 1,
            "owner": "demo-owner",
            "repository": "updated-site",
            "base_branch": "main",
        }
    )
    assert (
        credentials.load_secrets(setup.TOKEN_FILE, {"ME3_GITHUB_WRITE_TOKEN"})[
            "ME3_GITHUB_WRITE_TOKEN"
        ]
        == "new-hidden-test-token"
    )
    assert json.loads(setup.CONFIG_FILE.read_text())["repository"] == "updated-site"


def test_setup_provider_status_does_not_print_credential_values(capsys, monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/setup_github_write.py"
    spec = importlib.util.spec_from_file_location("swift_github_setup_status", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    secret = "must-never-be-printed"
    monkeypatch.setattr(
        setup,
        "load_secrets",
        lambda _path, keys: {
            key: secret for key in keys if key in set(setup.PROVIDER_KEYS.values())
        },
    )
    monkeypatch.setattr(setup, "PROVIDER_FILE", tmp_path / "credentials.env")

    setup._provider_status()

    output = capsys.readouterr().out
    assert "configured" in output
    assert secret not in output


def test_nonwebsite_artifacts_are_isolated_and_project_collisions_are_refused():
    from me3_voice.github_publish import project_artifact_paths

    mapped = project_artifact_paths(
        {"src/main.py": "print('hello')", "README.md": "project notes"}, "demo-app"
    )
    assert mapped == {
        "projects/demo-app/README.md": "project notes",
        "projects/demo-app/main.py": "print('hello')",
    }
    with pytest.raises(GitHubPublishError, match="github_project_path_collision"):
        project_artifact_paths({"src/a/config.json": "1", "src/b/config.json": "2"}, "demo-app")
    with pytest.raises(GitHubPublishError, match="github_project_path_invalid"):
        project_artifact_paths({"index.html": "x"}, "../other")


class Response:
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self, **_kwargs):
        return self.payload


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def request(self, method, url, json=None, **options):
        if url == "https://api.github.com/user/repos":
            assert options.get("allow_redirects") is False
        self.calls.append((method, url, json))
        status, payload = next(self.responses)
        return Response(status, payload)


@pytest.mark.parametrize("outcome", ["created", "duplicate", "wrong_account", "unconfirmed"])
async def test_create_public_repository_targets_only_authenticated_owner(outcome):
    url = "https://github.com/demo-owner/new-site"
    responses = [(200, {"login": "different" if outcome == "wrong_account" else "demo-owner"})]
    if outcome != "wrong_account":
        responses.append((200 if outcome == "duplicate" else 404, {}))
    if outcome in {"created", "unconfirmed"}:
        responses.append((201, {
            "full_name": "demo-owner/new-site", "private": False,
            "visibility": "public", "html_url": url if outcome == "created" else "https://example.com/",
        }))
    client = Client(responses)

    class ApiSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher("fixture-token", "demo-owner", "demo-repo", "main", session_factory=ApiSession)
    if outcome == "created":
        assert await publisher.create_public_repository("new-site") == {
            "status": "repository_created", "repository_url": url,
        }
    else:
        code = {
            "duplicate": "github_repository_exists", "wrong_account": "github_repository_owner_unverified",
            "unconfirmed": "github_repository_create_unverified",
        }[outcome]
        with pytest.raises(GitHubPublishError, match=code) as error:
            await publisher.create_public_repository("new-site")
        assert error.value.unknown == (outcome == "unconfirmed")
    attempted = [call for call in client.calls if call[0] != "GET"]
    assert attempted == ([
        ("POST", "https://api.github.com/user/repos", {"name": "new-site", "private": False, "auto_init": False})
    ] if outcome in {"created", "unconfirmed"} else [])
    assert publisher.repository_name == "demo-owner/demo-repo"


@pytest.mark.parametrize("matching", [True, False])
async def test_create_public_repository_checks_exact_readme_after_commit(matching):
    readme = "# AI weekly\n- [Source](https://example.com/week)\n"
    encoded = base64.b64encode(readme.encode()).decode()
    commit = "e" * 40
    url = "https://github.com/demo-owner/ai-weekly"
    client = Client([
        (200, {"login": "demo-owner"}),
        (404, {}),
        (201, {"full_name": "demo-owner/ai-weekly", "private": False, "visibility": "public", "html_url": url}),
        (201, {"commit": {"sha": commit}}),
        (200, {"encoding": "base64", "content": encoded if matching else "YXNkZg=="}),
    ])

    class ApiSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher("fixture-token", "demo-owner", "ai231", "main", session_factory=ApiSession)
    receipt = await publisher.create_public_repository("ai-weekly", readme=readme)
    assert receipt["repository_url"] == url
    assert receipt.get("readme_commit") == (commit if matching else None)
    assert receipt["status"] == ("repository_created" if matching else "repository_created_readme_unverified")
    assert client.calls[-2:] == [
        ("PUT", "https://api.github.com/repos/demo-owner/ai-weekly/contents/README.md", {
            "message": "Add sourced weekly AI briefing", "content": encoded,
        }),
        ("GET", "https://api.github.com/repos/demo-owner/ai-weekly/contents/README.md?ref=" + commit, None),
    ]
    assert publisher.repository_name == "demo-owner/ai231"


@pytest.mark.asyncio
@pytest.mark.parametrize(("private", "expected"), [(False, "public"), (True, "private")])
async def test_repository_visibility_is_read_from_authenticated_metadata(private, expected):
    client = Client([(200, {"private": private})])

    class ApiSession:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["headers"]["Authorization"] == "Bearer fixture-token"

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher(
        "fixture-token", "demo-owner", "demo-repo", "main", session_factory=ApiSession
    )

    assert await publisher.repository_visibility() == expected
    assert client.calls == [("GET", "https://api.github.com/repos/demo-owner/demo-repo", None)]


@pytest.mark.asyncio
async def test_repository_visibility_unknown_fails_before_any_approval_summary():
    client = Client([(200, {"name": "demo-repo"})])

    class ApiSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher(
        "fixture-token", "demo-owner", "demo-repo", "main", session_factory=ApiSession
    )

    with pytest.raises(GitHubPublishError, match="github_repository_visibility_unavailable"):
        await publisher.repository_visibility()


@pytest.mark.asyncio
async def test_github_publisher_creates_new_branch_from_task_files_without_network():
    commit_sha = "b" * 40
    tree_sha = "c" * 40
    client = Client(
        [
            (404, {"message": "branch missing"}),
            (201, {"sha": tree_sha}),
            (201, {"sha": commit_sha}),
            (201, {"ref": "refs/heads/" + ARTIFACT_BRANCH}),
        ]
    )

    class Session:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
            assert kwargs["headers"]["Authorization"] == "Bearer fixture-token"

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher(
        "fixture-token", "private-owner", "demo-repo", "main", session_factory=Session
    )
    result = await publisher.publish(
        "task_" + "1" * 24,
        "publish-000001",
        {"projects/demo-app/main.py": "print('task artifact')"},
        "Add task artifact",
    )

    assert result["status"] == "published"
    assert result["branch"] == ARTIFACT_BRANCH
    assert result["commit"] == commit_sha
    assert result["files"] == ["projects/demo-app/main.py"]
    assert len(client.calls) == 4
    assert "base_tree" not in client.calls[1][2]
    assert "parents" not in client.calls[2][2]
    assert client.calls[-1][2] == {
        "ref": "refs/heads/" + ARTIFACT_BRANCH,
        "sha": commit_sha,
    }


@pytest.mark.asyncio
async def test_github_publisher_rejects_unsafe_paths_before_remote_access():
    client = Client([])

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher(
        "fixture-token", "private-owner", "demo-repo", "main", session_factory=Session
    )
    with pytest.raises(GitHubPublishError, match="github_invalid_artifact_path"):
        await publisher.publish(
            "task_" + "1" * 24, "publish-000001", {"../outside.txt": "no"}, "bad path"
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_update_existing_file_appends_to_persistent_artifact_branch():
    base_commit = "a" * 40
    base_tree = "b" * 40
    new_tree = "c" * 40
    new_commit = "d" * 40
    client = Client(
        [
            (200, {"object": {"sha": base_commit}}),
            (200, {"sha": "e" * 40}),
            (200, {"tree": {"sha": base_tree}}),
            (201, {"sha": new_tree}),
            (201, {"sha": new_commit}),
            (200, {"ref": "refs/heads/" + ARTIFACT_BRANCH}),
        ]
    )

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher("fixture-token", "owner", "repo", "main", session_factory=Session)
    result = await publisher.publish(
        "task_" + "2" * 24,
        "update-000001",
        {"sites/one/index.html": "<h1>Revised</h1>"},
        "Update existing site",
        operation="update",
    )
    assert result["status"] == "published"
    assert result["branch"] == ARTIFACT_BRANCH
    assert client.calls[2][1].endswith("/git/commits/" + base_commit)
    assert client.calls[3][2]["base_tree"] == base_tree
    assert client.calls[3][2]["tree"] == [
        {"path": "sites/one/index.html", "mode": "100644", "type": "blob", "content": "<h1>Revised</h1>"}
    ]
    assert client.calls[4][2]["parents"] == [base_commit]
    assert client.calls[5][0] == "PATCH"


@pytest.mark.asyncio
async def test_create_mode_refuses_existing_path_before_creating_a_branch():
    client = Client([(200, {"object": {"sha": "a" * 40}}), (200, {"sha": "b" * 40})])

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher("fixture-token", "owner", "repo", "main", session_factory=Session)
    with pytest.raises(GitHubPublishError, match="github_create_path_exists"):
        await publisher.publish(
            "task_" + "3" * 24,
            "create-000001",
            {"me3-voice-agent/artifacts/site/index.html": "<h1>New site</h1>"},
            "Create site",
            operation="create",
        )
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_github_publisher_returns_pages_url_only_after_build_and_public_http_check():
    review_commit = "a" * 40
    page_parent = "b" * 40
    page_tree = "c" * 40
    page_commit = "d" * 40
    base_tree = "e" * 40
    gallery_commit = "9" * 40
    branch = GitHubPublisher.branch_for("task_" + "1" * 24, "publish-000002")
    pages_source = {"branch": "gh-pages", "path": "/"}
    existing_path = "me3-voice-agent/artifacts/existing-site"
    metadata_sha = "f" * 40
    stamp = base64.b64encode(json.dumps({existing_path: "2026-11-19T03:30:58Z"}).encode())
    client = Client(
        [
            (200, {"object": {"sha": review_commit}}),
            (200, {"source": pages_source}),
            (200, {"object": {"sha": page_parent}}),
            (200, {"tree": {"sha": base_tree}}),
            (200, {"tree": [{"path": existing_path + "/index.html"}, {"path": GALLERY_TIMESTAMPS, "sha": metadata_sha}]}),
            (200, {"encoding": "base64", "content": stamp.decode()}),
            (404, {}),
            (404, {}),
            (200, {"object": {"sha": page_parent}}),
            (200, {"tree": {"sha": base_tree}}),
            (201, {"sha": page_tree}),
            (201, {"sha": page_commit}),
            (200, {"ref": "refs/heads/gh-pages"}),
            (200, {"source": pages_source}),
            (200, {"status": "built", "commit": page_commit, "updated_at": "2026-11-20T19:04:00Z"}),
            (200, {"source": pages_source, "html_url": "https://private-owner.github.io/demo-repo/"}),
            (200, {"object": {"sha": page_commit}}),
            (200, {"tree": {"sha": page_tree}}),
            (201, {"sha": "8" * 40}),
            (201, {"sha": gallery_commit}),
            (200, {"ref": "refs/heads/gh-pages"}),
            (200, {"source": pages_source}),
            (200, {"status": "built", "commit": gallery_commit}),
            (200, {"source": pages_source, "html_url": "https://private-owner.github.io/demo-repo/"}),
        ]
    )

    class ApiSession:
        def __init__(self, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer fixture-token"
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    class PublicResponse:
        status = 200
        content_type = "text/html"
        url = (
            "https://private-owner.github.io/demo-repo/"
            "me3-voice-agent/artifacts/trials/nasa-artemis/"
        )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class PublicSession:
        def __init__(self, **kwargs):
            assert "headers" not in kwargs
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, **kwargs):
            assert url in {"https://private-owner.github.io/demo-repo/", PublicResponse.url}
            assert kwargs["allow_redirects"] is True
            response = PublicResponse()
            response.url = url
            return response


    publisher = GitHubPublisher(
        "fixture-token",
        "private-owner",
        "demo-repo",
        "main",
        session_factory=ApiSession,
        public_session_factory=PublicSession,
        pages_poll_interval=0,
    )
    files = {
            "me3-voice-agent/artifacts/trials/nasa-artemis/index.html": "<h1>NASA trial</h1>",
            "me3-voice-agent/artifacts/trials/nasa-artemis/styles.css": "body { color: navy; }",
        }
    result = await publisher.enable_pages_for_published_branch(
        branch,
        review_commit,
        "me3-voice-agent/artifacts/trials/nasa-artemis",
        files=files,
        task_id="task_" + "1" * 24,
        operation_id="publish-000002",
    )

    assert result["deployment_status"] == "verified"
    assert result["published_at_pht"] == "2026-11-21T03:04:00+08:00"
    assert result["deployment_url"] == (
        "https://private-owner.github.io/demo-repo/"
        "me3-voice-agent/artifacts/trials/nasa-artemis/"
    )
    assert result["branch"] == "gh-pages"
    assert result["review_branch"] == branch
    assert result["gallery_status"] == "verified"
    assert len(client.calls) == 24
    tree_call = client.calls[10]
    assert tree_call[2]["base_tree"] == base_tree
    assert {item["path"] for item in tree_call[2]["tree"]} == set(files)
    gallery_tree = client.calls[18][2]
    assert gallery_tree["base_tree"] == page_tree
    entries = {item["path"]: item["content"] for item in gallery_tree["tree"]}
    assert set(entries) == {"index.html", GALLERY_TIMESTAMPS}
    latest, archive = entries["index.html"].split("<details>")
    assert "Nasa Artemis" in latest and "Existing Site" in archive
    assert 'datetime="2026-11-19T11:30:58+08:00"' in archive
    assert json.loads(entries[GALLERY_TIMESTAMPS])[existing_path] == "2026-11-19T03:30:58Z"
    assert not any(method == "PUT" and url.endswith("/pages") for method, url, _ in client.calls)
    assert not any(method == "POST" and url.endswith("/pages") for method, url, _ in client.calls)


@pytest.mark.asyncio
async def test_github_pages_nested_site_requires_matching_index_path_before_remote_access():
    client = Client([])

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher(
        "fixture-token", "private-owner", "demo-repo", "main", session_factory=Session
    )
    with pytest.raises(GitHubPublishError, match="github_pages_path_mismatch"):
        await publisher.publish(
            "task_" + "1" * 24,
            "publish-000004",
            {"nested/index.html": "<h1>NASA trial</h1>"},
            "Add NASA trial site",
            deploy_pages=True,
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_github_pages_failure_never_returns_a_verified_deployment_url():
    review_commit = "a" * 40
    page_parent = "b" * 40
    page_commit = "d" * 40
    branch = GitHubPublisher.branch_for("task_" + "1" * 24, "publish-000003")
    pages_source = {"branch": "gh-pages", "path": "/"}
    client = Client(
        [
            (200, {"object": {"sha": review_commit}}),
            (200, {"source": pages_source}),
            (200, {"object": {"sha": page_parent}}),
            (200, {"tree": {"sha": "c" * 40}}),
            (200, {"tree": []}),
            (404, {}),
            (200, {"object": {"sha": page_parent}}),
            (200, {"tree": {"sha": "c" * 40}}),
            (201, {"sha": "e" * 40}),
            (201, {"sha": page_commit}),
            (200, {"ref": "refs/heads/gh-pages"}),
            (200, {"source": pages_source}),
            (200, {"status": "errored", "commit": page_commit}),
        ]
    )

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher(
        "fixture-token", "private-owner", "demo-repo", "main", session_factory=Session
    )
    with pytest.raises(GitHubPublishError, match="github_pages_build_failed"):
        await publisher.enable_pages_for_published_branch(
            branch,
            review_commit,
            "me3-voice-agent/artifacts/site",
            files={"me3-voice-agent/artifacts/site/index.html": "<h1>Site</h1>"},
            task_id="task_" + "1" * 24,
            operation_id="publish-000003",
        )


@pytest.mark.asyncio
async def test_first_pages_publication_uses_one_stable_source_and_nested_site_path():
    review_commit = "a" * 40
    deployed_commit = "b" * 40
    branch = "swift/" + "1" * 24 + "-trial"
    pages_source = {"branch": "gh-pages", "path": "/"}
    client = Client(
        [
            (200, {"object": {"sha": review_commit}}),
            (404, {"message": "Pages not configured"}),
            (404, {"message": "site path missing"}),
            (404, {"message": "branch missing"}),
            (201, {"sha": "c" * 40}),
            (201, {"sha": deployed_commit}),
            (201, {"ref": "refs/heads/gh-pages"}),
            (204, None),
            (200, {"source": pages_source}),
            (200, {"status": "built", "commit": deployed_commit}),
            (200, {"source": pages_source, "html_url": "https://owner.github.io/repo/"}),
        ]
    )

    class ApiSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    class PublicResponse:
        status = 200
        content_type = "text/html"
        url = "https://owner.github.io/repo/me3-voice-agent/artifacts/demo/"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class PublicSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def get(self, url, **_kwargs):
            assert url == "https://owner.github.io/repo/me3-voice-agent/artifacts/demo/"
            return PublicResponse()

    publisher = GitHubPublisher(
        "fixture-token",
        "owner",
        "repo",
        "main",
        session_factory=ApiSession,
        public_session_factory=PublicSession,
        pages_poll_interval=0,
    )
    result = await publisher.enable_pages_for_published_branch(
        branch,
        review_commit,
        "me3-voice-agent/artifacts/demo",
        files={"me3-voice-agent/artifacts/demo/index.html": "<h1>Demo</h1>"},
        task_id="task_" + "1" * 24,
        operation_id="first-site",
    )

    assert result["branch"] == "gh-pages"
    assert result["deployment_url"] == "https://owner.github.io/repo/me3-voice-agent/artifacts/demo/"
    page_config_calls = [call for call in client.calls if call[1].endswith("/pages")]
    assert [(method, payload) for method, _url, payload in page_config_calls] == [
        ("GET", None),
        ("POST", {"build_type": "legacy", "source": pages_source}),
        ("GET", None),
        ("GET", None),
    ]
    assert not any(method == "PUT" for method, _url, _payload in client.calls)
    root_commit_tree = next(
        payload for method, url, payload in client.calls
        if method == "POST" and url.endswith("/git/trees")
    )
    assert [row["path"] for row in root_commit_tree["tree"]] == [
        "me3-voice-agent/artifacts/demo/index.html"
    ]
    assert result["gallery_status"] == "timestamp_unavailable"


def test_pages_root_index_lists_only_sanitized_published_site_paths():
    latest = "me3-voice-agent/artifacts/games/tic-tac-toe-" + "c" * 24
    older = "me3-voice-agent/artifacts/demos/site-one"
    rendered = render_pages_index(
        [latest, older, "unpublished/draft", "me3-voice-agent/artifacts/pending"],
        published_at={
            latest: "2026-11-20T03:30:58Z",
            older: "2026-11-19T03:30:58Z",
            "unpublished/draft": "2026-11-21T03:30:58Z",
        },
    )
    featured, archive = rendered.split("<details>")
    assert "From conversation to live web projects" in featured
    assert ">Latest publication</h2>" in featured and "Tic Tac Toe</strong>" in featured
    assert "<summary>More published projects</summary>" in archive and "Site One" in archive
    assert "Other Trials" not in rendered and "Practice" not in rendered
    assert "2026-11-19 11:30:58 PHT" in archive
    assert "unpublished" not in rendered and "pending" not in rendered
    assert f'href="{latest}/"' in featured and f'href="{older}/"' in archive
    assert rendered.count(f'href="{latest}/"') == 1
    assert urljoin("https://owner.github.io/repo/", older + "/").startswith(
        "https://owner.github.io/repo/me3-voice-agent/"
    )


@pytest.mark.asyncio
async def test_pages_source_on_base_branch_is_refused_without_commit_or_source_switch():
    branch = "swift/" + "1" * 24 + "-trial"
    commit = "a" * 40
    client = Client(
        [
            (200, {"object": {"sha": commit}}),
            (200, {"source": {"branch": "main", "path": "/"}}),
        ]
    )

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher("fixture-token", "owner", "repo", "main", session_factory=Session)
    with pytest.raises(GitHubPublishError, match="github_pages_source_is_base_branch"):
        await publisher.enable_pages_for_published_branch(
            branch,
            commit,
            "me3-voice-agent/artifacts/site",
            files={"me3-voice-agent/artifacts/site/index.html": "<h1>Site</h1>"},
            task_id="task_" + "4" * 24,
            operation_id="base-source",
        )
    assert len(client.calls) == 2
    assert not any(method in {"POST", "PATCH", "PUT"} for method, _url, _payload in client.calls)


@pytest.mark.asyncio
async def test_repository_discovery_and_read_are_bounded_and_scoped():
    encoded = base64.b64encode(b"<h1>Existing site</h1>").decode()
    client = Client(
        [
            (200, [{"name": "index.html", "path": "sites/one/index.html", "type": "file"}]),
            (200, {"type": "file", "sha": "a" * 40, "content": encoded}),
        ]
    )

    class Session:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    publisher = GitHubPublisher("fixture-token", "owner", "repo", "main", session_factory=Session)
    listing = await publisher.list_repository_files("sites/one")
    content = await publisher.read_repository_file("sites/one/index.html")
    assert listing["entries"] == [{"name": "index.html", "path": "sites/one/index.html", "type": "file"}]
    assert content["content"] == "<h1>Existing site</h1>"
    assert "?ref=main" in client.calls[0][1]
    assert "?ref=main" in client.calls[1][1]


def test_github_writer_loads_only_private_config_and_selected_credential(tmp_path):
    cache = tmp_path / "var" / "cache"
    cache.mkdir(parents=True)
    config = cache / "github-write.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner": "private-owner",
                "repository": "demo-repo",
                "base_branch": "main",
            }
        )
    )
    os.chmod(config, 0o600)

    assert load_github_writer({}, project_root=tmp_path) is None
    writer = load_github_writer(
        {"ME3_GITHUB_WRITE_TOKEN": "fixture-token", "ME3_OPENROUTER_API_KEY": "unread"},
        project_root=tmp_path,
    )
    assert writer.repository_name == "private-owner/demo-repo"
    assert writer.client_state == "INITIALIZED_UNVERIFIED"

    os.chmod(config, 0o644)
    with pytest.raises(GitHubPublishError, match="github_config_permissions"):
        load_github_writer({"ME3_GITHUB_WRITE_TOKEN": "fixture-token"}, project_root=tmp_path)
