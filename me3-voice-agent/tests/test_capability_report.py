import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from me3_voice.capability_report import (
    build_capability_report,
    export_capability_report,
    render_capability_markdown,
)
from me3_voice.configuration import BackendSpec, load_backends, load_runtime


def test_profile_report_projects_registered_surfaces_and_redacts_credentials(monkeypatch):
    sentinel = "never-serialize-this-provider-secret"
    monkeypatch.setenv("ME3_GROQ_API_KEY", sentinel)
    backends = load_backends()
    backends["llm"] = replace(
        BackendSpec(),
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        auth_env="ME3_GROQ_API_KEY",
        model="openai/gpt-oss-120b",
        allow_live=True,
        request_limit=2,
        max_input_tokens=4000,
        max_input_tokens_total=12000,
    )
    options = load_runtime(engine="native")
    report = build_capability_report(
        backends=backends,
        options=options,
        credentials={"ME3_GROQ_API_KEY": sentinel},
        device_capabilities={"target": "mock", "status": "simulated", "available": True},
        music_capabilities={"configured": False, "output_backend": "mock"},
        wake_configured=False,
    )

    rows = report["capabilities"]
    pipecat = {row["name"]: row for row in rows if row["surface"] == "native_pipecat"}
    langgraph = {row["name"] for row in rows if row["surface"] == "langgraph_task"}
    asr_mcp = {row["name"] for row in rows if row["surface"] == "project_owned_asr_mcp"}
    tts_mcp = {row["name"] for row in rows if row["surface"] == "project_owned_tts_mcp"}

    assert pipecat["get_runtime_time"]["registered"] is True
    assert pipecat["web_search"]["registered"] is True
    assert pipecat["web_search"]["advertised"] is False  # gated by disabled research profile
    assert pipecat["run_device_action"]["supported_actions"]
    assert {"write_artifact", "read_artifact", "list_workspace"} <= langgraph
    assert {"asr_open", "asr_append", "asr_finish"} <= asr_mcp
    assert {"tts_open", "tts_next"} <= tts_mcp

    llm_tool = pipecat["get_runtime_time"]
    llm_credential = next(
        item for item in llm_tool["credential_requirements"]
        if item["name"] == "ME3_GROQ_API_KEY"
    )
    assert llm_credential["presence"] == "present_unvalidated"
    assert llm_tool["configured"] is True
    assert llm_tool["operationally_verified"] is False
    assert llm_credential["presence"] == "present_unvalidated"

    serialized = json.dumps(report)
    assert sentinel not in serialized
    assert "ME3_GROQ_API_KEY" in serialized
    assert '"credential_values_included": false' in serialized
    markdown = render_capability_markdown(report)
    assert "get_runtime_time" in markdown and "ME3_GROQ_API_KEY" in markdown
    assert sentinel not in markdown


def test_export_writes_deck_ready_json_and_markdown_exclusively():
    root = Path("var/scratch")
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"capability-report-{uuid4().hex}"
    directory.mkdir(mode=0o700)
    stem = directory / "profile"
    report = build_capability_report(
        backends=load_backends(),
        options=load_runtime(engine="native"),
        wake_configured=False,
    )
    json_path, markdown_path = export_capability_report(report, stem)
    assert json.loads(json_path.read_text()) == report
    assert "# Swift ME3 Capability Profile" in markdown_path.read_text()
    assert json_path.stat().st_mode & 0o077 == 0
    assert markdown_path.stat().st_mode & 0o077 == 0

    try:
        export_capability_report(report, stem)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing reports must not be overwritten")
    for path in (json_path, markdown_path):
        path.unlink()
    directory.rmdir()
