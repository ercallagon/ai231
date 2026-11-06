"""Opt-in live Session/Onit project and dashboard readback; inspect code separately before execution."""

import asyncio
import argparse
import hashlib
import json
import os
import time
from argparse import Namespace
from pathlib import Path
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright
import me3_voice.voice_bridge as bridge
from me3_voice.configuration import load_backends
from me3_voice.credentials import load_secrets
from me3_voice.onit_tasks import OnitTaskExecutor, OnitPrimaryReasoning
from me3_voice.runtime import AudioApplication

ROOT = Path.cwd()
parser = argparse.ArgumentParser(
    description="Opt-in live Onit project validation; no microphone or physical effects."
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--max-requests", type=int)
parser.add_argument("--allow-live", action="store_true", required=True)
parser.add_argument("--backend-config", type=Path, default=Path("var/cache/voice-backends.json"))
parser.add_argument("--secrets-file", type=Path, default=Path("var/cache/dgx3-llm.env"))
parser.add_argument("--task-file", type=Path)
parser.add_argument(
    "--onit-python", type=Path, default=Path("var/cache/onit-venv/bin/python").resolve()
)
parser.add_argument("--reasoning-effort", choices=["none", "low", "medium"])
parser.add_argument("--no-stream", action="store_true")
parser.add_argument("--max-output-tokens", type=int, default=512)
args = parser.parse_args()
BASE = args.output_dir.resolve()
if not BASE.is_relative_to((ROOT / "var").resolve()):
    raise ValueError("Evidence must remain under var/")
CAP = args.max_requests
PROJECT_NAME = "swift-validation"
TASK = (
    "Create a new project named swift-validation in the configured output root. Create index.html with browser title "
    '"Swift validation", visible heading "ME3 output check", and a clearly labelled button that increments a visible '
    "counter starting at 0. Use responsive styling and embedded JavaScript. Do not use external dependencies or network "
    "requests. Do not install, run, or deploy it; briefly report the files created."
)

if args.task_file:
    TASK = args.task_file.read_text()


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        yield b"\0\0" * 128


async def main():
    BASE.mkdir(mode=0o700)  # Exclusive destination: never overwrite/replay an earlier trial.
    project_root = BASE / "projects"
    project_root.mkdir(mode=0o700)
    assert not (project_root / PROJECT_NAME).exists()
    report = {
        "evidence": "canonical OnitPrimaryReasoning/Session/executor/installed Onit/university/tools; transcript injection; mock speech only; actual temporary bridge/dashboard; fake zero capture; no physical microphone",
        "task": TASK,
        "cap": CAP,
        "provider_timeout_s": 45,
        "session_timeout_s": None,
        "model": "Qwen/Qwen3.8-27B",
        "project_root": str(project_root),
        "status": "started",
    }
    spec = load_backends(args.backend_config)["llm"]
    report.update(
        model=spec.model,
        endpoint=spec.endpoint,
        reasoning_effort=args.reasoning_effort,
        streaming=not args.no_stream,
        max_output_tokens=args.max_output_tokens,
    )
    assert spec.timeout_seconds == 45
    effective_cap = spec.request_limit if CAP is None else CAP
    report["effective_request_guard"] = effective_cap
    keys = load_secrets(args.secrets_file, {spec.auth_env})
    tasks = OnitTaskExecutor(
        args.onit_python,
        spec,
        keys,
        root=BASE / "workspace",
        project_output_root=project_root,
        max_requests=effective_cap,
        await_final_reply=True,
        reasoning_effort=args.reasoning_effort,
        streaming=not args.no_stream,
        max_output_tokens=args.max_output_tokens,
    )
    holder = {}

    def capture(*args, **kwargs):
        result = AudioApplication(*args, **kwargs)
        holder["app"] = result
        return result

    bridge.AudioApplication = capture
    bridge_args = Namespace(
        kws_assets=ROOT / "var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
        keywords_file=ROOT / "var/cache/wake/hey-swift-keywords.txt",
        kws_threshold=0.25,
        kws_boost=1.0,
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        wake_threshold=0.5,
    )
    os.environ.pop("ME3_MUSIC_CONFIG", None)
    bridge_backends = load_backends(None)
    bridge_backends["llm"] = spec
    app = bridge.build_app(ROOT, bridge_backends, {}, bridge_args)
    handler = next(
        r.handler
        for r in app.router.routes()
        if r.resource.canonical == "/api/voice/ws" and r.method == "GET"
    )
    services = next(
        c.cell_contents
        for c in handler.__closure__
        if isinstance(c.cell_contents, bridge.VoiceRuntimeServices)
    )
    services.tasks = tasks
    services.devices = tasks.mock.actions

    async def open_services():
        return Speech(), OnitPrimaryReasoning(), tasks

    services.open = open_services
    browser = None
    s = None
    try:
        tasks.prewarm()
        await tasks.startup
        report["startup_ms"] = tasks.startup_ms
        async with TestServer(app, host="127.0.0.1") as server, async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(
                viewport={"width": 1280, "height": 720}, reduced_motion="reduce"
            )
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.add_init_script(
                """navigator.mediaDevices.getUserMedia=async()=>{const c=new AudioContext();const d=c.createMediaStreamDestination();const t=d.stream.getAudioTracks()[0];const stop=t.stop.bind(t);t.stop=()=>{stop();if(c.state!=='closed')c.close();};t.getSettings=()=>({echoCancellation:false});return d.stream;};"""
            )
            await page.goto(str(server.make_url("/")))
            await page.click("#voice-start")
            await page.wait_for_function(
                "() => document.getElementById('voice-start').textContent==='End conversation'"
            )
            s = holder["app"].session
            s.activate("wake_injected_file")
            (BASE / "submission-started").touch(exist_ok=False)
            start = time.monotonic()
            await (await s.submit(TASK.encode()))
            report["turn_elapsed_ms"] = (time.monotonic() - start) * 1000
            # A turn timeout does not cancel independent Onit work. Reconcile without resubmission.
            await asyncio.gather(*(j for _, j in tasks.operations.values()), return_exceptions=True)
            await asyncio.sleep(0.05)
            report["transcript"] = s.transcript.snapshot()
            report["events"] = s.events
            result = None
            for operation, (_, job) in tasks.operations.items():
                folder = tasks.root / hashlib.sha256(operation.encode()).hexdigest()[:24]
                report["task_workspace"] = str(folder)
                if job.done() and not job.cancelled() and job.exception() is None:
                    result = tasks._with_location(job.result(), folder.name)
                else:
                    report["job_error_type"] = (
                        type(job.exception()).__name__ if not job.cancelled() else "CancelledError"
                    )
                report["stages"] = (
                    json.loads((folder / "worker-stages.json").read_text())
                    if (folder / "worker-stages.json").exists()
                    else None
                )
                report["request_audit"] = (
                    [
                        json.loads(line)
                        for line in (folder / "request-attempts.jsonl").read_text().splitlines()
                    ]
                    if (folder / "request-attempts.jsonl").exists()
                    else []
                )
                report["workspace_files"] = sorted(x.name for x in folder.iterdir())
                if (folder / "final-reply.json").exists():
                    report["exact_final_reply"] = json.loads(
                        (folder / "final-reply.json").read_text()
                    )
            report["executor_result"] = result
            if result and "index.html" in result.get("artifacts", []):
                body = await tasks.read_artifact(result["task_id"], "index.html")
                response = await page.request.get(
                    str(
                        server.make_url(
                            "/api/voice/tasks/" + result["task_id"] + "/artifacts/index.html"
                        )
                    )
                )
                downloaded = await response.body()
                report["artifact_access"] = {
                    "status": response.status,
                    "identical_bytes": downloaded == body,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "content_disposition": response.headers.get("content-disposition"),
                }
                await page.wait_for_selector("[data-preview-artifact]", timeout=5000)
                await page.locator("[data-preview-artifact]").first.click()
                await page.wait_for_function(
                    "expected => document.getElementById('artifact-content').textContent === expected",
                    arg=body.decode(),
                )
                report["dashboard_preview_identical"] = (
                    await page.locator("#artifact-content").text_content() == body.decode()
                )
                await page.screenshot(path=str(BASE / "dashboard-preview.png"), full_page=True)
                await page.click("#close-artifact")
                report["status"] = "files_created_requires_independent_inspection"
            else:
                report["status"] = "no_completed_artifact"
            await page.screenshot(path=str(BASE / "dashboard-desktop.png"), full_page=True)
            await page.set_viewport_size({"width": 390, "height": 844})
            await page.screenshot(path=str(BASE / "dashboard-narrow.png"), full_page=True)
            # Speech/engagement controls must not re-execute or delete the completed output.
            old_count = len(tasks.operations)
            await s.interrupt()
            report["after_speech_stop_operation_count"] = len(tasks.operations)
            if result:
                operation = next(iter(tasks.operations))
                _, job = tasks.operations[operation]
                report["duplicate_check_count_before"] = old_count
                report["artifact_preserved_after_stop"] = (
                    await tasks.read_artifact(result["task_id"], "index.html") == body
                    if result.get("artifacts")
                    else None
                )
            await page.click("#voice-start")
            await page.wait_for_function(
                "() => document.getElementById('voice-start').textContent==='Start conversation'"
            )
            report["ended_preserves_operations"] = len(tasks.operations) == old_count
            report["queue_empty"] = s.playback.queue.empty()
            report["pending_audio"] = len(s.playback.sink.pending)
            report["browser_errors"] = errors
            await browser.close()
            browser = None
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        if not tasks.operations:
            report["pre_inference_error"] = str(exc)
        report["status"] = "check_failed_inspect_partial_evidence"
    finally:
        if browser:
            await browser.close()
        if s:
            report.setdefault("transcript", s.transcript.snapshot())
            report.setdefault("events", s.events)
        await tasks.close()
        report["worker_reaped"] = tasks.child is None or tasks.child.returncode is not None
        report["worker_returncode"] = tasks.child.returncode if tasks.child else None
        report["pending_operations"] = sum(not j.done() for _, j in tasks.operations.values())
        report["total_request_attempts"] = sum(
            json.loads(p.read_text())["attempts"] for p in tasks.root.glob("*/worker-stages.json")
        )
        report["workspace_directories"] = [str(x) for x in tasks.root.iterdir() if x.is_dir()]
        report["assignment_total_attempts"] = report["total_request_attempts"]
        # Raw stage/attempt files survive even if a surrounding browser check failed.
        (BASE / "result.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                {
                    k: report.get(k)
                    for k in (
                        "status",
                        "total_request_attempts",
                        "task_workspace",
                        "error_type",
                        "worker_reaped",
                        "pending_operations",
                    )
                }
            )
        )


asyncio.run(main())
