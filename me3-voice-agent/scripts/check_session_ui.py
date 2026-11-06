"""Actual temporary bridge/dashboard; text-injected Session and fake zero-input stream."""

import argparse
import os
import asyncio
import json
from argparse import Namespace
from pathlib import Path
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright
import me3_voice.voice_bridge as bridge
from me3_voice.configuration import load_backends
from me3_voice.devices import DeviceActions
from me3_voice.providers import MockLLM
from me3_voice.runtime import AudioApplication
from me3_voice.session import State
from me3_voice.artifacts import output_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=output_path, default=Path("var/results/session-ui-check"))
BASE = parser.parse_args().output_dir
BASE.mkdir(parents=True, exist_ok=True)
# Explicit fixture mode: do not inherit an active Spotify configuration.
os.environ.pop("ME3_MUSIC_CONFIG", None)


class Speech:
    async def transcribe(self, key, pcm):
        return pcm.decode()

    async def synthesize(self, key, text):
        for _ in range(25):
            yield b"\1\0" * 512
            await asyncio.sleep(0.002)


async def main():
    root = Path.cwd()
    devices = DeviceActions()
    from me3_voice.onit_tasks import OnitPrimaryReasoning, OnitTaskExecutor, OnitTaskUnavailable
    from me3_voice.configuration import BackendSpec
    from me3_voice.contracts import TurnKey
    from me3_voice.providers import ToolCall
    import sys

    entered, release = asyncio.Event(), asyncio.Event()
    writes = []

    async def draft_peer(task, directory, budget):
        if "answer validation" in task:
            return {
                "mode": "local-answer",
                "summary": "Onit answered the request; no external action was reported.",
                "answer": "A concise test answer.",
                "artifacts": [],
                "requests": 0,
            }
        if "fail validation" in task:
            error = OnitTaskUnavailable(
                "onit_no_output",
                "The agent returned no answer or confirmed output. Inspect its workspace before retrying.",
            )
            error.task_status = "failed"
            raise error
        entered.set()
        await release.wait()
        (directory / "index.html").write_text("<h1>Offline late task receipt</h1>")
        writes.append(directory)
        return {
            "mode": "local-artifact",
            "summary": "One offline draft created.",
            "artifacts": ["index.html"],
            "requests": 1,
        }

    offline_spec = BackendSpec(
        backend="external",
        protocol="chat-completions-v1",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="nvidia/nemotron-3.5-lightning:free",
        auth_env="ME3_LLM_API_KEY",
        allow_live=True,
        request_limit=3,
    )
    independent = OnitTaskExecutor(
        Path(sys.executable).resolve(),
        offline_spec,
        {"ME3_LLM_API_KEY": "offline"},
        root=(BASE / "offline-task-workspaces").resolve(),
        max_requests=3,
        runner=draft_peer,
        devices=devices,
    )
    tasks = independent
    holder = {}

    def capture(*args, **kwargs):
        app = AudioApplication(*args, **kwargs)
        holder["app"] = app
        return app

    bridge.AudioApplication = capture
    args = Namespace(
        kws_assets=root / "var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
        keywords_file=root / "var/cache/wake/hey-swift-keywords.txt",
        kws_threshold=0.25,
        kws_boost=1.0,
        wake_model=None,
        mel_model=None,
        embedding_model=None,
        wake_threshold=0.5,
    )
    app = bridge.build_app(root, load_backends(None), {}, args)
    voice = next(
        r.handler
        for r in app.router.routes()
        if r.resource.canonical == "/api/voice/ws" and r.method == "GET"
    )
    services = next(
        c.cell_contents
        for c in voice.__closure__
        if isinstance(c.cell_contents, bridge.VoiceRuntimeServices)
    )
    services.devices = devices
    services.tasks = tasks

    async def open_services():
        return Speech(), MockLLM(), tasks

    services.open = open_services
    async with TestServer(app, host="127.0.0.1") as server, async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1280, "height": 720}, reduced_motion="reduce"
        )
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.add_init_script(
            """navigator.mediaDevices.getUserMedia=async()=>{const c=new AudioContext();const d=c.createMediaStreamDestination();window.fakeCapture={c,d};const track=d.stream.getAudioTracks()[0];const stop=track.stop.bind(track);track.stop=()=>{stop();if(c.state!=="closed")c.close();};d.stream.getAudioTracks()[0].getSettings=()=>({echoCancellation:false});return d.stream;};"""
        )
        await page.goto(str(server.make_url("/")))
        await page.click("#voice-start")
        try:
            await page.wait_for_function(
                "() => document.getElementById('voice-start').textContent==='End conversation'"
            )
        except Exception:
            (BASE / "reconnect-failure.json").write_text(
                json.dumps(
                    {"body": await page.locator("body").inner_text(), "browser_errors": errors},
                    indent=2,
                )
            )
            await page.screenshot(path=str(BASE / "reconnect-failure.png"), full_page=True)
            raise
        s = holder["app"].session
        s.activate("wake_injected_file")
        old = await s.submit(b"Set brightness to 70 percent.")
        async with asyncio.timeout(3):
            while s.state != State.SPEAKING:
                await asyncio.sleep(0.01)
        await page.wait_for_function(
            "() => document.getElementById('session-summary').textContent.includes('Speaking')"
        )
        await page.screenshot(path=str(BASE / "interactive-speaking-desktop.png"))
        print("STOP CLICK", flush=True)
        await page.click("#voice-interrupt")
        await asyncio.gather(old, return_exceptions=True)
        async with asyncio.timeout(2):
            while s.generation is not None:
                await asyncio.sleep(0.01)
        print("CORRECTION", flush=True)
        await (await s.submit(b"Actually, 30 percent."))
        print("CORRECTION DONE", flush=True)
        await page.wait_for_function("() => document.body.innerText.includes('30%')", timeout=5000)
        assert devices.effects == 2 and devices.brightness == 30
        await (await s.submit(b"Turn it off again."))
        assert devices.effects == 3 and not devices.light
        before = devices.effects
        await (await s.submit(b"Cancel that."))
        assert devices.effects == before
        (BASE / "before-clarification-check.json").write_text(
            json.dumps(
                {
                    "transcript": s.transcript.snapshot(),
                    "history": s.history,
                    "events": s.events,
                    "body": await page.locator("body").inner_text(),
                },
                indent=2,
            )
        )
        await page.wait_for_function(
            "() => document.body.innerText.includes('Which task')", timeout=5000
        )
        await page.select_option("#theme-select", "dark")
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.screenshot(
            path=str(BASE / "interactive-correction-narrow-dark.png"), full_page=True
        )
        # Actual microphone control disables the fake track, without touching hardware.
        await page.click("#voice-mute")
        assert await page.locator("#voice-mute").inner_text() == "Unmute microphone"
        await page.click("#voice-mute")
        await (await s.submit(b"agent task: answer validation"))
        answer_receipt = s.transcript.snapshot()["tasks"][-1]
        assert answer_receipt["answer_only"] is True
        await page.wait_for_function(
            "() => document.body.innerText.includes('A concise test answer.')"
        )
        assert await page.locator(f'[data-task-id="{answer_receipt["task_id"]}"]').count() == 0
        await page.screenshot(path=str(BASE / "answer-only-narrow.png"), full_page=True)
        await (await s.submit(b"Stop speaking."))
        assert s.playback.queue.empty()
        from me3_voice.onit_tasks import OnitPrimaryReasoning

        try:
            s.tasks = services.tasks = independent
            pending = await s.submit(b"agent task: create one offline draft")
            await entered.wait()
            task_turn = s.key.turn_id
            await s.interrupt()
            await asyncio.gather(pending, return_exceptions=True)
            await (await s.submit(b"Hello."))
            release.set()
            await asyncio.gather(*(j for _, j in independent.operations.values()))
            await asyncio.sleep(0)
            settled = next(t for t in s.transcript.snapshot()["tasks"] if t["turn_id"] == task_turn)
            assert settled["status"] == "reported" and settled["artifacts"] == ["index.html"]
            await page.locator("[data-preview-artifact]").first.wait_for(state="visible")
            await page.locator("[data-preview-artifact]").first.click()
            await page.wait_for_function(
                "() => document.getElementById('artifact-content').textContent.includes('Offline late task receipt')"
            )
            await page.screenshot(path=str(BASE / "late-task-preview-narrow.png"), full_page=True)
            await page.click("#close-artifact")
            operation, (task_text, job) = next(
                (operation, entry)
                for operation, entry in independent.operations.items()
                if operation.split(":")[1] == str(task_turn)
            )
            duplicate = await independent.execute(
                TurnKey(session_id=s.session_id, turn_id=task_turn),
                ToolCall(operation.rsplit(":", 1)[1], "run_agent_task", {"task": task_text}),
            )
            assert len(writes) == 1 and len(independent.operations) == 2
            assert (
                await independent.read_artifact(duplicate["task_id"], "index.html")
                == b"<h1>Offline late task receipt</h1>"
            )
        finally:
            release.set()
            s.tasks = services.tasks = tasks

        await (await s.submit(b"agent task: fail validation before any file"))
        failure = s.transcript.snapshot()["tasks"][-1]
        assert (
            failure["status"] == "failed" and "no answer or confirmed output" in failure["summary"]
        )
        await page.wait_for_function(
            "() => document.body.innerText.includes('no answer or confirmed output')"
        )
        await page.screenshot(path=str(BASE / "no-output-failure-narrow.png"), full_page=True)
        await (await s.submit(b"Hello again."))
        assert len(writes) == 1 and s.playback.queue.empty()

        s.llm = OnitPrimaryReasoning()  # Capacity fails locally; no worker/model call.
        await (await s.submit(b"x" * 7900))
        await page.locator("#fresh-context").wait_for(state="visible")
        await page.screenshot(path=str(BASE / "context-limit-recovery.png"), full_page=True)
        await page.click("#fresh-context")
        assert devices.effects == 3
        assert [e.get("error_code") for e in s.events if e["event"] == "turn_error"] == [
            "onit_no_output",
            "context_capacity",
        ]
        await page.wait_for_function(
            "() => document.getElementById('voice-start').textContent==='Start conversation'"
        )
        await page.wait_for_function(
            "async () => (await (await fetch('/api/health')).json()).voice_session==='idle'"
        )
        await page.click("#voice-start")
        await page.wait_for_function(
            "() => document.getElementById('voice-start').textContent==='End conversation'"
        )
        fresh = holder["app"].session
        assert fresh is not s and fresh.history == [] and devices.effects == 3
        assert (
            fresh.transcript.snapshot()["tasks"] == []
            and len(tasks.operations) == 3
            and len(writes) == 1
        )
        assert (
            await tasks.read_artifact(duplicate["task_id"], "index.html")
            == b"<h1>Offline late task receipt</h1>"
        )
        await page.click("#voice-start")
        await page.wait_for_function(
            "() => document.getElementById('voice-start').textContent==='Start conversation'"
        )
        await page.screenshot(path=str(BASE / "interactive-ended-narrow-dark.png"), full_page=True)
        assert not errors, errors
        (BASE / "interactive-session.json").write_text(
            json.dumps(
                {
                    "evidence": "actual bridge WebSocket/BrowserSink ACK/publisher/dashboard, transcript-injected Session, mock reasoning/TTS/device output, fake zero microphone stream",
                    "effects": devices.effects,
                    "answer_only_hidden_with_receipt_retained": answer_receipt["answer_only"],
                    "fresh_context_verified": fresh.history == [],
                    "late_task_receipt_verified": settled["status"] == "reported",
                    "late_task_file_writes": len(writes),
                    "reconnect_did_not_replay_receipt": fresh.transcript.snapshot()["tasks"] == [],
                    "known_artifact_access_after_reconnect": True,
                    "no_output_failure_and_next_turn_verified": failure["status"] == "failed",
                    "output_events": devices.output.events,
                    "transcript": s.transcript.snapshot(),
                    "events": s.events,
                    "queue_empty": s.playback.queue.empty(),
                    "pending_audio": len(s.playback.sink.pending),
                    "browser_errors": errors,
                },
                indent=2,
            )
        )
        await page.evaluate(
            "() => window.fakeCapture.c.state === 'closed' ? undefined : window.fakeCapture.c.close()"
        )
        await browser.close()


try:
    asyncio.run(asyncio.wait_for(main(), 45))
except BaseException:
    import traceback

    (BASE / "interactive-error.txt").write_text(traceback.format_exc())
    raise
