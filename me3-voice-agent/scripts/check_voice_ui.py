"""Local Chromium contract check for Swift's browser voice projection (mock events)."""

import argparse
import copy
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

from me3_voice.dashboard import handler, render
from playwright.sync_api import expect, sync_playwright


FAKE_TRANSPORT = r"""
(() => {
  const nativeContext = window.AudioContext;
window.testOscillatorCount = 0;
const nativeCreateOscillator = nativeContext.prototype.createOscillator;
nativeContext.prototype.createOscillator = function() {
  window.testOscillatorCount++;
  return nativeCreateOscillator.call(this);
};
  navigator.mediaDevices.getUserMedia = async () => {
    if(window.testMicError){const error=new Error('mock microphone failure');error.name=window.testMicError;throw error;}
    const context = new nativeContext({sampleRate:48000});
    const destination = context.createMediaStreamDestination();
    const oscillator = context.createOscillator();
    const gain = context.createGain(); gain.gain.value = .3;
    oscillator.connect(gain); gain.connect(destination); oscillator.start();
    destination.stream.getAudioTracks()[0].getSettings = () => ({echoCancellation:true});
    window.testCapture = {context, oscillator, gain, destination};
    return destination.stream;
  };
  class FakeSocket {
    static OPEN = 1;
    constructor() {
      this.readyState = 0; this.sent = []; window.testSocket = this; window.testSocketCount=(window.testSocketCount||0)+1;
      setTimeout(() => { this.readyState = 1; this.onopen?.(); }, 0);
    }
    send(value) {
      if (typeof value === 'string') {
        const message = JSON.parse(value); this.sent.push(message);
                if (message.type === 'hello') {
          const ready={type:'ready',run_id:'mock-run',session_id:'mock-session',mode:'mock',task_artifact_access:true,devices:window.testDeviceCapabilities||{target:'mock',status:'simulated',available:true,physical_readback:false,message:'Mock device adapter · no GPIO output'}};
          if(window.testDelayReady)window.releaseTestReady=()=>this.emit(ready);else this.emit(ready);
        }
      }
    }
    emit(value) { setTimeout(() => this.onmessage?.({data:JSON.stringify(value)}), 0); }
    close() { if(this.readyState===3)return; this.readyState = 3; setTimeout(() => this.onclose?.(), 0); }
  }
  window.WebSocket = FakeSocket;
})()
"""


def capture_matrix(page, out, state):
    viewports = (
        ("1280x720", 1280, 720),
        ("1440x900", 1440, 900),
        ("1024x768", 1024, 768),
        ("390x844", 390, 844),
    )
    for label, width, height in viewports:
        page.set_viewport_size({"width": width, "height": height})
        page.evaluate("window.scrollTo(0, 0)")
        page.evaluate("document.getElementById('task-items')?.scrollTo(0, 0)")
        for theme in ("light", "dark"):
            page.locator("#theme-select").select_option(theme)
            page.wait_for_timeout(250)
            expected_canvas = "#111827" if theme == "dark" else "#F6F8FC"
            actual_canvas = page.evaluate(
                "getComputedStyle(document.documentElement).getPropertyValue('--canvas').trim()"
            )
            assert actual_canvas.lower() == expected_canvas.lower(), (
                f"theme did not settle in {state}/{theme}/{label}: {actual_canvas}"
            )
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                f"horizontal overflow in {state}/{theme}/{label}"
            )
            page.screenshot(path=str(out / f"mock-{state}-{theme}-{label}.png"))
            if width == 390 and state.startswith(("task-output", "music-")):
                page.screenshot(
                    path=str(out / f"mock-{state}-{theme}-390x844-full.png"), full_page=True
                )
    page.locator("#theme-select").select_option("light")
    page.set_viewport_size({"width": 1280, "height": 720})


def check_spotify_refresh(out):
    """Exercise report replacement against the actual UI with a mock Spotify SDK."""
    report = {
        "schema_version": 1,
        "generated_utc": "2026-11-01T00:00:00Z",
        "reports": [],
        "warnings": [],
    }
    capability = {
        "available": True,
        "music": {
            "configured": True,
            "output_backend": "web_playback_sdk",
            "browser_player": True,
            "setup_required": False,
        },
    }
    sdk = """
    window.Spotify = {Player: class {
      constructor(options) { this.listeners={}; window.latestSpotifyPlayer=this; window.spotifyConnections=(window.spotifyConnections||0)+1; }
      addListener(name,callback) {this.listeners[name]=callback;}
      activateElement() {return Promise.resolve(true);}
      nextTrack() {window.mockNext=(window.mockNext||0)+1;return Promise.resolve(true);}
      previousTrack() {window.mockPrevious=(window.mockPrevious||0)+1;return Promise.resolve(true);}
      connect() {const frame=document.createElement('iframe');frame.src='https://sdk.scdn.co/embedded/index.html';this.frame=frame;window.addEventListener('message',event=>{if(event.origin==='https://sdk.scdn.co'&&event.data==='mock-embedded-ready'&&!window.spotifySilent)this.listeners.ready({device_id:'mock-swift-device'});},{once:true});document.body.append(frame);return Promise.resolve(true);}
      disconnect() {this.frame?.remove();}
    }};
    window.onSpotifyWebPlaybackSDKReady();
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.add_init_script(FAKE_TRANSPORT)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(
            "http://127.0.0.1:8765/",
            lambda route: route.fulfill(content_type="text/html", body=render(report, live=True)),
        )
        page.route("**/api/voice/capabilities", lambda route: route.fulfill(json=capability))
        page.route(
            "**/api/health",
            lambda route: route.fulfill(
                json={
                    "bridge": "ready",
                    "voice_session": "idle",
                    "speech_services": "lazy",
                    "session_resume": False,
                }
            ),
        )
        polls = []

        def snapshot_route(route):
            polls.append(True)
            route.fulfill(json=report)

        page.route("**/api/snapshot", snapshot_route)
        page.route(
            "**/api/voice/music/device", lambda route: route.fulfill(json=capability["music"])
        )
        music_status = {"status": "not_playing", "output_backend": "web_playback_sdk"}
        status_failed = [False]
        page.route(
            "**/api/voice/music/status",
            lambda route: route.fulfill(
                status=503 if status_failed[0] else 200,
                json={"message": "Spotify connection failed; playback state is uncertain"}
                if status_failed[0]
                else music_status,
            ),
        )
        control_requests = []

        def music_control(route):
            command = route.request.post_data_json
            control_requests.append(command)
            route.fulfill(
                json={
                    "available": True,
                    "state": music_status,
                    "receipt": {"summary": "Spotify accepted the control."},
                }
            )

        page.route("**/api/voice/music/control", music_control)
        page.route(
            "https://sdk.scdn.co/spotify-player.js",
            lambda route: route.fulfill(content_type="text/javascript", body=sdk),
        )
        page.route(
            "https://sdk.scdn.co/embedded/index.html",
            lambda route: route.fulfill(
                content_type="text/html",
                body="<script>parent.postMessage('mock-embedded-ready','*')</script>",
            ),
        )
        page.goto("http://127.0.0.1:8765/")
        expect(page.locator("#voice-controls")).to_be_visible()
        page.wait_for_timeout(
            2500
        )  # Reproduce the first automatic snapshot replacement BEFORE Start.
        assert polls, "automatic report polling did not run"
        page.locator("#voice-start").click()
        expect(page.locator("#session-summary")).to_have_text("Waiting for ‘Hey Swift’")
        expect(page.locator("#spotify-output-label")).to_contain_text("this browser")
        expect(page.locator("#spotify-player")).to_be_hidden()
        expect(page.locator("footer #music-setup-status")).to_have_text("Spotify connected")
        expect(page.locator("#notice")).not_to_contain_text("Spotify connected")
        music_status.update(
            status="paused",
            target_matched=False,
            track={"name": "Risk it All — other device", "artists": "Mock"},
        )
        page.wait_for_timeout(3200)
        expect(page.locator("#spotify-player")).to_be_hidden()
        music_status.update(
            status="playing",
            target_matched=True,
            track={"name": "Mock track — UI fixture", "artists": "Mock artist"},
            duration_ms=120000,
            progress_ms=1000,
        )
        expect(page.locator("#spotify-player")).to_be_visible()
        page.wait_for_timeout(4500)  # Player must survive further automatic replacements.
        expect(page.locator("#spotify-player")).to_be_visible()
        assert page.evaluate("window.spotifyConnections") == 1, "polling recreated the player"
        status_failed[0] = True
        page.wait_for_timeout(3200)
        expect(page.locator("#spotify-track-title")).to_have_text("Mock track — UI fixture")
        expect(page.locator("#spotify-caption")).to_contain_text("temporarily unavailable")
        expect(page.locator("#spotify-toggle")).to_be_disabled()
        status_failed[0] = False
        page.wait_for_timeout(3200)
        expect(page.locator("#spotify-track-title")).to_have_text("Mock track — UI fixture")
        expect(page.locator("#spotify-toggle")).to_be_enabled()
        page.screenshot(path=str(out / "mock-spotify-after-poll-desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#spotify-player").scroll_into_view_if_needed()
        expect(page.locator("#spotify-connect")).to_be_hidden()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(out / "mock-spotify-after-poll-narrow.png"), full_page=True)
        page.locator("#file-input").set_input_files(
            {
                "name": "report.json",
                "mimeType": "application/json",
                "buffer": json.dumps(report).encode(),
            }
        )
        expect(page.locator("#spotify-player")).to_be_visible()
        assert page.evaluate("window.spotifyConnections") == 1
        page.locator("#voice-start").click()
        expect(page.locator("#session-summary")).to_have_text("Conversation ended")
        page.locator("#voice-start").click()
        expect(page.locator("#session-summary")).to_have_text("Waiting for ‘Hey Swift’")
        expect(page.locator("#spotify-player")).to_be_visible()
        assert page.evaluate("window.spotifyConnections") == 1
        page.evaluate(
            "window.latestSpotifyPlayer.listeners.player_state_changed({paused:false,position:1000,duration:120000,disallows:{},track_window:{current_track:{id:'mock-track-a',name:'Fixture track A',artists:[],album:{images:[]}}}})"
        )
        observed = page.evaluate(
            "testSocket.sent.filter(row=>row.type==='music_playback_observed')"
        )
        assert observed and observed[-1] == {
            "type": "music_playback_observed",
            "status": "playing",
            "source": "web_playback_sdk",
            "track_changed": True,
        }
        assert not any("track" in key and key != "track_changed" for key in observed[-1])
        page.evaluate(
            "window.latestSpotifyPlayer.listeners.player_state_changed({paused:false,position:2000,duration:120000,disallows:{},track_window:{current_track:{id:'mock-track-a',name:'Fixture track A',artists:[],album:{images:[]}}}})"
        )
        assert page.evaluate(
            "testSocket.sent.filter(row=>row.type==='music_playback_observed').length"
        ) == len(observed), "ordinary playback progress must not flood the trace"
        page.evaluate(
            "window.latestSpotifyPlayer.listeners.player_state_changed({paused:false,position:0,duration:120000,disallows:{},track_window:{current_track:{id:'mock-track-b',name:'Fixture track B',artists:[],album:{images:[]}}}})"
        )
        latest = page.evaluate(
            "testSocket.sent.filter(row=>row.type==='music_playback_observed').at(-1)"
        )
        assert latest["status"] == "playing" and latest["track_changed"] is True
        page.locator("#spotify-next").click()
        page.locator("#spotify-previous").click()
        assert [command["action"] for command in control_requests] == ["next", "previous"]
        assert all(
            re.fullmatch(r"ui-[a-f0-9]{32}", command["command_id"]) for command in control_requests
        )
        for _ in range(40):
            accepted_controls = page.evaluate(
                "testSocket.sent.filter(row=>row.type==='music_ui_control'&&row.outcome==='accepted').length"
            )
            if accepted_controls == 2:
                break
            page.wait_for_timeout(50)
        assert accepted_controls == 2, "both control acknowledgements should reach the live feed"
        assert (
            page.evaluate(
                "testSocket.sent.filter(row=>row.type==='music_ui_control'&&row.outcome==='accepted').length"
            )
            == 2
        )
        page.evaluate(
            "window.latestSpotifyPlayer.listeners.player_state_changed({paused:false,position:1000,duration:120000,disallows:{skipping_next:true,skipping_prev:true},track_window:{current_track:{name:'Mock queue end',artists:[],album:{images:[]}}}})"
        )
        expect(page.locator("#spotify-next")).to_be_disabled()
        expect(page.locator("#spotify-previous")).to_be_disabled()
        # A loaded SDK that never emits ready must release the button and ignore late events.
        music_status.clear()
        music_status.update(status="not_playing", output_backend="web_playback_sdk")
        page.add_init_script("window.spotifySilent=true")
        page.goto("http://127.0.0.1:8765/")
        expect(page.locator("#voice-controls")).to_be_visible()
        page.locator("#voice-start").click()
        page.wait_for_timeout(16000)
        expect(page.locator("#music-setup-status")).to_contain_text("timed out")
        expect(page.locator("#music-setup-connect")).to_be_enabled()
        expect(page.locator("#spotify-output-label")).not_to_contain_text("Connecting")
        page.screenshot(path=str(out / "mock-spotify-timeout.png"), full_page=True)
        page.evaluate(
            "window.oldSpotifyPlayer=window.latestSpotifyPlayer;window.spotifySilent=false"
        )
        page.locator("#music-setup-connect").click()
        expect(page.locator("#spotify-output-label")).to_contain_text("this browser")
        assert page.evaluate("window.spotifyConnections") == 1, "retry must reuse the SDK singleton"
        page.evaluate("window.oldSpotifyPlayer.listeners.ready({device_id:'stale-device'})")
        expect(page.locator("#spotify-connect")).to_be_hidden()
        # A report carrying live=true must never enable live capabilities on an offline page.
        page.route(
            "http://127.0.0.1:8765/offline",
            lambda route: route.fulfill(content_type="text/html", body=render(report, live=False)),
        )
        page.goto("http://127.0.0.1:8765/offline")
        forged = {**report, "live": True}
        page.locator("#file-input").set_input_files(
            {
                "name": "report.json",
                "mimeType": "application/json",
                "buffer": json.dumps(forged).encode(),
            }
        )
        expect(page.locator("#spotify-player")).to_be_hidden()
        expect(page.locator("#voice-controls")).to_be_hidden()
        expect(page.locator("#aec-probe-link")).to_be_hidden()
        assert not errors, errors
        browser.close()
    (out / "spotify-refresh-checks.json").write_text(
        json.dumps(
            {
                "passed": True,
                "automatic_snapshot_polls": len(polls),
                "sdk": "mock",
                "physical_audio_verified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Spotify UI: delayed Start, repeated polling, import, end/reconnect, offline isolation passed (mock SDK)."
    )


def check_background_task_ui(scenario_dir, summary_file, terminal_dir, out):
    """Exercise task snapshots through the canonical loopback dashboard handler."""
    scenario_dir = Path(scenario_dir).resolve()
    summary_file = Path(summary_file).resolve()
    terminal_dir = Path(terminal_dir).resolve()
    for name in ("task-failed.json", "task-cancelled.json"):
        if not (terminal_dir / name).is_file():
            raise FileNotFoundError(f"missing mock terminal state: {terminal_dir / name}")

    def wait_for_file(path, timeout=180):
        deadline = time.monotonic() + timeout
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"scenario did not publish {path.name} within {timeout}s")
            time.sleep(0.25)
        return path

    wait_for_file(scenario_dir / "task-accepted.json", timeout=5)
    accepted = json.loads((scenario_dir / "task-accepted.json").read_text(encoding="utf-8"))
    task_id = next(task.get("task_id") for task in accepted["tasks"] if task["kind"] == "agent")
    if not isinstance(task_id, str) or len(task_id) < 8:
        raise AssertionError("background scenario has no stable task ID")

    server_root = Path("var/scratch/016-background-task-ui/served-root").resolve()
    result_root = server_root / "var/results"
    result_root.mkdir(parents=True, exist_ok=True)
    live_report = result_root / "ui-transcript.json"

    def install(stage):
        wait_for_file(scenario_dir / stage)
        payload = json.loads((scenario_dir / stage).read_text(encoding="utf-8"))
        temporary = live_report.with_suffix(".json.new")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(live_report)

    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args: None)
    server.RequestHandlerClass = handler(server_root, server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    browser_errors = []
    snapshots = []
    try:
        install("task-accepted.json")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.add_init_script(FAKE_TRANSPORT)
            page.on("pageerror", lambda error: browser_errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.locator("#history-select").select_option(accepted["session_id"])
            expect(page.locator("#session-summary")).to_have_text("Recorded session")
            expect(page.locator("#task-panel-heading h2")).to_have_text("Task at snapshot")
            expect(page.locator("#active-task-items .task-card")).to_have_count(1)
            active = page.locator("#active-task-items .task-card")
            expect(active).to_have_attribute("data-task-id", task_id)
            expect(active.locator(".task-evidence")).to_contain_text(
                "Recorded · was running · Mock"
            )
            expect(active.locator(".task-id")).to_have_count(0)
            expect(active.locator(".task-copy strong")).to_have_text("Background task")
            expect(active.locator("[data-task-elapsed]")).to_contain_text("at snapshot")
            expect(active.locator("[data-task-progress-age]")).to_contain_text("before snapshot")
            expect(active.locator(".background-task-phase")).to_contain_text(
                "Planning the requested project files"
            )
            expect(active.locator(".task-steps li")).to_have_count(0)
            expect(page.locator("#voice-start")).to_have_text("Start conversation")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / "task-accepted-desktop-1280x720.png"))

            # UI-only fixture: concurrent tasks retain distinct IDs and are
            # never resubmitted by rendering or report replacement.
            parallel = copy.deepcopy(accepted)
            second = copy.deepcopy(
                next(task for task in parallel["tasks"] if task["kind"] == "agent")
            )
            second.update(
                id="9:mock-task",
                turn_id=9,
                call_id="mock-task",
                task_id="parallel_mock_task_000002",
                title="Second background task",
                phase="Checking a separate project outline",
                summary="Separate mock-only fixture.",
            )
            parallel["tasks"].append(second)
            parallel_path = live_report.with_suffix(".json.new")
            parallel_path.write_text(
                json.dumps(parallel, ensure_ascii=True, allow_nan=False), encoding="utf-8"
            )
            parallel_path.replace(live_report)
            expect(page.locator("#active-task-items .task-card")).to_have_count(2)
            parallel_ids = page.locator("#active-task-items .task-card").evaluate_all(
                "cards => cards.map(card => card.dataset.taskId)"
            )
            assert len(set(parallel_ids)) == 2
            snapshots.append("two simultaneous mock task IDs remain distinct")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / "task-multiple-active-desktop-1280x720.png"))
            install("task-accepted.json")
            expect(page.locator("#active-task-items .task-card")).to_have_count(1)

            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / "task-accepted-narrow-390x844.png"), full_page=True)
            page.set_viewport_size({"width": 1280, "height": 720})

            install("task-interleaved.json")
            expect(page.locator("#active-task-items .task-card")).to_have_count(1, timeout=6000)
            active = page.locator("#active-task-items .task-card")
            expect(page.locator("#task-state-summary")).to_contain_text("MUSIC · SIMULATION")
            expect(page.locator("#task-state-summary")).to_contain_text("LIGHT · SIMULATION")
            for command in (
                "Turn on the lights",
                "Play music",
                "Pause music",
                "Change the light color",
            ):
                expect(page.locator("#transcript-messages")).to_contain_text(command)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / "task-interleaved-desktop-1280x720.png"))
            page.locator("#theme-select").select_option("dark")
            page.screenshot(path=str(out / "task-interleaved-dark-1280x720.png"))
            page.locator("#theme-select").select_option("light")

            # Refresh while the worker is active. This reloads the saved mock
            # projection only; the read-only dashboard cannot resubmit work.
            page.reload()
            expect(page.locator("#history-select")).to_have_value(accepted["session_id"])
            expect(page.locator("#session-summary")).to_have_text("Recorded session")
            expect(page.locator("#task-panel-heading h2")).to_have_text("Task at snapshot")
            expect(page.locator("#active-task-items .task-card")).to_have_count(1)
            expect(page.locator("#active-task-items .task-card")).to_have_attribute(
                "data-task-id", task_id
            )
            snapshots.append("active refresh preserved task ID; read-only UI started no work")

            install("task-interrupted.json")
            expect(page.locator("#active-task-items .task-card")).to_have_count(1, timeout=6000)
            interrupted_card = page.locator("#active-task-items .task-card")
            expect(interrupted_card).to_have_attribute("data-task-id", task_id)
            expect(interrupted_card).to_have_attribute("data-status", "running")
            expect(interrupted_card.locator(".task-steps")).to_contain_text("Scope captured")
            heartbeat_text = interrupted_card.locator("span[data-task-heartbeat]")
            expect(heartbeat_text).to_contain_text("snapshot")
            expect(page.locator("#transcript-messages")).to_contain_text("Interrupted")
            page.screenshot(path=str(out / "task-interrupted-desktop-1280x720.png"))

            install("task-completed.json")
            expect(page.locator("#active-task-items")).to_be_hidden(timeout=6000)
            completed = page.locator(f'#task-items [data-task-id="{task_id}"]')
            expect(completed).to_have_attribute("data-status", "completed")
            expect(completed.locator(".task-evidence")).to_contain_text("Completed · mock")
            expect(completed).to_contain_text("project-plan.md")
            expect(completed).to_contain_text("ME3 mock workspace")
            expect(completed).to_contain_text("Mock file exists")
            expect(page.locator("#task-items .task-card").first).to_have_attribute(
                "data-task-id", task_id
            )
            page.screenshot(path=str(out / "task-completed-desktop-1280x720.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / "task-completed-narrow-390x844.png"), full_page=True)

            for terminal in ("failed", "cancelled"):
                payload = json.loads(
                    (terminal_dir / f"task-{terminal}.json").read_text(encoding="utf-8")
                )
                terminal_task = next(task for task in payload["tasks"] if task["kind"] == "agent")
                install_path = live_report.with_suffix(".json.new")
                install_path.write_text(
                    json.dumps(payload, ensure_ascii=True, allow_nan=False), encoding="utf-8"
                )
                install_path.replace(live_report)
                page.set_viewport_size({"width": 1280, "height": 720})
                page.reload()
                page.locator("#history-select").select_option(payload["session_id"])
                card = page.locator(f'#task-items [data-task-id="{terminal_task["task_id"]}"]')
                expect(card).to_have_attribute("data-status", terminal)
                expect(card.locator(".task-evidence")).to_contain_text(
                    "Failed · Mock" if terminal == "failed" else "Cancelled"
                )
                expect(card).to_contain_text(
                    "Controlled mock failure" if terminal == "failed" else "cancelled"
                )
                page.screenshot(path=str(out / f"task-{terminal}-desktop-1280x720.png"))
                if terminal == "failed":
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    page.screenshot(
                        path=str(out / "task-failed-narrow-390x844.png"), full_page=True
                    )
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.screenshot(path=str(out / "task-failed-narrow-scrolled-390x844.png"))
                snapshots.append(f"actual Session/mock-worker {terminal} receipt rendered")
            assert not browser_errors, browser_errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    summary = json.loads(wait_for_file(summary_file).read_text(encoding="utf-8"))
    if summary.get("mode") != "mock" or summary.get("worker_start_count") != 1:
        raise AssertionError("scenario evidence must show one mock worker start")
    if summary.get("task_id") != task_id:
        raise AssertionError("UI snapshot task ID differs from the completed worker receipt")
    result = {
        "kind": "background-task-dashboard-check",
        "schema_version": 1,
        "passed": True,
        "evidence": "canonical dashboard served over loopback from mock transcript snapshots",
        "task_id": task_id,
        "worker_start_count": summary["worker_start_count"],
        "checks": snapshots,
        "screenshots": sorted(path.name for path in out.glob("task-*.png")),
        "limitations": (
            "The browser displayed saved mock Session projections. It did not control or reconnect "
            "to a worker, open a microphone, reach Onit/provider services, or verify audible/Pi behavior."
        ),
    }
    (out / "background-task-ui-check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Background task dashboard: active/interleaved/refresh/interrupted/completed states passed (mock)."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="var/results/ui-voice-orb")
    parser.add_argument("--spotify-only", action="store_true")
    parser.add_argument("--background-task-dir")
    parser.add_argument("--background-task-summary")
    parser.add_argument("--background-terminal-dir")
    args = parser.parse_args()
    root = Path.cwd()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.spotify_only:
        check_spotify_refresh(out)
        return
    if args.background_task_dir:
        if not args.background_task_summary:
            parser.error("--background-task-summary is required with --background-task-dir")
        if not args.background_terminal_dir:
            parser.error("--background-terminal-dir is required with --background-task-dir")
        check_background_task_ui(
            args.background_task_dir,
            args.background_task_summary,
            args.background_terminal_dir,
            out,
        )
        return
    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args: None)
    server.RequestHandlerClass = handler(root, server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.add_init_script(FAKE_TRANSPORT)
            health_failure = [False]

            def health_route(route):
                route.fulfill(
                    status=503 if health_failure[0] else 200,
                    json={"error": "unavailable"}
                    if health_failure[0]
                    else {
                        "bridge": "ready",
                        "voice_session": "idle",
                        "speech_services": "lazy",
                        "session_resume": False,
                    },
                )

            page.route(
                "**/api/health",
                health_route,
            )
            page.route(
                "**/api/voice/capabilities",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {"available": True, "asr_partial": True, "task_artifact_access": True}
                    ),
                ),
            )
            page.route(
                "**/api/voice/tasks/**/artifacts/**",
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html",
                    body="<h1>Draft content is shown as text</h1>",
                ),
            )
            page.route(
                "**/api/snapshot",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "schema_version": 1,
                            "generated_utc": "2026-10-29T00:00:00Z",
                            "reports": [],
                            "warnings": [],
                        }
                    ),
                ),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            saved = {
                "schema_version": 1,
                "kind": "conversation-transcript",
                "display_authorized": True,
                "session_id": "saved-startup",
                "state": "ended",
                "mode": "HPC-live",
                "revision": 1,
                "updated_utc": "2026-11-01T00:00:00Z",
                "messages": [
                    {
                        "id": "1:user",
                        "turn_id": 1,
                        "role": "user",
                        "text": "Saved music request",
                        "confirmed_text": "",
                        "status": "final",
                        "delivery": "asr_final",
                    }
                ],
                "tasks": [
                    {
                        "id": "1:saved-music",
                        "turn_id": 1,
                        "call_id": "saved-music",
                        "kind": "music_play",
                        "status": "failed",
                        "mode": "reported",
                        "title": "Control music",
                        "summary": "Open Spotify on the laptop and select an active playback device",
                        "display": "",
                    }
                ],
            }
            fixture = {
                "schema_version": 1,
                "generated_utc": saved["updated_utc"],
                "warnings": [],
                "reports": [
                    {
                        "id": "saved-fixture",
                        "kind": "transcript",
                        "source": "mock-saved-session.json",
                        "evidence": "HPC-live",
                        "modified_utc": saved["updated_utc"],
                        "sha256": "",
                        "data": saved,
                    }
                ],
            }
            fixture_url = f"http://127.0.0.1:{server.server_port}/history-fixture"
            page.route(
                fixture_url,
                lambda route: route.fulfill(
                    status=200, content_type="text/html", body=render(fixture, live=True)
                ),
            )
            page.goto(fixture_url)
            expect(page.locator("#welcome")).to_be_visible()
            expect(page.locator("#task-activity")).to_be_hidden()
            expect(page.locator("#transcript-messages .message")).to_have_count(0)
            page.locator("#history-select").select_option("saved-startup")
            expect(page.locator(".task-recovery")).to_be_visible()
            expect(page.locator("#session-summary")).to_have_text("Recorded session")
            page.screenshot(path=str(out / "explicit-saved-history.png"))
            page.locator("#history-select").select_option("")
            expect(page.locator("#welcome")).to_be_visible()
            expect(page.locator("#task-activity")).to_be_hidden()
            page.locator("#history-select").select_option("saved-startup")
            page.reload()
            expect(page.locator("#history-select")).to_have_value("saved-startup")
            expect(page.locator("#session-summary")).to_have_text("Recorded session")
            page.locator("#history-select").select_option("")
            expect(page.locator("#welcome")).to_be_visible()
            expect(page.locator("#task-activity")).to_be_hidden()
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            expect(page.locator("#voice-controls")).to_be_visible()
            expect(page.locator("#session-summary")).to_have_text("Voice disconnected")
            page.locator('nav [data-view="diagnostics"]').click()
            expect(page.locator("#aec-probe-link")).to_be_visible()
            page.locator("#aec-probe-link").click()
            expect(page).to_have_title("Swift · Laptop echo check")
            expect(page.locator("#start")).to_have_text("Start guided trial · AEC off")
            expect(page.locator("#status")).to_contain_text(
                "Microphone starts only when you press Start"
            )
            page.go_back()
            page.locator('nav [data-view="assistant"]').click()
            expect(page.locator("#session-summary")).to_have_text("Voice disconnected")
            expect(page.locator("#welcome")).to_be_visible()
            expect(page.locator("#voice-start")).to_have_text("Start conversation")
            expect(page.locator("#voice-interrupt")).to_be_hidden()
            expect(page.locator("#voice-mute")).to_be_hidden()
            expect(page.locator(".brand-word")).to_have_text("Swift")
            expect(page.locator(".brand-mark")).to_be_visible()
            expect(page.locator(".brand-mark")).to_have_count(1)
            capture_matrix(page, out, "welcome")
            health_failure[0] = True
            page.locator("#voice-start").click()
            expect(page.locator("#notice")).to_contain_text("voice bridge is unreachable")
            expect(page.locator("#notice")).to_contain_text("No microphone was opened")
            assert page.evaluate("window.testCapture") is None
            health_failure[0] = False
            page.evaluate("window.testMicError='NotAllowedError'")
            page.locator("#voice-start").click()
            expect(page.locator("#notice")).to_contain_text("Microphone permission was denied")
            assert page.evaluate("window.testSocket") is None
            page.evaluate("window.testMicError='NotFoundError'")
            page.locator("#voice-start").click()
            expect(page.locator("#notice")).to_contain_text("No microphone is available")
            assert page.evaluate("window.testSocket") is None
            page.evaluate("window.testMicError=null")
            page.evaluate("window.testDelayReady=true")
            page.locator("#voice-start").click()
            expect(page.locator("#voice-start")).to_have_text("Starting voice…")
            expect(page.locator("#voice-start")).to_be_disabled()
            page.locator("#voice-start").evaluate("button => button.click()")
            assert page.evaluate("window.testSocketCount") == 1, (
                "duplicate start created a second socket"
            )
            page.evaluate("window.releaseTestReady()")
            page.evaluate("window.testDelayReady=false")
            expect(page.locator("#session-summary")).to_have_text("Waiting for ‘Hey Swift’")
            expect(page.locator("#task-activity")).to_be_hidden()
            expect(page.locator("#transcript-empty")).to_have_text("Say Hey Swift to begin.")
            expect(page.locator("#voice-status")).to_have_attribute("data-live", "true")
            expect(page.locator("#transcript-mode")).to_have_text("Mock voice runtime")
            assert page.locator("#view-assistant").get_attribute("data-conversation") == "true"
            page.locator("#inspect-session").click()
            expect(page.locator("#session-details")).to_contain_text("Browser reports on")
            expect(page.locator("#session-dialog-note")).to_contain_text(
                "does not measure acoustic cancellation"
            )
            page.locator("#close-session").click()
            assert 64 <= page.locator(".idle-stage").bounding_box()["height"] <= 80, page.locator(
                ".idle-stage"
            ).bounding_box()
            expect(page.locator(".signal-bars i")).to_have_count(7)
            assert page.locator(".orb-microphone").is_visible()
            oscillators_before_listening = page.evaluate("testOscillatorCount")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'listening',run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:0,status_sequence:1})"""
            )
            page.wait_for_timeout(20)
            assert page.evaluate("testOscillatorCount") == oscillators_before_listening, (
                "a listening status alone must not fabricate a wake confirmation"
            )
            expect(page.locator("#session-summary")).to_have_text("Listening…")
            expect(page.locator("#welcome")).to_be_hidden()
            expect(page.locator("#transcript-panel")).to_be_visible()
            expect(page.locator("#transcript-empty")).to_have_text("Go ahead, I’m listening.")
            expect(page.locator("#voice-status-context")).not_to_contain_text("Hey Swift")
            assert 64 <= page.locator(".idle-stage").bounding_box()["height"] <= 80, page.locator(
                ".idle-stage"
            ).bounding_box()
            page.wait_for_timeout(180)
            active_bar = (
                page.locator(".signal-bars i").nth(3).evaluate("el => parseFloat(el.style.height)")
            )
            assert active_bar > 42, "captured PCM did not move the listening signal"
            page.evaluate("testCapture.gain.gain.value = 0")
            page.wait_for_timeout(360)
            quiet_bar = (
                page.locator(".signal-bars i").nth(3).evaluate("el => parseFloat(el.style.height)")
            )
            assert quiet_bar < active_bar, "listening signal did not settle in silence"
            page.evaluate("testCapture.gain.gain.value = .3")
            expect(page.locator("#voice-interrupt")).to_be_disabled()
            expect(page.locator("#voice-interrupt")).to_have_text("Stop speaking")
            expect(page.locator("#voice-interrupt")).to_be_hidden()
            expect(page.locator("#voice-start")).not_to_have_class("primary")
            compact_anchor = page.locator("#voice-status").bounding_box()
            controls = page.locator("#voice-controls").bounding_box()
            surface_y = page.locator(".assistant-layout").bounding_box()["y"]
            status_offset = compact_anchor["y"] - surface_y
            controls_offset = controls["y"] - surface_y
            transcript = {
                "type": "transcript",
                "data": {
                    "schema_version": 1,
                    "kind": "conversation-transcript",
                    "display_authorized": True,
                    "session_id": "mock-session",
                    "revision": 1,
                    "mode": "mock",
                    "backend_modes": {"asr": "mock", "llm": "mock", "tts": "mock"},
                    "state": "listening",
                    "updated_utc": "2026-10-29T00:00:00Z",
                    "messages": [
                        {
                            "id": "1:user",
                            "turn_id": 1,
                            "role": "user",
                            "text": "What is a Raspberry Pi?",
                            "confirmed_text": "",
                            "status": "final",
                            "delivery": "asr_final",
                        }
                    ],
                },
            }
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".message-user .message-text")).to_have_text(
                "What is a Raspberry Pi?"
            )
            capture_matrix(page, out, "active-conversation")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'thinking',run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:0,status_sequence:2})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Thinking…")
            thinking_anchor = page.locator("#voice-status").bounding_box()
            thinking_surface_y = page.locator(".assistant-layout").bounding_box()["y"]
            assert abs((thinking_anchor["y"] - thinking_surface_y) - status_offset) < 2
            assert (
                abs(
                    (page.locator("#voice-controls").bounding_box()["y"] - thinking_surface_y)
                    - controls_offset
                )
                < 2
            )
            capture_matrix(page, out, "thinking")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'speaking',run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:0,status_sequence:3})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Speaking…")
            silent_bar = page.locator(".signal-bars i").first.evaluate(
                "el => parseFloat(el.style.height)"
            )
            expect(page.locator("#voice-interrupt")).to_be_enabled()
            speaking_surface_y = page.locator(".assistant-layout").bounding_box()["y"]
            assert (
                abs(
                    (page.locator("#voice-status").bounding_box()["y"] - speaking_surface_y)
                    - status_offset
                )
                < 2
            )
            assert (
                abs(
                    (page.locator("#voice-controls").bounding_box()["y"] - speaking_surface_y)
                    - controls_offset
                )
                < 2
            )
            page.wait_for_timeout(250)
            assert not page.evaluate("testSocket.sent.some(row => row.type === 'interrupt')"), (
                "microphone amplitude interrupted playback"
            )
            expect(page.locator("#mic-indicator")).to_have_text("Microphone on")
            expect(page.locator("#mic-indicator")).to_have_attribute("data-activity", "true")
            assert not page.evaluate("testSocket.sent.some(row => row.type === 'interrupt')")
            transcript["data"]["revision"] = 2
            transcript["data"]["state"] = "speaking"
            transcript["data"]["messages"].append(
                {
                    "id": "1:assistant",
                    "turn_id": 1,
                    "role": "assistant",
                    "text": "A Raspberry Pi is a small computer.",
                    "confirmed_text": "",
                    "status": "partial",
                    "delivery": "generated",
                    "generation_complete": False,
                }
            )
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".message-assistant .pill")).to_have_text("Draft")
            expect(page.locator(".message-draft")).to_have_css("font-style", "normal")
            expect(page.locator(".transcript-note")).to_have_count(0)
            expect(page.locator(".app-shell")).to_have_attribute("data-ui-build", "swift-studio-14")
            page.screenshot(path=str(out / "mock-live-draft-1280x720.png"))
            assert page.locator("#view-assistant").get_attribute("data-conversation") == "true"
            transcript["data"]["revision"] = 3
            transcript["data"]["messages"][1]["generation_complete"] = True
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".message-assistant .pill")).to_have_count(0)
            expect(page.locator("#session-summary")).to_have_text("Speaking…")
            expect(page.locator(".message-assistant .message-text").first).to_have_text(
                "A Raspberry Pi is a small computer."
            )
            capture_matrix(page, out, "speaking")
            transcript["data"]["messages"].extend(
                [
                    {
                        "id": "2:user",
                        "turn_id": 2,
                        "role": "user",
                        "text": "Set a demo timer for ten seconds.",
                        "confirmed_text": "",
                        "status": "final",
                        "delivery": "asr_final",
                    },
                    {
                        "id": "2:assistant",
                        "turn_id": 2,
                        "role": "assistant",
                        "text": "A simulated timer is set for ten seconds.",
                        "confirmed_text": "A simulated timer is set for ten seconds.",
                        "status": "final",
                        "delivery": "mock_sink_confirmed",
                        "generation_complete": True,
                    },
                ]
            )
            transcript["data"]["revision"] = 4
            transcript["data"]["tasks"] = [
                {
                    "id": "2:timer-call",
                    "turn_id": 2,
                    "call_id": "timer-call",
                    "kind": "set_timer",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Set a timer",
                    "summary": "Simulated 10-second timer; no alert will ring.",
                    "display": "10 seconds",
                    "widget": {
                        "type": "timer",
                        "schedule_kind": "timer",
                        "active": True,
                        "deadline_utc": (
                            datetime.now(timezone.utc) + timedelta(seconds=30)
                        ).isoformat(),
                        "target_local": "",
                        "display_value": "10 seconds",
                    },
                },
            ]
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator("#task-activity")).to_be_visible()
            expect(page.locator(".task-card .task-copy strong")).to_have_text(["Set a timer"])
            expect(page.locator(".task-card[data-latest='true'] .task-recency")).to_have_text(
                "Latest result"
            )
            expect(page.locator(".task-timer .task-widget-kicker")).to_have_text(
                "SIMULATED TIMER · NO ALERT"
            )
            expect(page.locator(".task-timer .timer-value")).to_have_attribute(
                "data-timer-deadline", transcript["data"]["tasks"][0]["widget"]["deadline_utc"]
            )
            assert re.fullmatch(
                r"\d{2}:\d{2}", page.locator(".task-timer .timer-value").inner_text()
            )
            expect(page.locator(".task-card[data-latest='true'] .task-evidence")).to_have_text(
                "Simulated"
            )
            assert page.locator(".transcript-column").evaluate(
                "el => el.getBoundingClientRect().width <= 740"
            )
            transcript["data"]["revision"] = 5
            transcript["data"]["messages"].extend(
                [
                    {
                        "id": "3:user",
                        "turn_id": 3,
                        "role": "user",
                        "text": "Turn on the light.",
                        "confirmed_text": "",
                        "status": "final",
                        "delivery": "asr_final",
                    },
                    {
                        "id": "3:assistant",
                        "turn_id": 3,
                        "role": "assistant",
                        "text": "The simulated light is on.",
                        "confirmed_text": "The simulated light is on.",
                        "status": "final",
                        "delivery": "mock_sink_confirmed",
                        "generation_complete": True,
                    },
                ]
            )
            transcript["data"]["tasks"].append(
                {
                    "id": "3:light-on-call",
                    "turn_id": 3,
                    "call_id": "light-on-call",
                    "kind": "light_on",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Turn on the light",
                    "summary": "Light is on in the simulation.",
                    "display": "blue",
                    "widget": {"type": "lamp", "on": True, "color": "blue", "brightness": 60},
                }
            )
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card[data-latest='true'] .task-light")).to_have_attribute(
                "data-light-state", "on"
            )
            expect(
                page.locator(".task-card[data-latest='true'] .task-light .task-widget-kicker")
            ).to_have_text("SIMULATED RGB LED")
            module = page.locator(".task-card[data-latest='true'] .rgb-module-illustration svg")
            expect(module.locator(".module-lens-color")).to_have_attribute("fill", "#0000FF")
            expect(module.locator(".module-resistor")).to_have_count(3)
            expect(module.locator(".module-pad")).to_have_count(4)
            expect(
                page.locator(".task-card[data-latest='true'] .task-light .task-widget-copy strong")
            ).to_have_text("ON · blue · 60%")
            capture_matrix(page, out, "task-light-on")
            controls = page.locator(".task-card[data-latest='true'] .device-controls")
            expect(controls.locator("summary")).to_have_text("RGB controls · mock simulation")
            controls.locator("summary").click()
            expect(controls.locator(".device-control")).to_have_count(4)
            expect(controls.locator("[data-light-brightness]")).to_have_attribute("min", "0")
            expect(controls.locator("[data-light-brightness]")).to_have_attribute("max", "100")
            expect(controls.locator("[data-light-brightness]")).to_have_attribute("step", "1")
            controls.locator("[data-light-color]").evaluate(
                "el => {el.value='#12ab34';el.dispatchEvent(new Event('input',{bubbles:true}));}"
            )
            expect(controls.locator("[data-color-output]")).to_have_text("#12AB34")
            controls.get_by_role("button", name="Apply color").click()
            color_command = page.evaluate(
                "testSocket.sent.filter(row=>row.type==='device_ui_action').at(-1)"
            )
            assert color_command["action"] == "set_color" and color_command["value"] == "#12AB34"
            page.evaluate(
                "commandId => testSocket.emit({type:'device_action_ack',status:'accepted',command_id:commandId})",
                color_command["command_id"],
            )
            expect(controls.locator("small")).to_have_text(
                "Command accepted by the voice session; waiting for its action receipt."
            )
            page.evaluate(
                "commandId => testSocket.emit({type:'runtime_event',sequence:1,data:{run_id:'mock-run',session_id:'mock-session',turn_id:1,event:'tool_completed',tool_call_id:commandId}})",
                color_command["command_id"],
            )
            controls.locator("[data-light-brightness]").evaluate(
                "el => {el.value='37';el.dispatchEvent(new Event('input',{bubbles:true}));}"
            )
            expect(controls.locator("[data-brightness-output]")).to_have_text("37%")
            controls.get_by_role("button", name="Apply brightness").click()
            brightness_command = page.evaluate(
                "testSocket.sent.filter(row=>row.type==='device_ui_action').at(-1)"
            )
            assert (
                brightness_command["action"] == "set_brightness"
                and brightness_command["value"] == "37"
            )
            page.evaluate(
                "commandId => testSocket.emit({type:'device_action_ack',status:'accepted',command_id:commandId})",
                brightness_command["command_id"],
            )
            page.evaluate(
                "commandId => testSocket.emit({type:'runtime_event',sequence:2,data:{run_id:'mock-run',session_id:'mock-session',turn_id:1,event:'tool_completed',tool_call_id:commandId}})",
                brightness_command["command_id"],
            )
            controls.get_by_role("button", name="Off").click()
            device_command = page.evaluate(
                "testSocket.sent.filter(row=>row.type==='device_ui_action').at(-1)"
            )
            assert re.fullmatch(r"ui-[a-f0-9]{32}", device_command["command_id"])
            assert device_command["action"] == "light_off" and "value" not in device_command
            # The browser emits only a typed command; the bridge acknowledgement
            # is a separate mock event, as in the live protocol.
            page.evaluate(
                "commandId => testSocket.emit({type:'device_action_ack',status:'accepted',command_id:commandId})",
                device_command["command_id"],
            )
            expect(controls.locator("small")).to_have_text(
                "Command accepted by the voice session; waiting for its action receipt."
            )
            expect(controls).to_have_attribute("open", "")
            assert controls.locator("button").evaluate_all(
                "buttons => buttons.every(button => button.disabled)"
            )
            page.screenshot(path=str(out / "mock-device-control-pending-1280x720.png"))
            transcript["data"]["revision"] = 6
            transcript["data"]["messages"].extend(
                [
                    {
                        "id": "4:assistant",
                        "turn_id": 4,
                        "role": "assistant",
                        "text": "The simulated RGB LED is off.",
                        "confirmed_text": "The simulated RGB LED is off.",
                        "status": "final",
                        "delivery": "mock_sink_confirmed",
                        "generation_complete": True,
                    },
                ]
            )
            transcript["data"]["tasks"].append(
                {
                    "id": "4:" + device_command["command_id"],
                    "turn_id": 4,
                    "call_id": device_command["command_id"],
                    "kind": "light_off",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Turn off the light",
                    "summary": "RGB LED is off in the simulation.",
                    "display": "off",
                    "widget": {"type": "lamp", "on": False, "color": "#12AB34", "brightness": 37},
                }
            )
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            page.evaluate(
                """commandId => testSocket.emit({type:'runtime_event',sequence:3,data:{run_id:'mock-run',session_id:'mock-session',turn_id:1,event:'tool_completed',tool_call_id:commandId}})""",
                device_command["command_id"],
            )
            expect(page.locator(".task-card[data-latest='true'] .task-light")).to_have_attribute(
                "data-light-state", "off"
            )
            expect(
                page.locator(".task-card[data-latest='true'] .task-light .task-widget-copy strong")
            ).to_have_text("OFF · #12AB34 · 37%")
            expect(controls.locator("small")).to_have_text(
                "Mock simulation · no physical hardware is changed."
            )
            expect(controls.locator("button").first).to_be_enabled()
            capture_matrix(page, out, "task-light-off")
            transcript["data"]["revision"] = 7
            transcript["data"]["messages"].extend(
                [
                    {
                        "id": "5:user",
                        "turn_id": 5,
                        "role": "user",
                        "text": "Create a Swift Demo directory.",
                        "confirmed_text": "",
                        "status": "final",
                        "delivery": "asr_final",
                    },
                    {
                        "id": "5:assistant",
                        "turn_id": 5,
                        "role": "assistant",
                        "text": "The example directory receipt is shown alongside this request.",
                        "confirmed_text": "The example directory receipt is shown alongside this request.",
                        "status": "final",
                        "delivery": "mock_sink_confirmed",
                        "generation_complete": True,
                    },
                ]
            )
            transcript["data"]["tasks"] = [
                {
                    "id": "2:timer-call",
                    "turn_id": 2,
                    "call_id": "timer-call",
                    "kind": "set_timer",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Set a timer",
                    "summary": "Simulated 10-second timer; no alert will ring.",
                    "display": "10 seconds",
                },
                {
                    "id": "3:light-on-call",
                    "turn_id": 3,
                    "call_id": "light-on-call",
                    "kind": "light_on",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Turn on the light",
                    "summary": "Light is on in the simulation.",
                    "display": "blue",
                },
                {
                    "id": "4:light-off-call",
                    "turn_id": 4,
                    "call_id": "light-off-call",
                    "kind": "light_off",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Turn off the light",
                    "summary": "Light is off in the simulation.",
                    "display": "off",
                },
                {
                    "id": "5:project-call",
                    "turn_id": 5,
                    "call_id": "project-call",
                    "kind": "agent",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Create Swift Demo directory",
                    "summary": "Example directory receipt; no filesystem operation ran.",
                    "display": "",
                    "execution_target": "Mock fixture workspace",
                    "absolute_path": "/mock-fixture/Swift-Demo",
                    "verification_level": "Mock fixture only; path was not checked.",
                },
            ]
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card .task-copy strong")).to_have_text(
                [
                    "Create Swift Demo directory",
                    "Turn off the light",
                    "Turn on the light",
                    "Set a timer",
                ]
            )
            expect(page.locator(".task-card[data-latest='true'] .task-recency")).to_have_text(
                "Latest result"
            )
            expect(page.get_by_text("/mock-fixture/Swift-Demo")).to_be_visible()
            capture_matrix(page, out, "task-output")
            transcript["data"]["revision"] = 8
            transcript["data"]["messages"].extend(
                [
                    {
                        "id": "6:user",
                        "turn_id": 6,
                        "role": "user",
                        "text": "Create a small website in the project folder.",
                        "confirmed_text": "",
                        "status": "final",
                        "delivery": "asr_final",
                    },
                    {
                        "id": "6:assistant",
                        "turn_id": 6,
                        "role": "assistant",
                        "text": "The mock draft artifact can be inspected or downloaded.",
                        "confirmed_text": "The mock draft artifact can be inspected or downloaded.",
                        "status": "final",
                        "delivery": "mock_sink_confirmed",
                        "generation_complete": True,
                    },
                ]
            )
            transcript["data"]["tasks"] = [
                {
                    "id": "2:timer-call",
                    "turn_id": 2,
                    "call_id": "timer-call",
                    "kind": "set_timer",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Set a timer",
                    "summary": "Simulated 10-second timer; no alert will ring.",
                    "display": "10 seconds",
                },
                {
                    "id": "3:light-on-call",
                    "turn_id": 3,
                    "call_id": "light-on-call",
                    "kind": "light_on",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Turn on the light",
                    "summary": "Light is on in the simulation.",
                    "display": "blue",
                },
                {
                    "id": "4:light-off-call",
                    "turn_id": 4,
                    "call_id": "light-off-call",
                    "kind": "light_off",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Turn off the light",
                    "summary": "Light is off in the simulation.",
                    "display": "off",
                },
                {
                    "id": "5:project-call",
                    "turn_id": 5,
                    "call_id": "project-call",
                    "kind": "agent",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Create Swift Demo directory",
                    "summary": "Example directory receipt; no filesystem operation ran.",
                    "display": "",
                    "execution_target": "Mock fixture workspace",
                    "absolute_path": "/mock-fixture/Swift-Demo",
                    "verification_level": "Mock fixture only; path was not checked.",
                },
                {
                    "id": "6:agent-task",
                    "turn_id": 6,
                    "call_id": "agent-task",
                    "kind": "agent",
                    "status": "reported",
                    "mode": "mock",
                    "title": "Create a small website",
                    "summary": "Mock artifact receipt; content was not executed or deployed.",
                    "display": "",
                    "execution_target": "Mock fixture workspace",
                    "absolute_path": "/mock-fixture/task-" + "b" * 24,
                    "verification_level": "Mock fixture only; path and file were not checked.",
                    "artifact_id": "b" * 24,
                    "artifacts": ["index.html"],
                },
            ]
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card[data-latest='true'] .task-evidence")).to_have_text(
                "Mock · unverified"
            )
            expect(page.locator(".task-card .task-recency").first).to_have_text("Latest result")
            expect(page.locator(".task-card .task-recency").nth(1)).to_have_text(
                "Earlier result · Turn 5"
            )
            expect(page.get_by_text("/mock-fixture/task-" + "b" * 24)).to_be_visible()
            expect(page.locator(".task-location dt").last).to_have_text("Verification status")
            expect(page.locator(".task-file-preview")).to_have_text("index.html")
            expect(page.locator(".task-files a")).to_have_attribute("download", "index.html")
            page.locator(".task-file-preview").click()
            expect(page.get_by_role("dialog", name="index.html")).to_be_visible()
            expect(page.locator("#artifact-content")).to_have_text(
                "<h1>Draft content is shown as text</h1>"
            )
            assert page.locator("#artifact-content h1").count() == 0
            page.screenshot(path=str(out / "mock-task-artifact-preview-desktop.png"))
            page.locator("#close-artifact").click()
            capture_matrix(page, out, "task-output")
            transcript["data"]["revision"] = 9
            transcript["data"]["messages"].append(
                {
                    "id": "7:user",
                    "turn_id": 7,
                    "role": "user",
                    "text": "Revise the website copy.",
                    "confirmed_text": "",
                    "status": "final",
                    "delivery": "asr_final",
                }
            )
            transcript["data"]["tasks"].append(
                {
                    "id": "7:agent-task",
                    "turn_id": 7,
                    "call_id": "agent-task",
                    "kind": "agent",
                    "status": "running",
                    "mode": "mock",
                    "title": "Revise the website copy",
                    "summary": "Mock request is pending; no work has been performed.",
                    "display": "",
                }
            )
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card[data-latest='true'] .task-recency")).to_contain_text(
                "Task "
            )
            page.evaluate(
                "() => { const node=document.querySelector('#task-items'); node.scrollTop=node.scrollHeight; }"
            )
            assert page.locator("#task-items").evaluate("el => el.scrollTop > 0")
            transcript["data"]["revision"] = 10
            transcript["data"]["tasks"][-1].update(
                {
                    "status": "failed",
                    "summary": "Mock request failed before execution.",
                }
            )
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card[data-latest='true'] .task-recency")).to_have_text(
                "Failed most recently · Turn 7"
            )
            expect(page.locator(".task-card[data-latest='true'] .task-evidence")).to_have_text(
                "Failed · Mock"
            )
            assert page.locator("#task-items").evaluate("el => el.scrollTop === 0")
            expect(page.locator(".task-card")).to_have_count(6)
            capture_matrix(page, out, "task-output-failed")
            page.set_viewport_size({"width": 1280, "height": 720})
            stale = json.loads(json.dumps(transcript))
            stale["data"]["revision"] = 1
            stale["data"]["messages"][0]["text"] = "Late old text."
            page.evaluate("payload => testSocket.emit(payload)", stale)
            expect(page.locator(".message-assistant .message-text").first).to_have_text(
                "A Raspberry Pi is a small computer."
            )
            page.evaluate("""() => {
              const pcm = new ArrayBuffer(8196), view = new DataView(pcm); view.setUint32(0, 7, true);
              for(let i=0;i<4096;i++) view.setInt16(4+i*2, 12000, true);
              testSocket.onmessage({data:pcm});
            }""")
            expect(page.locator("#voice-status")).to_have_attribute("data-wave", "true")
            expect(page.locator("#voice-status")).to_have_attribute("data-level-source", "playback")
            expect(page.locator(".signal-bars i").first).to_have_css("width", "3px")
            assert (
                page.locator(".signal-bars i").first.evaluate("el => parseFloat(el.style.height)")
                > silent_bar
            )
            page.screenshot(path=str(out / "mock-speaking-measured-playback-level.png"))
            page.locator("#voice-interrupt").click()
            expect(page.locator("#session-summary")).to_have_text("Stopping speech…")
            assert (
                page.evaluate("testSocket.sent.filter(row => row.type === 'interrupt').length") == 1
            )
            page.evaluate("""() => {
              const pcm = new ArrayBuffer(8196), view = new DataView(pcm); view.setUint32(0, 8, true);
              for(let i=0;i<4096;i++) view.setInt16(4+i*2, 12000, true);
              testSocket.onmessage({data:pcm});
            }""")
            expect(page.locator("#voice-status")).to_have_attribute("data-wave", "false")
            page.wait_for_timeout(100)
            assert not page.evaluate(
                "testSocket.sent.some(row => row.type === 'audio_ack' && row.id === 7)"
            ), "cancelled playback was acknowledged"
            assert not page.evaluate(
                "testSocket.sent.some(row => row.type === 'audio_ack' && row.id === 8)"
            ), "late cancelled audio was acknowledged"
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:5,data:{event:'playback_stopped_browser',run_id:'mock-run',session_id:'mock-session',turn_id:1}})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Interrupted")
            capture_matrix(page, out, "interrupted")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'listening',run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:5,status_sequence:4})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Listening…")
            expect(page.locator(".message-user .message-text").first).to_have_text(
                "What is a Raspberry Pi?"
            )
            expect(page.locator("#welcome")).to_be_hidden()
            assert 64 <= page.locator(".idle-stage").bounding_box()["height"] <= 80, page.locator(
                ".idle-stage"
            ).bounding_box()
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'thinking',run_id:'mock-run',session_id:'mock-session',turn_id:2,event_sequence:5,status_sequence:5})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Thinking…")
            expect(page.locator(".message-user .message-text").first).to_have_text(
                "What is a Raspberry Pi?"
            )
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'speaking',run_id:'mock-run',session_id:'mock-session',turn_id:2,event_sequence:5,status_sequence:6})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Speaking…")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'thinking',run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:5,status_sequence:3})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Speaking…")
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:6,data:{event:'turn_completed',run_id:'mock-run',session_id:'mock-session',turn_id:2}})"""
            )
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'listening',run_id:'mock-run',session_id:'mock-session',turn_id:2,event_sequence:6,status_sequence:7})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Listening…")
            expect(page.locator("#voice-status-context")).to_contain_text("Reply complete")
            expect(page.locator("#voice-control-note")).to_be_empty()
            oscillators_before_wake = page.evaluate("testOscillatorCount")
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:7,data:{event:'wake_detected_file',run_id:'mock-run',session_id:'next-session',turn_id:0}})"""
            )
            page.wait_for_timeout(20)
            assert page.evaluate("testOscillatorCount") == oscillators_before_wake + 2, (
                "an accepted wake event should produce the two-note confirmation"
            )
            expect(page.locator("#voice-status")).to_have_attribute("data-wake-confirmed", "true")
            page.screenshot(path=str(out / "wake-confirmed-two-tone-and-ring.png"))
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:8,data:{event:'wake_detected_file',run_id:'mock-run',session_id:'next-session',turn_id:0}})"""
            )
            page.wait_for_timeout(20)
            assert page.evaluate("testOscillatorCount") == oscillators_before_wake + 2, (
                "duplicate wake events for one session should not repeat the confirmation"
            )
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'listening',run_id:'mock-run',session_id:'next-session',turn_id:0,event_sequence:8,status_sequence:8})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Listening…")
            expect(page.locator("#voice-status-context")).not_to_contain_text("Reply complete")
            expect(page.locator("#task-activity")).to_be_hidden()
            next_session = {
                "type": "transcript",
                "data": {
                    **transcript["data"],
                    "session_id": "next-session",
                    "revision": 1,
                    "state": "listening",
                    "messages": [
                        {
                            "id": "1:user",
                            "turn_id": 1,
                            "role": "user",
                            "text": "Tell me about the Swift demo.",
                            "confirmed_text": "",
                            "status": "final",
                            "delivery": "asr_final",
                        }
                    ],
                },
            }
            next_session["data"].pop("tasks", None)
            page.evaluate("payload => testSocket.emit(payload)", next_session)
            expect(page.locator(".message-user .message-text")).to_have_text(
                "Tell me about the Swift demo."
            )
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'speaking',run_id:'mock-run',session_id:'mock-session',turn_id:2,event_sequence:7,status_sequence:9})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Listening…")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'speaking',run_id:'mock-run',session_id:'next-session',turn_id:1,event_sequence:8,status_sequence:10})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Speaking…")
            page.evaluate("""() => {
              const pcm = new ArrayBuffer(8196), view = new DataView(pcm); view.setUint32(0, 9, true);
              for(let i=0;i<4096;i++) view.setInt16(4+i*2, 9000, true);
              testSocket.onmessage({data:pcm});
            }""")
            expect(page.locator("#voice-status")).to_have_attribute("data-level-source", "playback")
            page.locator("#voice-mute").click()
            expect(page.locator("#mic-indicator")).to_have_text("Microphone muted")
            expect(page.locator("#voice-status")).to_have_attribute("data-level-source", "playback")
            expect(page.locator("#voice-status")).to_have_attribute("data-wave", "true")
            expect(page.locator(".signal-bars")).to_be_visible()
            page.locator("#voice-mute").click()
            expect(page.locator("#mic-indicator")).to_have_text("Microphone on")
            page.emulate_media(reduced_motion="reduce")
            assert (
                page.locator(".voice-glyph").evaluate(
                    "el => getComputedStyle(el).transitionDuration"
                )
                == "0s"
            )
            assert (
                page.locator(".signal-bars").evaluate(
                    "el => getComputedStyle(el).animationDuration"
                )
                == "0s"
            )
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(out / "mock-live-speaking-narrow-reduced-motion.png"))
            page.locator("#voice-start").click()
            expect(page.locator("#session-summary")).to_have_text("Conversation ended")
            expect(page.locator("#voice-status")).to_have_attribute("data-live", "false")
            assert page.locator("#view-assistant").get_attribute("data-conversation") == "true"
            expect(page.locator("#voice-start")).to_have_text("Start conversation")
            expect(page.locator("#voice-interrupt")).to_be_hidden()
            expect(page.locator("#voice-mute")).to_be_hidden()
            capture_matrix(page, out, "ended-with-history")
            page.locator("#voice-start").click()
            expect(page.locator("#session-summary")).to_have_text("Waiting for ‘Hey Swift’")
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'thinking',run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:0,status_sequence:1})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Thinking…")
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:1,data:{event:'tool_started',run_id:'mock-run',session_id:'mock-session',turn_id:1}})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Working on your request…")
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:2,data:{event:'tool_completed',run_id:'mock-run',session_id:'mock-session',turn_id:1}})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Thinking…")
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:3,data:{event:'tool_started',run_id:'mock-run',session_id:'old-session',turn_id:1}})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Thinking…")
            transcript["data"]["messages"] = [
                {
                    "id": "1:user",
                    "turn_id": 1,
                    "role": "user",
                    "text": "Play music",
                    "confirmed_text": "",
                    "status": "final",
                    "delivery": "asr_final",
                }
            ]
            transcript["data"]["tasks"] = []
            transcript["data"]["revision"] = 99
            transcript["data"]["tasks"].append(
                {
                    "id": "1:music-playlist-99",
                    "turn_id": 1,
                    "call_id": "music-playlist-99",
                    "kind": "music_playlist",
                    "status": "simulated",
                    "mode": "mock",
                    "title": "Play a playlist",
                    "summary": "Simulated playlist search: chill. No audio is played.",
                    "display": "playlist",
                }
            )
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card[data-latest='true'] .task-music")).to_contain_text(
                "chill"
            )
            expect(page.locator(".task-card[data-latest='true'] .task-widget-kicker")).to_have_text(
                "SIMULATED MUSIC · NO AUDIO"
            )
            page.screenshot(path=str(out / "mock-music-playlist-1280x720.png"))
            for revision, artist in ((100, "Taylor Swift"), (101, "Bruno Mars")):
                transcript["data"]["revision"] = revision
                transcript["data"]["tasks"].append(
                    {
                        "id": f"1:music-{revision}",
                        "turn_id": 1,
                        "call_id": f"music-{revision}",
                        "kind": "music_play",
                        "status": "simulated",
                        "mode": "mock",
                        "title": "Control music",
                        "summary": f"Simulated music play: {artist}. No audio is played.",
                        "display": "play",
                    }
                )
                page.evaluate("payload => testSocket.emit(payload)", transcript)
                expect(page.locator(".task-card[data-latest='true'] .task-music")).to_contain_text(
                    artist
                )
                expect(
                    page.locator(".task-card[data-latest='true'] .task-widget-kicker")
                ).to_have_text("SIMULATED MUSIC · NO AUDIO")
                capture_matrix(page, out, "music-correction" if revision == 101 else "music-first")
            for revision, status, display, summary in (
                (102, "running", "", "Requesting music playback."),
                (
                    103,
                    "reported",
                    "playing",
                    "Device reports playing. Audible playback is unverified.",
                ),
                (
                    104,
                    "failed",
                    "",
                    "Open Spotify on the laptop and select an active playback device",
                ),
                (105, "unknown", "", "Outcome not confirmed after disconnect."),
            ):
                transcript["data"]["revision"] = revision
                transcript["data"]["tasks"][-1].update(
                    status=status, mode="reported", display=display, summary=summary
                )
                page.evaluate("payload => testSocket.emit(payload)", transcript)
                expect(page.locator(".task-card[data-latest='true']")).to_have_attribute(
                    "data-status", status
                )
                if status == "failed":
                    expect(
                        page.locator(".task-card[data-latest='true'] .task-recovery")
                    ).to_be_visible()
                elif status == "unknown":
                    expect(
                        page.locator(".task-card[data-latest='true'] .task-recovery")
                    ).to_have_count(0)
                    expect(
                        page.locator(".task-card[data-latest='true'] .task-music")
                    ).to_contain_text("Request outcome uncertain")
                capture_matrix(page, out, "music-" + status)
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:4,data:{event:'turn_error',run_id:'mock-run',session_id:'mock-session',turn_id:1,error_role:'llm',error_code:'provider_rate_limited'}})"""
            )
            expect(page.locator("#notice")).to_contain_text(
                "Reasoning is rate-limiting requests. Wait before trying again"
            )
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:5,data:{event:'turn_error',run_id:'mock-run',session_id:'mock-session',turn_id:1,error_role:'llm',error_code:'request_limit_reached',reasoning_budget:{used:2,limit:2,remaining:0,scope:'bridge runtime'}}})"""
            )
            expect(page.locator("#notice")).to_contain_text(
                "Reasoning request allowance is exhausted for this bridge run."
            )
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:6,data:{event:'turn_error',run_id:'mock-run',session_id:'mock-session',turn_id:1,error_role:'llm',error_code:'provider_invalid_response',error_stage:'response_parse',error_detail:'invalid_tool_arguments'}})"""
            )
            expect(page.locator("#notice")).to_contain_text(
                "Reasoning returned invalid task arguments. Check task results before retrying."
            )
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:7,data:{event:'llm_empty_response',run_id:'mock-run',session_id:'mock-session',turn_id:1}})"""
            )
            expect(page.locator("#notice")).to_contain_text(
                "Reasoning returned no answer. Swift asked you to try again."
            )
            transcript["data"]["revision"] = 106
            transcript["data"]["tasks"] = [
                {
                    "id": "1:music-stop",
                    "turn_id": 1,
                    "call_id": "music-stop",
                    "kind": "music_stop",
                    "status": "unknown",
                    "mode": "reported",
                    "title": "Stop music",
                    "summary": (
                        "Spotify could not confirm the stop request; the browser player now "
                        "reports paused. Whether the request caused this state remains unverified."
                    ),
                    "display": "stop",
                }
            ]
            page.evaluate("payload => testSocket.emit(payload)", transcript)
            expect(page.locator(".task-card[data-latest='true'] .task-music strong")).to_have_text(
                "Player reports paused"
            )
            expect(page.locator(".task-card[data-latest='true'] .task-music")).to_contain_text(
                "Audible sound is not verified"
            )
            expect(page.locator(".task-card[data-latest='true'] .task-recovery")).to_have_count(0)
            page.evaluate(
                """() => testSocket.emit({type:'runtime_event',sequence:8,data:{event:'turn_error',run_id:'mock-run',session_id:'mock-session',turn_id:1,error_role:'llm',error_stage:'tool',error_code:'music_outcome_uncertain'}})"""
            )
            expect(page.locator("#notice")).to_be_hidden()
            page.screenshot(path=str(out / "spotify-stop-uncertain-player-paused.png"))
            page.locator(".task-card[data-latest='true']").screenshot(
                path=str(out / "spotify-stop-uncertain-task-card.png")
            )
            page.evaluate(
                """() => testSocket.emit({type:'status',state:'listening',input_open:false,engagement:{mode:'followup',remaining_seconds:5,window_seconds:7,awaiting_answer:false},run_id:'mock-run',session_id:'mock-session',turn_id:1,event_sequence:8,status_sequence:100})"""
            )
            expect(page.locator("#session-summary")).to_have_text("Listening for a follow-up")
            expect(page.locator("#voice-status")).to_have_attribute("data-waiting", "true")
            page.screenshot(path=str(out / "mock-followup-window.png"))
            sockets_before_disconnect = page.evaluate("window.testSocketCount")
            page.evaluate("testSocket.close()")
            expect(page.locator("#session-summary")).to_have_text("Recorded session")
            expect(page.locator("#voice-status")).to_have_attribute("data-live", "false")
            expect(page.locator("#notice")).to_contain_text("Checking for recovery")
            assert page.evaluate("window.testSocketCount") == sockets_before_disconnect
            expect(page.locator("#notice")).to_contain_text("reachable again")
            assert page.evaluate("window.testSocketCount") == sockets_before_disconnect
            page.locator("#voice-start").click()
            expect(page.locator("#session-summary")).to_have_text("Waiting for ‘Hey Swift’")
            assert page.evaluate("window.testSocketCount") == sockets_before_disconnect + 1
            page.evaluate(
                "window.testDeviceCapabilities={target:'pi',status:'unavailable',available:false,physical_readback:false,message:'Raspberry Pi is not connected'}"
            )
            page.locator("#voice-start").click()
            expect(page.locator("#session-summary")).to_have_text("Conversation ended")
            page.locator("#voice-start").click()
            expect(page.locator("#session-summary")).to_have_text("Waiting for ‘Hey Swift’")
            expect(page.locator("#device-adapter-status")).to_have_text(
                "Raspberry Pi is not connected"
            )
            unavailable = {
                "type": "transcript",
                "data": {
                    "schema_version": 1,
                    "kind": "conversation-transcript",
                    "display_authorized": True,
                    "session_id": "mock-session",
                    "revision": 1,
                    "mode": "mock",
                    "backend_modes": {"asr": "mock", "llm": "mock", "tts": "mock"},
                    "state": "listening",
                    "updated_utc": "2026-11-04T00:00:00Z",
                    "messages": [
                        {
                            "id": "1:user",
                            "turn_id": 1,
                            "role": "user",
                            "text": "Turn on the light",
                            "confirmed_text": "",
                            "status": "final",
                            "delivery": "asr_final",
                        }
                    ],
                    "tasks": [
                        {
                            "id": "1:pi-light",
                            "turn_id": 1,
                            "call_id": "pi-light",
                            "kind": "light_on",
                            "status": "unavailable",
                            "mode": "reported",
                            "title": "Turn on the light",
                            "summary": "Raspberry Pi is not connected. No device action was sent.",
                            "display": "",
                        }
                    ],
                },
            }
            page.evaluate("payload => testSocket.emit(payload)", unavailable)
            controls = page.locator(".task-card[data-latest='true'] .device-controls")
            expect(controls.locator("summary")).to_have_text("RGB controls · Raspberry Pi")
            expect(controls.locator("small")).to_contain_text("Controls are disabled")
            expect(controls.locator("button")).to_have_count(4)
            assert controls.locator("button").evaluate_all(
                "buttons => buttons.every(button => button.disabled)"
            )
            page.set_viewport_size({"width": 1280, "height": 720})
            page.screenshot(path=str(out / "mock-pi-unavailable-controls-1280x720.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(
                path=str(out / "mock-pi-unavailable-controls-390x844.png"), full_page=True
            )
            page.locator("#voice-start").click()
            expect(page.locator("#session-summary")).to_have_text("Conversation ended")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print(
        "Voice UI mock contract: ready, multi-turn, duplex mic, confirmed interruption, stale-turn fence, mute, end, reconnect, disconnect, reduced motion: passed"
    )


if __name__ == "__main__":
    main()
