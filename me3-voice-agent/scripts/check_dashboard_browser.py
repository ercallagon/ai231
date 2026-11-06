"""Actual browser checks for the assistant UI and terminal report-file adapter."""

import argparse
from copy import deepcopy
import json
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from me3_voice.artifacts import output_path
from me3_voice.dashboard import handler, render, snapshot
from me3_voice.contracts import TurnKey
from me3_voice.transcript import Transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=output_path, default="var/results/ui-check")
    args = parser.parse_args()
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import expect, sync_playwright

    root = Path.cwd()
    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args: None)
    server.RequestHandlerClass = handler(root, server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            errors, unexpected = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "console",
                lambda message: errors.append(message.text) if message.type == "error" else None,
            )
            page.on(
                "request",
                lambda request: (
                    unexpected.append(request.url)
                    if not request.url.startswith(
                        (f"http://127.0.0.1:{server.server_port}/", "file:", "data:", "blob:")
                    )
                    else None
                ),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/")

            def capture(name, *, projected=False):
                original = page.viewport_size
                sizes = [("laptop", 1366, 768), ("narrow", 390, 844)]
                if projected:
                    sizes.extend(
                        [("projector", 1920, 1080), ("small", 320, 740), ("zoom-layout", 683, 384)]
                    )
                for label, width, height in sizes:
                    page.set_viewport_size({"width": width, "height": height})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                        name,
                        label,
                    )
                    assert page.evaluate("""() => [...document.querySelectorAll('body *')].every(el => {
                        const hasText=[...el.childNodes].some(n=>n.nodeType===3 && n.textContent.trim());
                        return !hasText || !el.getClientRects().length || getComputedStyle(el).visibility==='hidden' || parseFloat(getComputedStyle(el).fontSize)>=12;
                    })"""), (name, label, "text smaller than 12px")
                    page.screenshot(path=str(directory / f"{name}-{label}.png"), full_page=True)
                page.set_viewport_size(original)

            source = snapshot(root)
            page.get_by_role("heading", name="Talk it through.").wait_for()
            expect(page.locator("#task-activity")).to_be_hidden()
            expect(page.locator("#history-select")).to_have_value("")
            assert page.locator("nav a").count() == 2
            assert page.get_by_role("textbox", name="Message Swift").count() == 0
            expect(page.get_by_text("Voice controls unavailable", exact=True)).to_be_visible()
            expect(page.locator("#session-summary")).to_contain_text("not connected")
            expect(page.locator("#report-sources")).to_be_hidden()
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            expect(page.locator("#report-sources")).to_be_visible()
            capture("diagnostics-report-sources")
            checks.append(
                "report import/feed controls and status live in Diagnostics, separate from Assistant"
            )
            page.get_by_role("link", name="Assistant", exact=True).click()
            page.screenshot(path=str(directory / "assistant-desktop.png"), full_page=True)
            capture("assistant", projected=True)
            page.get_by_role("link", name="Swift assistant").focus()
            page.keyboard.press("Tab")
            assert page.evaluate(
                "getComputedStyle(document.activeElement).outlineStyle === 'solid'"
            )
            page.mouse.move(0, 0)
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            connect = page.get_by_role("button", name="Connect report feed")
            base_background = connect.evaluate("el=>getComputedStyle(el).backgroundColor")
            connect.hover()
            assert connect.evaluate("el=>getComputedStyle(el).backgroundColor") != base_background
            connect.click()
            expect(page.get_by_role("dialog", name="Connect report updates")).to_contain_text(
                "does not connect a microphone"
            )
            capture("connection-dialog", projected=True)
            for _ in range(6):
                page.keyboard.press("Tab")
                assert page.evaluate(
                    "Boolean(document.activeElement.closest('#connection-dialog'))"
                )
            page.keyboard.press("Escape")
            expect(page.get_by_role("button", name="Connect report feed")).to_be_focused()
            checks.append(
                "two-view assistant, readable type, keyboard focus and report-only connection dialog"
            )

            contrast_js = """() => {
                const styles=getComputedStyle(document.documentElement);
                const lum=hex=>{const rgb=hex.trim().replace('#','').match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2];};
                const pairs=[['--ink','--panel'],['--muted','--panel'],['--accent','--panel'],['--accent','--selected'],['--accent-hover','--panel'],['--muted','--neutral'],['--accent','--canvas'],['--success','--success-soft'],['--amber','--warning-soft'],['--error','--error-soft']];
                return pairs.map(([fg,bg])=>{const a=lum(styles.getPropertyValue(fg)),b=lum(styles.getPropertyValue(bg));return {fg,bg,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};});
            }"""
            contrast = page.evaluate(contrast_js)
            assert all(pair["ratio"] >= 4.5 for pair in contrast), contrast
            (directory / "contrast.json").write_text(json.dumps(contrast, indent=2) + "\n")
            checks.append("primary, secondary, accent and semantic text token contrasts meet 4.5:1")

            theme_select = page.get_by_role("combobox", name="Color theme")
            theme_select.select_option("light")
            theme_select.select_option("dark")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            dark_contrast = page.evaluate(contrast_js)
            assert all(pair["ratio"] >= 4.5 for pair in dark_contrast), dark_contrast
            (directory / "contrast-dark.json").write_text(
                json.dumps(dark_contrast, indent=2) + "\n"
            )
            capture("assistant-dark")
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            capture("diagnostics-dark")
            page.get_by_role("link", name="Assistant", exact=True).click()
            theme_select.select_option("system")
            page.emulate_media(color_scheme="dark")
            expect(theme_select.locator("option:checked")).to_have_text("System (Dark)")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            page.emulate_media(color_scheme="light")
            expect(theme_select.locator("option:checked")).to_have_text("System (Light)")
            expect(page.locator("html")).to_have_attribute("data-theme", "light")
            checks.append("coherent light/dark/system themes with readable semantic contrast")

            page.get_by_role("link", name="Diagnostics", exact=True).click()
            page.get_by_role("heading", name="Diagnostics", exact=True).wait_for()
            expect(page.get_by_text("Pi verification unknown", exact=True)).to_be_visible()
            source = snapshot(root)
            if any(report["kind"] == "services" for report in source["reports"]):
                expect(page.locator("#service-panel")).to_be_visible()
                expect(page.locator("#service-content")).to_contain_text("Historical file trials")
                checks.append(
                    "DGX3 service history separates discovery from correlated inference trials"
                )
            replays = [r for r in source["reports"] if r["kind"] == "replay"]
            if replays:
                assert page.locator("#replay-select option").count() == len(replays)
                expect(page.get_by_text("Unavailable", exact=True).first).to_be_visible()
                options = page.locator("#replay-select option").all()
                if len(options) > 1:
                    page.select_option("#replay-select", options[-1].get_attribute("value"))
                    expect(page.locator("#audio-content .compact-note").last).to_contain_text(
                        ".json"
                    )
                    page.select_option("#replay-select", options[0].get_attribute("value"))
            assert page.locator("#conversation-content tbody tr").count() > 0
            page.locator("#conversation-content summary", has_text="Stage details").click()
            expect(page.locator("#conversation-content details")).to_contain_text(
                "Tokens/second Unavailable"
            )
            expect(page.locator("#conversation-content details")).to_contain_text(
                "First network byte Unavailable"
            )
            checks.append(
                "stage timings distinguish measured PCM/text boundaries from unavailable token and network-byte metrics"
            )
            page.screenshot(path=str(directory / "diagnostics-desktop.png"), full_page=True)
            page.get_by_role("link", name="Skip to content").focus()
            page.keyboard.press("Enter")
            expect(page.get_by_role("heading", name="Diagnostics", exact=True)).to_be_visible()
            assert page.evaluate("document.activeElement.id === 'main'")
            capture("diagnostics", projected=True)
            checks.append(
                "readable diagnostics at laptop/projector/390/320 and zoom-equivalent layout widths"
            )
            monitored = next(
                (
                    r
                    for r in replays
                    if r["data"].get("state")
                    and not r["data"].get("agent_mode")
                    and any(
                        e.get("event") == "utterance_start"
                        for e in r["data"].get("frontend_events", [])
                    )
                ),
                None,
            )
            if monitored:
                page.select_option("#replay-select", monitored["id"])
                expect(page.get_by_role("heading", name="Diagnostic session")).to_be_visible()
                expect(page.get_by_role("heading", name="CPU inference · p95")).to_be_visible()
                page.get_by_text("Speech boundaries & events", exact=True).click()
                expect(
                    page.locator("#audio-content details").filter(
                        has=page.get_by_text("Speech boundaries & events", exact=True)
                    )
                ).to_contain_text("utterance_start")
                expect(page.locator("#audio-content")).to_contain_text("synthetic")
                page.screenshot(path=str(directory / "wake-vad-desktop.png"), full_page=True)
                checks.append(
                    "wake/VAD diagnostic state, event offsets, CPU timing and synthetic label"
                )

            integrated = next(
                (
                    r
                    for r in replays
                    if r["data"].get("agent_mode") == "mock"
                    and any(
                        e.get("phrase") == "HEY_SWIFT" for e in r["data"].get("wake_events", [])
                    )
                ),
                None,
            )
            if integrated:
                page.select_option("#replay-select", integrated["id"])
                expect(page.locator("#audio-content")).to_contain_text("Hey Swift")
                expect(page.locator("#component-list")).to_contain_text("Mock via our MCP pair")
                assert page.locator("#transcript-state").count() == 0
                capture("integrated-diagnostics")
                expect(page.locator("#scenario-panel")).to_be_visible()
                page.locator("#scenario-panel summary").click()
                expect(page.locator("#scenario-content")).to_contain_text("Passed")
                checks.append(
                    "correlated mixed CPU/mock application evidence and mock scenario outcomes"
                )

            page.locator(".evidence-panel summary").click()
            page.select_option("#evidence-filter", "HPC-live")
            assert page.locator("#evidence-rows .pill").count() > 0
            assert all(
                x == "HPC execution"
                for x in page.locator("#evidence-rows .pill").all_text_contents()
            )
            page.locator("#evidence-rows [data-inspect]").first.click()
            expect(page.locator("#evidence-dialog")).to_be_visible()
            assert "SHA-256" in page.locator("#dialog-source").inner_text()
            capture("evidence-dialog")
            page.get_by_role("button", name="Close report details").click()
            with page.expect_download() as download:
                page.get_by_role("button", name="Export data").click()
            download.value.save_as(str(directory / "snapshot.json"))
            exported = json.loads((directory / "snapshot.json").read_text())
            assert all("history" not in r["data"] for r in exported["reports"])
            page.get_by_role("button", name="Refresh", exact=True).click()
            expect(page.get_by_role("button", name="Refresh", exact=True)).to_be_enabled()
            checks.append("evidence filtering, details, sanitized download and server refresh")

            # A real browser DirectoryHandle/read/write path, using origin-private storage.
            # Only the user-facing folder picker is substituted; no networking fallback.
            page.evaluate(
                """async (snapshot) => {
                const root = await navigator.storage.getDirectory();
                window.testFeed = await root.getDirectoryHandle('me3-browser-test', {create:true});
                const handle = await window.testFeed.getFileHandle('me3-dashboard.json', {create:true});
                const writer = await handle.createWritable();
                for (const report of snapshot.reports) if(report.kind==='replay') report.data.input_rate=32000;
                snapshot.generated_utc=new Date().toISOString();
                await writer.write(JSON.stringify(snapshot)); await writer.close();
                window.showDirectoryPicker=async () => window.testFeed;
            }""",
                source,
            )
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            page.get_by_role("button", name="Connect report feed").click()
            page.get_by_role("button", name="Choose feed folder").click()
            expect(page.get_by_text("Reports · Folder feed", exact=True)).to_be_visible()
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            expect(page.get_by_text("32 kHz", exact=True)).to_be_visible()
            expect(page.locator("#session-summary")).to_contain_text("not connected")
            capture("connected-reports")
            assert page.get_by_text("Pi verification unknown", exact=True).is_visible()
            checks.append("terminal folder adapter reads actual browser FileSystem handles")

            scored_packet = deepcopy(source)
            scored_report = next(
                r for r in scored_packet["reports"] if r["id"] == page.input_value("#replay-select")
            )
            scored_packet["reports"].append(
                {
                    "id": "browser-score-fixture",
                    "source": "var/results/browser-score-fixture.json",
                    "kind": "frontend_score",
                    "evidence": "mock",
                    "sha256": "browser-fixture",
                    "modified_utc": scored_packet["generated_utc"],
                    "data": {
                        "replay_sha256": scored_report["sha256"],
                        "audio_origin": "synthetic",
                        "split": "development",
                        "wake_detected": 3,
                        "wake_expected": 4,
                        "false_activations": 1,
                        "vad_precision": 0.75,
                        "vad_recall": None,
                    },
                }
            )
            page.evaluate(
                """async snapshot => {
                snapshot.generated_utc=new Date().toISOString();
                const h=await window.testFeed.getFileHandle('me3-dashboard.json');
                const w=await h.createWritable(); await w.write(JSON.stringify(snapshot)); await w.close();
            }""",
                scored_packet,
            )
            expect(page.get_by_text("3 / 4", exact=True)).to_be_visible(timeout=8000)
            expect(page.locator("#audio-content")).to_contain_text("synthetic · development")
            checks.append(
                "matching-report annotation-score display using an explicit browser fixture"
            )

            def push_transcript(book):
                packet = deepcopy(source)
                packet["reports"] = [r for r in packet["reports"] if r["kind"] != "transcript"]
                packet["reports"].append(
                    {
                        "id": "browser-transcript",
                        "source": "var/results/ui-transcript.json",
                        "kind": "transcript",
                        "evidence": "mock",
                        "sha256": "browser-fixture",
                        "modified_utc": packet["generated_utc"],
                        "data": book.snapshot(),
                    }
                )
                page.evaluate(
                    """async snapshot => {
                    snapshot.generated_utc=new Date().toISOString();
                    const h=await window.testFeed.getFileHandle('me3-dashboard.json');
                    const w=await h.createWritable(); await w.write(JSON.stringify(snapshot)); await w.close();
                }""",
                    packet,
                )
                page.locator("#history-select option").filter(has_text="Mock ·").first.wait_for(
                    state="attached"
                )
                page.wait_for_function(
                    "id => [...document.getElementById('history-select').options].some(o => o.value === id)",
                    arg=book.session_id,
                )
                page.locator("#history-select").select_option(book.session_id)

            page.get_by_role("link", name="Assistant", exact=True).click()
            transcript = Transcript(mode="mock")
            transcript.start_session("browser-session")
            turn = TurnKey(session_id="browser-session", turn_id=1)
            transcript.begin_turn(turn)
            transcript.set_state("listening")
            transcript.user_text(turn, "Find a", final=False)
            push_transcript(transcript)
            expect(page.get_by_text("Listening…", exact=True)).to_be_visible(timeout=8000)
            expect(page.locator("#voice-status")).to_have_attribute("data-state", "disconnected")
            expect(page.locator("#voice-status")).to_have_attribute("data-mode", "mock")
            expect(page.locator("#voice-status")).to_have_attribute("data-live", "false")
            assert (
                page.locator(".voice-glyph").evaluate("el=>getComputedStyle(el).animationName")
                == "none"
            )
            assert page.locator(".message-user").count() == 1
            transcript.user_text(
                turn, 'Find a paper <img src=x onerror="window.transcriptPwned=true">', final=True
            )
            transcript.set_state("thinking")
            push_transcript(transcript)
            expect(page.locator("#voice-status")).to_have_attribute("data-state", "disconnected")
            transcript.assistant_segment(turn, 1, "Confirmed finding.")
            transcript.confirm_segment(turn, 1)
            transcript.set_state(
                "speaking"
            )  # Explicit simulated sink activity, not generated text.
            transcript.assistant_segment(turn, 2, "Unplayed draft.")
            push_transcript(transcript)
            expect(page.locator(".message-draft")).to_contain_text("Unplayed draft.", timeout=8000)
            expect(page.locator("#voice-status")).to_have_attribute("data-state", "disconnected")
            assert (
                page.locator(".message-user").count() == 1
                and page.locator(".message-assistant").count() == 1
            )
            assert page.locator(".message-user img").count() == 0
            assert page.evaluate("window.transcriptPwned === undefined")
            # Resolve the actual painted background up the DOM, not just root
            # palette tokens: this catches hardcoded white message surfaces.
            message_contrast_js = """() => {
              const rgb=s=>s.match(/[0-9.]+/g).map(Number);
              const lum=arr=>arr.slice(0,3).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0);
              return [...document.querySelectorAll('.message-text,.message-draft,.message-header strong,.message-header .pill,.voice-signature')].map(el=>{
                let node=el,bg;
                while(node){bg=rgb(getComputedStyle(node).backgroundColor);if(bg.length===3||bg[3]>=.99)break;node=node.parentElement;}
                const a=lum(rgb(getComputedStyle(el).color)),b=lum(bg);
                return {selector:el.className,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05)};
              });
            }"""
            for choice in ("light", "dark"):
                theme_select.select_option(choice)
                ratios = page.evaluate(message_contrast_js)
                assert ratios and all(row["ratio"] >= 4.5 for row in ratios), (choice, ratios)
                (directory / f"message-contrast-{choice}.json").write_text(
                    json.dumps(ratios, indent=2) + "\n"
                )
                capture(f"transcript-{choice}")
            page.get_by_role("button", name="Inspect voice session details", exact=True).focus()
            page.keyboard.press("Enter")
            expect(page.get_by_role("dialog", name="Session status", exact=True)).to_be_visible()
            expect(page.locator("#session-details")).to_contain_text("Not connected")
            expect(page.locator("#session-details")).to_contain_text("Simulated session")
            capture("session-details-dark")
            page.keyboard.press("Escape")
            expect(
                page.get_by_role("button", name="Inspect voice session details", exact=True)
            ).to_be_focused()
            page.get_by_role("button", name="Inspect voice session details", exact=True).click()
            page.get_by_role("button", name="Inspect transcript evidence", exact=True).click()
            expect(page.locator("#evidence-dialog")).to_be_visible()
            expect(page.locator("#dialog-content")).to_contain_text("conversation-transcript")
            page.get_by_role("button", name="Close report details").click()
            theme_select.select_option("system")
            checks.append(
                "painted transcript surfaces/text pass contrast in both themes, including drafts"
            )
            checks.append(
                "Swift status graphic opens read-only session details and existing transcript evidence"
            )
            page.screenshot(path=str(directory / "transcript-draft-desktop.png"), full_page=True)
            capture("transcript-draft")
            checks.append(
                "partial ASR replacement, assistant draft/confirmed distinction and escaped text"
            )
            transcript.interrupt(turn)
            push_transcript(transcript)
            expect(
                page.locator("#transcript-messages").get_by_text("Interrupted", exact=True)
            ).to_be_visible(timeout=8000)
            expect(page.locator(".message-assistant .message-text")).to_have_text(
                "Confirmed finding."
            )
            assert page.locator(".message-draft").count() == 0
            expect(page.locator("#voice-status")).to_have_attribute("data-state", "disconnected")
            page.screenshot(
                path=str(directory / "transcript-interrupted-desktop.png"), full_page=True
            )
            capture("transcript-interrupted")
            checks.append(
                "interruption removes unconfirmed assistant tail without duplicating bubbles"
            )
            checks.append("recorded mock status remains still and visibly non-live")
            for number in range(2, 10):
                key = TurnKey(session_id="browser-session", turn_id=number)
                transcript.begin_turn(key)
                transcript.user_text(key, f"Question {number}", final=True)
                transcript.assistant_segment(
                    key, 1, "A confirmed response with several lines.\n" * 5
                )
                transcript.confirm_segment(key, 1)
                transcript.complete(key)
            push_transcript(transcript)
            expect(page.locator(".message-user")).to_have_count(9, timeout=8000)
            page.locator("#transcript-messages").evaluate("el => el.scrollTop=0")
            key = TurnKey(session_id="browser-session", turn_id=10)
            transcript.begin_turn(key)
            transcript.user_text(key, "Follow-up", final=True)
            push_transcript(transcript)
            expect(page.locator(".message-user")).to_have_count(10, timeout=8000)
            assert page.locator("#transcript-messages").evaluate("el => el.scrollTop") == 0
            expect(page.get_by_role("button", name="Jump to latest")).to_be_visible()
            capture("transcript-reading-earlier")
            page.get_by_role("button", name="Jump to latest").click()
            expect(page.get_by_role("button", name="Jump to latest")).to_be_hidden()
            transcript.assistant_segment(key, 1, "New draft while following the bottom.")
            push_transcript(transcript)
            expect(page.get_by_text("New draft while following the bottom.")).to_be_visible(
                timeout=8000
            )
            assert page.locator("#transcript-messages").evaluate(
                "el => el.scrollHeight-el.scrollTop-el.clientHeight < 2"
            )
            checks.append(
                "transcript scroll position retained above bottom; new text follows only at bottom"
            )
            transcript.start_session("browser-next-session")
            push_transcript(transcript)
            expect(page.get_by_role("heading", name="Talk it through.")).to_be_visible(timeout=8000)
            assert page.locator(".message").count() == 0
            checks.append("new session clears the prior conversation projection")
            page.get_by_role("link", name="Diagnostics", exact=True).click()

            page.evaluate(
                """async (snapshot) => {
                snapshot.generated_utc=new Date(Date.now()+1).toISOString();
                for (const report of snapshot.reports) if(report.kind==='replay') report.data.input_rate=44100;
                const h=await window.testFeed.getFileHandle('me3-dashboard.json');
                const w=await h.createWritable(); await w.write(JSON.stringify(snapshot)); await w.close();
            }""",
                source,
            )
            expect(page.get_by_text("44.1 kHz", exact=True)).to_be_visible(timeout=8000)
            page.evaluate("""async () => {
                const h=await window.testFeed.getFileHandle('me3-dashboard.json');
                const w=await h.createWritable(); await w.write('{invalid'); await w.close();
            }""")
            expect(page.locator("#notice")).to_contain_text("Updates unavailable", timeout=8000)
            expect(page.get_by_text("44.1 kHz", exact=True)).to_be_visible()
            checks.append("ongoing file updates and last-good-data preservation on invalid JSON")

            page.evaluate(
                """async (snapshot) => {
                snapshot.generated_utc=new Date(Date.now()+2).toISOString();
                const h=await window.testFeed.getFileHandle('me3-dashboard.json');
                const w=await h.createWritable(); await w.write(JSON.stringify(snapshot)); await w.close();
            }""",
                source,
            )
            expect(page.locator("#notice")).to_be_hidden(timeout=8000)
            expect(page.get_by_text("Reports · Stale", exact=True)).to_be_visible(timeout=18000)
            capture("stale-diagnostics")
            page.get_by_role("link", name="Assistant", exact=True).click()
            capture("stale-assistant")
            checks.append("recovery from partial file and frozen-feed stale detection")
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            page.get_by_role("button", name="Connect report feed").click()
            capture("stale-connection-dialog")
            page.get_by_role("button", name="Disconnect feed").click()
            capture("disconnected-assistant")
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            refresh_button = page.get_by_role("button", name="Refresh", exact=True)
            expect(refresh_button).to_be_disabled()
            disabled_background = refresh_button.evaluate(
                "el=>getComputedStyle(el).backgroundColor"
            )
            refresh_button.hover()
            assert (
                refresh_button.evaluate("el=>getComputedStyle(el).backgroundColor")
                == disabled_background
            )
            checks.append("enabled hover feedback and disabled refresh appearance remain distinct")
            checks.append("explicit feed disconnect; no microphone or task connection implied")

            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_role("link", name="Assistant", exact=True).click()
            if page.locator("#view-assistant").get_attribute("data-conversation") == "true":
                expect(page.locator("#transcript-panel")).to_be_visible()
            else:
                expect(page.get_by_role("heading", name="Talk it through.")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(directory / "assistant-mobile.png"), full_page=True)
            page.get_by_role("link", name="Diagnostics", exact=True).click()
            expect(page.get_by_role("heading", name="Diagnostics", exact=True)).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(directory / "diagnostics-mobile.png"), full_page=True)
            checks.append("390 px mobile layout and both main navigation destinations")

            with tempfile.TemporaryDirectory(
                dir=root / "var/scratch", prefix="ui-browser-"
            ) as temporary:
                fixture = Path(temporary)
                empty = directory / "empty.html"
                empty.write_text(render(snapshot(fixture)))
                page.goto(empty.resolve().as_uri())
                assert page.evaluate("window.isSecureContext")
                assert page.evaluate("typeof window.showDirectoryPicker === 'function'")
                page.get_by_role("link", name="Diagnostics", exact=True).click()
                expect(page.get_by_role("button", name="Refresh", exact=True)).to_be_disabled()
                expect(page.get_by_text("No audio report yet.", exact=True)).to_be_visible()
                capture("empty-diagnostics")
                page.get_by_role("link", name="Assistant", exact=True).click()
                capture("empty-assistant")
                assert page.locator("#voice-status").evaluate(
                    "el=>el.getBoundingClientRect().top >= document.querySelector('.app-header').getBoundingClientRect().bottom"
                )
                expect(page.locator("#session-summary")).to_have_text("Voice not connected")
                page.get_by_role("link", name="Diagnostics", exact=True).click()
                page.get_by_role("button", name="Connect report feed").click()
                capture("snapshot-connection-dialog")
                page.keyboard.press("Escape")
                page.get_by_role("link", name="Diagnostics", exact=True).click()
                checks.append(
                    "offline empty state with unknown capabilities; Chromium folder API on file URL"
                )
                # Portable manual import works even when the browser lacks a directory picker.
                payload = json.dumps(source).encode()
                page.locator("#file-input").set_input_files(
                    {
                        "name": "me3-dashboard.json",
                        "mimeType": "application/json",
                        "buffer": payload,
                    }
                )
                expect(page.get_by_text("Reports · Imported file", exact=True)).to_be_visible()
                expect(page.locator("#audio-content .metrics").first).to_be_visible()
                page.locator("#file-input").set_input_files(
                    {"name": "bad.json", "mimeType": "application/json", "buffer": b"{bad"}
                )
                expect(page.locator("#notice")).to_contain_text("not a supported report")
                expect(page.locator("#audio-content .metrics").first).to_be_visible()
                capture("imported-diagnostics")
                page.get_by_role("link", name="Assistant", exact=True).click()
                capture("imported-assistant")
                page.get_by_role("link", name="Diagnostics", exact=True).click()
                checks.append("manual snapshot import and invalid-import retention")
                results = fixture / "var/results"
                results.mkdir(parents=True)
                (results / "broken.json").write_text("{bad")
                (results / "injection.json").write_text(
                    json.dumps(
                        {
                            "mode": "HPC-live-file-inference",
                            "wake_backend": "</script><script>window.pwned=true</script>",
                        }
                    )
                )
                hostile = directory / "malformed.html"
                hostile.write_text(render(snapshot(fixture)))
                page.goto(hostile.resolve().as_uri())
                page.get_by_role("link", name="Diagnostics", exact=True).click()
                page.locator(".evidence-panel summary").click()
                expect(page.locator("#report-warnings")).to_be_visible()
                assert page.evaluate("window.pwned === undefined")
                expect(page.locator("#audio-content")).to_contain_text(
                    "Voiced fraction unavailable"
                )
                assert page.locator("#vad-fill").count() == 0
                expect(page.locator("#audio-content .metric-value").last).to_have_text(
                    "Unavailable"
                )
                page.emulate_media(reduced_motion="reduce")
                assert (
                    page.locator("summary").first.evaluate(
                        "el => getComputedStyle(el, '::before').transitionDuration"
                    )
                    == "0s"
                )
                checks.append(
                    "missing metrics stay unavailable; reduced-motion preference disables transitions"
                )
                page.locator("#evidence-rows [data-inspect]").first.click()
                assert "</script>" in page.locator("#dialog-content").inner_text()
                checks.append("malformed report isolation and script-string escaping")
            # Explicit presentation fixtures: do not depend on the latest local run.
            service_fixture = deepcopy(source)
            service_fixture["live"] = False
            service_fixture["reports"] = [
                {
                    "id": "trial-fixture",
                    "source": "var/results/browser-fixture.json",
                    "kind": "replay",
                    "evidence": "reported",
                    "modified_utc": "2026-10-28T12:00:00Z",
                    "data": {
                        "wake_events": [],
                        "agent_mode": "external-services",
                        "run_id": "trial-run",
                        "session_ids": ["trial-session"],
                        "sink_chunks": 4,
                        "backends": {"asr": "external", "tts": "external", "llm": "mock"},
                        "backend_contracts": {
                            "asr": "vllm-qwen-audio-v1",
                            "tts": "vllm-qwen-tts-v1",
                        },
                        "turns": [
                            {
                                "run_id": "trial-run",
                                "session_id": "trial-session",
                                "turn_id": 1,
                                "outcome": "completed",
                                "durations_ms": {
                                    "endpoint_to_asr_final_ms": [125],
                                    "tts_first_chunk_ms": [480],
                                },
                            }
                        ],
                    },
                }
            ]
            fixture_path = directory / "service-trial-fixture.html"
            fixture_path.write_text(render(service_fixture))
            page.goto(fixture_path.resolve().as_uri() + "#diagnostics")
            expect(
                page.locator("#service-content").get_by_text("Trial passed", exact=True)
            ).to_have_count(2)
            expect(page.locator("#service-content")).to_contain_text("125 ms")
            page.get_by_role("button", name="Inspect TTS trial").click()
            expect(page.locator("#dialog-content")).to_contain_text("trial-run")
            page.keyboard.press("Escape")
            capture("service-trial-evidence")
            fixture_data = service_fixture["reports"][0]["data"]
            fixture_data["turns"][0]["run_id"] = "unrelated-run"
            fixture_path.write_text(render(service_fixture))
            page.reload()
            expect(
                page.locator("#service-content").get_by_text("Trial passed", exact=True)
            ).to_have_count(0)
            expect(page.locator("#service-content")).not_to_contain_text("125 ms")
            fixture_data["turns"][0]["run_id"] = "trial-run"
            fixture_data["turns"][0]["outcome"] = "error"
            fixture_path.write_text(render(service_fixture))
            page.reload()
            expect(
                page.locator("#service-content").get_by_text("Trial failed", exact=True)
            ).to_have_count(2)
            checks.append(
                "service trials require same run/session, completed outcomes and software timings; failed/unrelated evidence cannot pass"
            )
            assert not errors, errors
            assert not unexpected, unexpected
            browser.close()
            (directory / "checks.json").write_text(
                json.dumps(
                    {"checks": checks, "page_errors": errors, "unexpected_requests": unexpected},
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps({"passed": len(checks), "checks": checks}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
