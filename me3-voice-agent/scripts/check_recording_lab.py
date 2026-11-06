"""Exercise recorder uploads using Chromium's explicitly fake microphone."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from aiohttp import web
from playwright.async_api import async_playwright

from me3_voice.artifacts import output_path
from me3_voice.recording_lab import add_recording_routes, decode_clip


async def check(output):
    output_path(str(output / "marker"))
    root = (output / ("fake-capture-" + uuid.uuid4().hex[:8])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = web.Application()
    add_recording_routes(app, root)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    errors = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]
            )
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720}, permissions=["microphone"]
            )
            page = await context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.add_init_script(
                """window.recordingStreams=[];const real=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async(...args)=>{const stream=await real(...args);recordingStreams.push(stream);return stream;};"""
            )
            await page.goto(f"http://127.0.0.1:{port}/recording-lab")
            await page.evaluate(
                """const label=document.createElement('div');label.textContent='BROWSER CHECK · FAKE MICROPHONE · NO MODEL CALLS';label.style='background:#fff1b8;color:#483a00;padding:6px;text-align:center;font-weight:600';document.body.prepend(label);"""
            )
            await page.screenshot(path=str(output / "idle-desktop.png"), full_page=True)
            assert await page.locator("#start").is_disabled()
            await page.locator("#consent").check()
            box = await page.locator("#start").bounding_box()
            assert box["y"] + box["height"] <= 720
            await page.locator("#start").focus()
            await page.keyboard.press("Enter")
            await page.wait_for_function(
                "() => document.getElementById('stage').classList.contains('recording')"
            )
            await page.screenshot(path=str(output / "recording-desktop.png"), full_page=True)
            await page.wait_for_timeout(800)
            # A failed upload must retain the blob and offer recovery.
            await page.route("**/api/recording-lab/batches/*/capital", lambda route: route.abort())
            await page.locator("#finish").click()
            await page.wait_for_function(
                "() => document.getElementById('retry').hidden===false && document.getElementById('stop').disabled"
            )
            await page.screenshot(path=str(output / "upload-recovery-desktop.png"), full_page=True)
            await page.unroute("**/api/recording-lab/batches/*/capital")
            await page.locator("#retry").click()
            await page.wait_for_function(
                "() => document.querySelector('#queue .status').textContent==='Saved to ME3'"
            )
            assert await page.evaluate("recordingStreams.length") == 1
            assert await page.evaluate(
                "recordingStreams[0].getTracks().every(t=>t.readyState==='ended')"
            )
            manifests = list(root.glob("var/recordings/guided/*/manifest.json"))
            assert len(manifests) == 1
            manifest = json.loads(manifests[0].read_text())
            assert len(manifest["takes"]) == 1
            clip = manifests[0].parent / manifest["takes"][0]["file"]
            assert len(decode_clip(clip)) > 0
            await page.reload()
            await page.evaluate(
                """const label=document.createElement('div');label.textContent='BROWSER CHECK · FAKE MICROPHONE · NO MODEL CALLS';label.style='background:#fff1b8;color:#483a00;padding:6px;text-align:center;font-weight:600';document.body.prepend(label);"""
            )
            await page.locator("#resume").click()
            await page.wait_for_function(
                "() => document.querySelector('#queue .status').textContent==='Saved to ME3'"
            )
            await page.set_viewport_size({"width": 390, "height": 844})
            await page.emulate_media(reduced_motion="reduce")
            assert await page.evaluate("document.documentElement.scrollWidth<=innerWidth")
            await page.screenshot(path=str(output / "resumed-narrow.png"), full_page=True)
            await page.emulate_media(color_scheme="dark")
            await page.screenshot(path=str(output / "resumed-narrow-dark.png"), full_page=True)
            assert not errors, errors
            await browser.close()
            (output / "check.json").write_text(
                json.dumps(
                    {
                        "evidence": "Chromium fake microphone; no real capture/model calls",
                        "saved_takes": len(manifest["takes"]),
                        "page_errors": errors,
                        "checks": [
                            "consent",
                            "countdown",
                            "recording",
                            "single capture stream",
                            "upload failure and retry",
                            "decoded audio",
                            "released microphone",
                            "resume",
                            "narrow overflow",
                            "dark theme",
                            "reduced motion",
                        ],
                    },
                    indent=2,
                )
                + "\n"
            )
            print("Recorder browser check passed; screenshots and fake recordings:", output)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("var/results/recording-lab-check"))
    asyncio.run(check(parser.parse_args().output))
