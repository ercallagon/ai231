"""Private capture contracts and replay checks; no real microphone or provider."""

import json
import re
from argparse import Namespace
from pathlib import Path

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from me3_voice import recording_lab as lab


def metadata(**values):
    return json.dumps(
        {
            "duration_ms": 1000,
            "started_utc": "2026-11-04T12:00:00+00:00",
            "note": "",
            "complete": True,
            "settings": {},
            **values,
        }
    )


async def new_batch(client, server, token, suite="core"):
    response = await client.post(
        server.make_url("/api/recording-lab/batches"),
        headers={"X-Swift-Token": token},
        json={"suite": suite, "settings": {"echoCancellation": True}, "consent": True},
    )
    assert response.status == 200
    return await response.json()


async def test_guided_upload_origin_tokens_retake_and_manifest(tmp_path):
    app = web.Application()
    lab.add_recording_routes(app, tmp_path)
    async with TestServer(app, host="127.0.0.1") as server, ClientSession() as client:
        page = await client.get(server.make_url("/recording-lab"))
        html = await page.text()
        assert "connect-src 'self'" in page.headers["Content-Security-Policy"]
        token = re.search(r'"token": "([\w-]+)"', html).group(1)
        denied = await client.post(server.make_url("/api/recording-lab/batches"), json={})
        assert denied.status == 403
        denied = await client.post(
            server.make_url("/api/recording-lab/batches"),
            headers={"X-Swift-Token": token, "Origin": "http://evil.invalid"},
            json={},
        )
        assert denied.status == 403
        batch = await new_batch(client, server, token)
        upload = server.make_url(f"/api/recording-lab/batches/{batch['batch_id']}/capital")
        headers = {"X-Swift-Token": token, "Content-Type": "audio/webm", "X-Swift-Clip": metadata()}
        bad = await client.post(upload, headers=headers, data=b"not an audio container" * 5)
        assert bad.status == 400
        audio = b"\x1aE\xdf\xa3" + b"\0" * 100
        for _ in range(2):
            saved = await client.post(upload, headers=headers, data=audio)
            assert saved.status == 200
        wrong = await client.post(
            server.make_url(f"/api/recording-lab/batches/{batch['batch_id']}/music-stop"),
            headers=headers,
            data=audio,
        )
        assert wrong.status == 400
        status = await client.get(
            server.make_url(f"/api/recording-lab/batches/{batch['batch_id']}"),
            headers={"X-Swift-Token": token},
        )
        value = await status.json()
        assert len(value["takes"]) == 2
        assert value["takes"][0]["file"] != value["takes"][1]["file"]
        root = tmp_path / value["path"]
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["cases"][0]["expected_text"] == lab.CASES["capital"]["expected_text"]
        assert manifest["consent"]["other_providers"] is False
        assert all((root / t["file"]).read_bytes() == audio for t in manifest["takes"])
        assert all((root / t["file"]).stat().st_mode & 0o077 == 0 for t in manifest["takes"])


def test_supplemental_suite_targets_observed_asr_errors_and_contextual_routes():
    cases = [case for case in lab.CASES.values() if case["suite"] == "supplemental"]
    assert lab.SUITES["supplemental"] == "6 · ASR misses & control variants"
    assert len(cases) == 13
    assert [case["id"] for case in cases] == [
        "silence-repeat",
        "music-play-repeat",
        "music-correction-repeat",
        "music-stop-repeat",
        "time-ph-repeat",
        "time-japan-repeat",
        "weather-marilao-repeat",
        "music-volume-exact",
        "music-volume-relative",
        "light-blue-repeat",
        "light-brightness-numeric",
        "timer-thirty-seconds",
        "timer-replace-one-minute",
    ]
    assert cases[0]["expected_text"] == ""
    assert cases[2]["scenario"] == cases[1]["scenario"]
    assert cases[3]["scenario"] == cases[1]["scenario"]
    assert cases[5]["scenario"] == cases[4]["scenario"]
    assert cases[-1]["scenario"] == cases[-2]["scenario"]
    assert cases[11]["expected_task"] == cases[12]["expected_task"] == "set_timer"


@pytest.mark.parametrize(
    "session_mode,case_id,recognized,exact",
    [
        (True, "light-on", "Turn on the lights.", True),
        (False, "capital", "The capital of the Philippines is Manila.", False),
    ],
)
async def test_replay_uses_actual_session_without_real_effects_or_duplicate_asr(
    tmp_path, monkeypatch, session_mode, case_id, recognized, exact
):
    from me3_voice import configuration, credentials, speech
    from me3_voice.dgx3 import profile

    monkeypatch.chdir(tmp_path)
    batch = tmp_path / "var/recordings/guided" / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    batch.mkdir(parents=True)
    clip = batch / "fixture.webm"
    clip.write_bytes(b"fixture")
    import hashlib

    case = dict(lab.CASES[case_id])
    case["expected_text"] = "Client-supplied reference must not override catalog"
    manifest = {
        "schema_version": 1,
        "batch_id": "a" * 32,
        "suite": case["suite"],
        "consent": {"university_asr_replay": True},
        "cases": [case],
        "takes": [
            {
                "case_id": case_id,
                "take_id": "take1",
                "file": clip.name,
                "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                "complete": True,
            }
        ],
    }
    (batch / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    class Speech:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def transcribe(self, key, pcm):
            calls.append(pcm)
            return recognized

        async def synthesize(self, key, text):
            yield b"\1\0" * 160

    monkeypatch.setattr(speech, "SpeechClient", Speech)
    monkeypatch.setattr(
        configuration, "load_backends", lambda _: profile(enable=True, request_limit=1)
    )
    monkeypatch.setattr(credentials, "load_secrets", lambda *_: {})
    monkeypatch.setattr(lab, "decode_clip", lambda _: b"\1\0" * 160)
    args = Namespace(
        batch=batch,
        backend_config=Path("unused"),
        secrets_file=Path("unused"),
        llm_secrets_file=None,
        max_requests=1,
        case=[case_id],
        session=session_mode,
        live_reasoning=False,
        live_tts=False,
        output=Path("var/results/replay.json"),
    )
    await lab.evaluate(args)
    report = json.loads(args.output.read_text())
    assert len(calls) == 1
    assert report["rows"][0]["exact_words"] is exact
    assert report["rows"][0]["expected"] == lab.CASES[case_id]["expected_text"]
    if session_mode:
        assert report["rows"][0]["routing_check"] == "matched"
        assert report["rows"][0]["tasks"][0]["status"] == "simulated"
        assert args.output.with_suffix(".jsonl").is_file()
    else:
        assert report["rows"][0]["word_edits"] > 0
        assert "tasks" not in report["rows"][0]
    with pytest.raises(FileExistsError):
        await lab.evaluate(args)
    assert len(calls) == 1


async def test_upload_bound_and_stopped_take_are_preserved_without_inference(tmp_path, monkeypatch):
    app = web.Application()
    lab.add_recording_routes(app, tmp_path)
    async with TestServer(app, host="127.0.0.1") as server, ClientSession() as client:
        page = await client.get(server.make_url("/recording-lab"))
        token = re.search(r'"token": "([\w-]+)"', await page.text()).group(1)
        batch = await new_batch(client, server, token)
        url = server.make_url(f"/api/recording-lab/batches/{batch['batch_id']}/capital")
        headers = {
            "X-Swift-Token": token,
            "Content-Type": "audio/webm",
            "X-Swift-Clip": metadata(complete=False, note="Stopped early"),
        }
        monkeypatch.setattr(lab, "MAX_UPLOAD", 128)
        bad = await client.post(url, headers=headers, data=b"\x1aE\xdf\xa3" + b"\0" * 200)
        assert bad.status == 413
        saved = await client.post(url, headers=headers, data=b"\x1aE\xdf\xa3" + b"\0" * 100)
        assert saved.status == 200
        manifest = json.loads((tmp_path / batch["path"] / "manifest.json").read_text())
        assert len(manifest["takes"]) == 1
        assert manifest["takes"][0]["complete"] is False
