"""Guided private recordings and bounded replay through existing speech/session contracts."""

import argparse
import asyncio
import hashlib
import json
import math
import re
import secrets
import subprocess
import time
import uuid
import wave
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from aiohttp import web

from .artifacts import output_path

# One server-owned catalog supplies the page, manifests and replay expectations.
CATALOG = [
    (
        "capital",
        "core",
        "questions",
        "What is the capital of the Philippines?",
        "Read exactly once.",
        None,
    ),
    (
        "boiling",
        "core",
        "questions",
        "What is the boiling point of water at sea level?",
        "Read exactly once.",
        None,
    ),
    ("ice", "core", "questions", "Why does ice float on water?", "Read naturally.", None),
    (
        "medical-full",
        "core",
        "questions",
        "What is the first line of treatment for rheumatoid arthritis?",
        "This is a transcription sample; no medical advice is being requested here.",
        None,
    ),
    (
        "medical-initials",
        "core",
        "questions",
        "What is the first line of treatment for R A?",
        "Say the letters R and A.",
        None,
    ),
    (
        "capital-repeat",
        "core",
        "questions",
        "What is the capital of the Philippines?",
        "Repeat with your normal speaking voice.",
        None,
    ),
    (
        "silence",
        "timing",
        "silence",
        "",
        "Stay silent for this entire recording. Do not whisper or play sound.",
        None,
    ),
    (
        "wake-command",
        "timing",
        "wake",
        "Hey Swift, what is the capital of the Philippines?",
        "Say it as one continuous request, with no long gap after Swift.",
        None,
    ),
    (
        "natural-pause",
        "timing",
        "pauses",
        "What is the boiling point of water at sea level?",
        "Pause about one second after “water”, then finish “at sea level”.",
        None,
    ),
    (
        "trailing-thought",
        "timing",
        "pauses",
        "Turn on the lights and set them to orange.",
        "Pause about one second after “and”, then continue.",
        None,
    ),
    (
        "followup-one",
        "timing",
        "followups",
        "What is the capital of Japan?",
        "First turn of a follow-up sequence.",
        None,
    ),
    (
        "followup-two",
        "timing",
        "followups",
        "Can you give me one more sentence about it?",
        "Separate follow-up clip; do not repeat the previous question.",
        None,
    ),
    (
        "followup-three",
        "timing",
        "followups",
        "Make that shorter, please.",
        "Third turn of the same sequence.",
        None,
    ),
    (
        "light-on",
        "actions",
        "lights",
        "Turn on the lights.",
        "Spoken sample only. Capture never executes actions.",
        "light_on",
    ),
    (
        "light-orange",
        "actions",
        "lights",
        "Set the lights to orange.",
        "Same simulated light sequence.",
        "set_color",
    ),
    (
        "light-level",
        "actions",
        "lights",
        "Set brightness to thirty-seven percent.",
        "Read the percentage naturally.",
        "set_brightness",
    ),
    (
        "light-off",
        "actions",
        "lights",
        "Turn off the lights.",
        "Finish the simulated light sequence.",
        "light_off",
    ),
    (
        "music-play",
        "actions",
        "music",
        "Play music by Bruno Mars.",
        "Capture only; no music will start.",
        "music_play",
    ),
    (
        "music-correction",
        "actions",
        "music",
        "No, play Maroon Five instead.",
        "Say Five as a word.",
        "music_play",
    ),
    (
        "music-volume",
        "actions",
        "music",
        "Volume level twenty.",
        "Regression sample for explicit volume routing.",
        "music_volume_set",
    ),
    ("music-pause", "actions", "music", "Pause the music.", "Capture only.", "music_pause"),
    ("music-resume", "actions", "music", "Resume the music.", "Capture only.", "music_resume"),
    ("music-next", "actions", "music", "Next song.", "Capture only.", "music_next"),
    ("music-previous", "actions", "music", "Previous song.", "Capture only.", "music_previous"),
    ("music-stop", "actions", "music", "Stop the music.", "Capture only.", "music_stop"),
    (
        "music-stop-correction",
        "actions",
        "music",
        "I said stop.",
        "Contextual correction after the preceding music requests.",
        "music_stop",
    ),
    (
        "music-fragment",
        "actions",
        "music",
        "Music.",
        "Ambiguous sample: expected clarification, never automatic playlist selection.",
        None,
    ),
    (
        "temperature",
        "tasks",
        "display",
        "Show a temperature of twenty-four degrees.",
        "Displayed demo setpoint, not a sensor or HVAC command.",
        "set_temperature",
    ),
    (
        "timer",
        "tasks",
        "timer",
        "Set a timer for two minutes.",
        "Capture only; replay uses mock devices.",
        "set_timer",
    ),
    (
        "timer-cancel",
        "tasks",
        "timer",
        "Cancel the timer.",
        "Same timer sequence.",
        "cancel_schedule",
    ),
    (
        "alarm",
        "tasks",
        "alarm",
        "Set an alarm for seven tomorrow morning.",
        "Timezone/date must be explicit in a future evaluation.",
        "set_alarm",
    ),
    (
        "time-ph",
        "tasks",
        "clock",
        "What time is it in the Philippines?",
        "Read naturally.",
        "get_time",
    ),
    (
        "time-japan",
        "tasks",
        "clock",
        "No, what time is it in Japan instead?",
        "Correction to the location.",
        "get_time",
    ),
    (
        "weather",
        "tasks",
        "weather",
        "What is the weather in Marilao, Bulacan?",
        "Pronounce your location normally.",
        "get_weather",
    ),
    (
        "reminder",
        "tasks",
        "reminders",
        "Remind me to bring my notebook tomorrow morning.",
        "Dummy reminder; no real notification is sent.",
        "create_reminder",
    ),
    (
        "list-reminders",
        "tasks",
        "reminders",
        "List my reminders.",
        "Same reminder sequence.",
        "list_reminders",
    ),
    (
        "project",
        "agent",
        "project",
        "Create a small website about Philippine tourist spots in a new test folder.",
        "Capture only. General agent execution is separately gated.",
        "agent",
    ),
    (
        "project-progress",
        "agent",
        "project",
        "How is that website going?",
        "Follow-up to the project request.",
        None,
    ),
    (
        "project-correction",
        "agent",
        "project",
        "Use a light theme and add a section about local food.",
        "Correction/addition to the existing task.",
        None,
    ),
    (
        "speech-stop",
        "agent",
        "project",
        "Stop speaking, but keep working on the website.",
        "Stopping speech and cancelling a task are different.",
        None,
    ),
    (
        "task-cancel",
        "agent",
        "project",
        "Cancel the website task.",
        "Capture only; requires an actual cancellable task in future harness tests.",
        None,
    ),
    (
        "silence-repeat",
        "supplemental",
        "silence-repeat",
        "",
        "Stay silent for this entire recording. Repeat the earlier silent sample.",
        None,
    ),
    (
        "music-play-repeat",
        "supplemental",
        "music-regression",
        "Play music by Bruno Mars.",
        "Repeat once at your normal speaking level. Capture only; nothing will play.",
        "music_play",
    ),
    (
        "music-correction-repeat",
        "supplemental",
        "music-regression",
        "No, play Maroon Five instead.",
        "Follow the previous music request; say Five as a word.",
        "music_play",
    ),
    (
        "music-stop-repeat",
        "supplemental",
        "music-regression",
        "I said stop.",
        "Repeat the phrase that was misrecognized. Capture only.",
        "music_stop",
    ),
    (
        "time-ph-repeat",
        "supplemental",
        "clock-regression",
        "What time is it in the Philippines?",
        "First location in a spoken time-correction sequence.",
        "get_time",
    ),
    (
        "time-japan-repeat",
        "supplemental",
        "clock-regression",
        "No, what time is it in Japan instead?",
        "Correct the previous location to Japan.",
        "get_time",
    ),
    (
        "weather-marilao-repeat",
        "supplemental",
        "weather-regression",
        "What is the weather in Marilao, Bulacan?",
        "Repeat the location phrase that was transcribed with a spelling variation.",
        "get_weather",
    ),
    (
        "music-volume-exact",
        "supplemental",
        "music-controls",
        "Set the music volume to fifty percent.",
        "Say the number naturally. Capture only; no player volume changes.",
        "music_volume_set",
    ),
    (
        "music-volume-relative",
        "supplemental",
        "music-controls",
        "Turn the music down a little.",
        "Natural relative-volume phrase; capture only.",
        "music_volume_down",
    ),
    (
        "light-blue-repeat",
        "supplemental",
        "light-variants",
        "Set the lights to blue.",
        "Capture only; no LED is connected or changed.",
        "set_color",
    ),
    (
        "light-brightness-numeric",
        "supplemental",
        "light-variants",
        "Set brightness to 37 percent.",
        "Use the numeric percentage wording; capture only.",
        "set_brightness",
    ),
    (
        "timer-thirty-seconds",
        "supplemental",
        "timer-replacement",
        "Set a timer for thirty seconds.",
        "First timer in a replacement sequence. Capture only.",
        "set_timer",
    ),
    (
        "timer-replace-one-minute",
        "supplemental",
        "timer-replacement",
        "Actually, make it one minute instead.",
        "Replace the timer from the previous clip; capture only.",
        "set_timer",
    ),
]
CASES = {
    row[0]: dict(
        id=row[0],
        suite=row[1],
        scenario=row[2],
        expected_text=row[3],
        instruction=row[4],
        expected_task=row[5],
        seconds=7 if not row[3] else min(18, max(8, len(row[3].split()) // 2 + 4)),
    )
    for row in CATALOG
}
SUITES = {
    "core": "1 · Core transcription",
    "timing": "2 · Wake, pauses & follow-ups",
    "actions": "3 · Lights & music",
    "tasks": "4 · Stateful tasks",
    "agent": "5 · Complex agent requests",
    "supplemental": "6 · ASR misses & control variants",
}
MAX_UPLOAD = 4_000_000
MIMES = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".mp4"}


def local_request(request):
    host = request.headers.get("Host", "")
    if not re.fullmatch(r"(?:localhost|127\.0\.0\.1):[0-9]{1,5}", host):
        raise web.HTTPForbidden()
    if not 1 <= int(host.rsplit(":", 1)[1]) <= 65535:
        raise web.HTTPForbidden()
    if request.headers.get("Origin") not in (None, "http://" + host):
        raise web.HTTPForbidden()


def write_json(path, value):
    temporary = path.with_name("." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, allow_nan=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def add_recording_routes(app, root):
    """A capture utility only: requests cannot invoke models or tools."""
    directory = Path(root) / "var/recordings/guided"
    token = secrets.token_urlsafe(32)
    lock = asyncio.Lock()
    batch_count = 0

    def batch_path(value):
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise web.HTTPNotFound()
        path = directory / value
        if (
            path.is_symlink()
            or not path.is_dir()
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise web.HTTPNotFound()
        return path

    def authorize(request):
        local_request(request)
        if not secrets.compare_digest(request.headers.get("X-Swift-Token", ""), token):
            raise web.HTTPForbidden()

    async def page(request):
        local_request(request)
        nonce = secrets.token_urlsafe(24)
        body = Path(__file__).with_name("ui").joinpath("recording_lab.html").read_text()
        config = json.dumps(
            {"cases": list(CASES.values()), "suites": SUITES, "token": token}
        ).replace("<", "\\u003c")
        body = body.replace("__NONCE__", nonce).replace("__CONFIG__", config)
        return web.Response(
            text=body,
            content_type="text/html",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Permissions-Policy": "microphone=(self)",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; media-src blob:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            },
        )

    async def create(request):
        nonlocal batch_count
        authorize(request)
        raw = await request.content.read(8193)
        if len(raw) > 8192:
            raise web.HTTPRequestEntityTooLarge(max_size=8192, actual_size=len(raw))
        try:
            data = json.loads(raw)
            if (
                set(data) != {"suite", "settings", "consent"}
                or data["suite"] not in SUITES
                or data["consent"] is not True
            ):
                raise ValueError
            settings = data["settings"]
            allowed = {
                "sampleRate",
                "channelCount",
                "echoCancellation",
                "noiseSuppression",
                "autoGainControl",
            }
            if (
                not isinstance(settings, dict)
                or set(settings) - allowed
                or any(
                    type(v) not in (int, float, bool, type(None))
                    or isinstance(v, float)
                    and not math.isfinite(v)
                    for v in settings.values()
                )
            ):
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise web.HTTPBadRequest(text="Invalid recording setup") from None
        async with lock:
            if batch_count >= 64:
                raise web.HTTPTooManyRequests(
                    text="Recording batch limit reached; restart the recorder"
                )
            if directory.is_symlink():
                raise web.HTTPForbidden()
            directory.mkdir(parents=True, exist_ok=True)
            batch_id = uuid.uuid4().hex
            target = directory / batch_id
            target.mkdir(mode=0o700)
            manifest = {
                "schema_version": 1,
                "kind": "swift-guided-recordings",
                "batch_id": batch_id,
                "suite": data["suite"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "capture_settings": settings,
                "consent": {
                    "project_storage": True,
                    "university_asr_replay": True,
                    "other_providers": False,
                },
                "cases": [c for c in CASES.values() if c["suite"] == data["suite"]],
                "takes": [],
            }
            await asyncio.to_thread(write_json, target / "manifest.json", manifest)
            batch_count += 1
        return web.json_response(
            {"batch_id": batch_id, "path": f"var/recordings/guided/{batch_id}", "saved": 0}
        )

    async def upload(request):
        authorize(request)
        target = batch_path(request.match_info["batch"])
        case_id = request.match_info["case"]
        if case_id not in CASES:
            raise web.HTTPBadRequest(text="Unknown recording prompt")
        mime = request.headers.get("Content-Type", "").split(";")[0].strip()
        if mime not in MIMES:
            raise web.HTTPBadRequest(text="Unsupported browser audio format")
        try:
            metadata = request.headers.get("X-Swift-Clip", "")
            if len(metadata) > 6000:
                raise ValueError
            meta = json.loads(unquote(metadata))
            if (
                set(meta) != {"duration_ms", "started_utc", "note", "complete", "settings"}
                or type(meta.get("complete")) is not bool
                or type(meta["duration_ms"]) not in (int, float)
                or not 200 <= meta["duration_ms"] <= 30000
                or not isinstance(meta["note"], str)
                or len(meta["note"]) > 500
            ):
                raise ValueError
            settings = meta["settings"]
            if (
                not isinstance(settings, dict)
                or set(settings)
                - {
                    "sampleRate",
                    "channelCount",
                    "echoCancellation",
                    "noiseSuppression",
                    "autoGainControl",
                }
                or any(
                    type(v) not in (int, float, bool, type(None))
                    or isinstance(v, float)
                    and not math.isfinite(v)
                    for v in settings.values()
                )
            ):
                raise ValueError
            datetime.fromisoformat(meta["started_utc"])
        except (ValueError, TypeError, KeyError):
            raise web.HTTPBadRequest(text="Invalid clip metadata") from None
        data = bytearray()
        async for chunk in request.content.iter_chunked(8192):
            data.extend(chunk)
            if len(data) > MAX_UPLOAD:
                raise web.HTTPRequestEntityTooLarge(max_size=MAX_UPLOAD, actual_size=len(data))
        valid_magic = (
            (mime == "audio/webm" and data[:4] == b"\x1aE\xdf\xa3")
            or (mime == "audio/ogg" and data[:4] == b"OggS")
            or (mime == "audio/mp4" and data[4:8] == b"ftyp")
        )
        if len(data) < 64 or not valid_magic:
            raise web.HTTPBadRequest(text="Invalid or empty audio container")
        async with lock:
            manifest = json.loads((target / "manifest.json").read_text())
            if CASES[case_id]["suite"] != manifest["suite"] or len(manifest["takes"]) >= 150:
                raise web.HTTPBadRequest(
                    text="Prompt does not belong to this batch or take limit reached"
                )
            take_id = uuid.uuid4().hex
            filename = case_id + "-" + take_id + MIMES[mime]
            path = target / filename
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(0o600)
            row = {
                "case_id": case_id,
                "take_id": take_id,
                "file": filename,
                "mime": mime,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                **meta,
            }
            manifest["takes"].append(row)
            await asyncio.to_thread(write_json, target / "manifest.json", manifest)
        return web.json_response(
            {"saved": True, "take_id": take_id, "count": len(manifest["takes"]), "file": filename}
        )

    async def status(request):
        authorize(request)
        target = batch_path(request.match_info["batch"])
        manifest = json.loads((target / "manifest.json").read_text())
        return web.json_response(
            {
                "batch_id": manifest["batch_id"],
                "path": str(target.relative_to(root)),
                "takes": manifest["takes"],
            },
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/recording-lab", page)
    app.router.add_post("/api/recording-lab/batches", create)
    app.router.add_post("/api/recording-lab/batches/{batch}/{case}", upload)
    app.router.add_get("/api/recording-lab/batches/{batch}", status)


def decode_clip(path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(path),
            "-t",
            "31",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        timeout=25,
        check=True,
    )
    if not result.stdout or len(result.stdout) > 960000:
        raise ValueError("Audio must decode to at most 30 seconds")
    return result.stdout


def word_edits(reference, hypothesis):
    def words(s):
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", s.casefold())

    left, right = words(reference), words(hypothesis)
    previous = list(range(len(right) + 1))
    for i, word in enumerate(left, 1):
        current = [i]
        for j, candidate in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != candidate))
            )
        previous = current
    return previous[-1], len(left)


async def evaluate(args):
    from .configuration import BackendSpec, load_backends
    from .contracts import TurnKey
    from .credentials import load_secrets
    from .music import MusicController
    from .providers import MockLLM
    from .remote import make_provider
    from .session import Session
    from .speech import SpeechClient
    from .telemetry import summarize, write_trace
    from .tools import MockTaskExecutor
    from .transcript import Transcript

    root = Path.cwd().resolve()
    batch = args.batch.resolve()
    allowed = root / "var/recordings/guided"
    if (
        args.batch.is_symlink()
        or not batch.is_relative_to(allowed)
        or not (batch / "manifest.json").is_file()
    ):
        raise ValueError("Select an existing guided batch under var/recordings/guided")
    manifest = json.loads((batch / "manifest.json").read_text())
    if manifest.get("consent", {}).get("university_asr_replay") is not True:
        raise ValueError("This batch does not authorize university ASR replay")
    backends = load_backends(args.backend_config)
    if (
        backends["asr"].protocol != "vllm-qwen-audio-v1"
        or urlsplit(backends["asr"].endpoint).hostname != "10.153.60.233"
    ):
        raise ValueError("This recording consent supports the university Qwen ASR endpoint only")
    if not 1 <= args.max_requests <= 150:
        raise ValueError("Choose a bounded request allowance from 1 to 150")
    backends["asr"] = replace(backends["asr"], request_limit=args.max_requests)
    if not args.live_tts:
        backends["tts"] = BackendSpec()
    elif urlsplit(backends["tts"].endpoint).hostname != "10.153.60.233":
        raise ValueError("Live TTS replay is restricted to the university endpoint")
    else:
        backends["tts"] = replace(backends["tts"], request_limit=args.max_requests)
    if args.live_reasoning:
        if urlsplit(backends["llm"].endpoint).hostname != "202.92.159.240":
            raise ValueError("Live reasoning replay is restricted to the university endpoint")
        backends["llm"] = replace(backends["llm"], request_limit=args.max_requests)
    else:
        backends["llm"] = BackendSpec()
    if (args.live_reasoning or args.live_tts) and not args.session:
        raise ValueError("Live downstream checks require --session")
    keys = load_secrets(
        args.secrets_file, {s.auth_env for s in backends.values() if s.backend == "external"}
    )
    if args.llm_secrets_file and args.live_reasoning:
        keys.update(load_secrets(args.llm_secrets_file, {backends["llm"].auth_env}))
    output = output_path(str(args.output))
    if output.exists() or output.with_suffix(".jsonl").exists():
        raise FileExistsError("Choose a new output report; previous evidence is preserved")
    takes = {}
    for row in manifest["takes"]:
        if row["case_id"] not in CASES:
            raise ValueError("Unknown prompt in manifest")
        if row.get("complete", True):
            takes[row["case_id"]] = row  # Latest complete take; preserve originals.
    selected = [
        c
        for c in CASES.values()
        if c["suite"] == manifest["suite"]
        and c["id"] in takes
        and (not args.case or c["id"] in args.case)
    ]
    if not selected or len(selected) > args.max_requests:
        raise ValueError("No selected clips, or request allowance is smaller than the clip count")
    display = Transcript(mode="HPC-live")
    music = MusicController()  # Explicit mock; never load Spotify environment/config.
    tasks = MockTaskExecutor(music=music)
    llm = (
        make_provider("llm", backends["llm"], credentials=keys, legacy_mock_tools=False)
        if args.live_reasoning
        else MockLLM()
    )
    rows = []
    audio = bytearray()

    async def sink(pcm):
        if len(audio) + len(pcm) > 1_920_000:
            raise ValueError("Replay audio exceeds 60-second bounded sink")
        audio.extend(pcm)

    try:
        async with SpeechClient(backends=backends, credentials=keys, timeout=60) as speech:
            async with Session(
                speech,
                llm,
                sink,
                tasks=tasks,
                transcript=display,
                direct_clock_enabled=True,
                turn_timeout=None,
                llm_timeout=backends["llm"].timeout_seconds if args.live_reasoning else 5,
                execution_mode="HPC-live",
            ) as session:
                scenario = None
                for case in selected:
                    take = takes[case["id"]]
                    clip = batch / take["file"]
                    if (
                        clip.is_symlink()
                        or not clip.resolve().is_relative_to(batch)
                        or clip.stat().st_size > MAX_UPLOAD
                        or hashlib.sha256(clip.read_bytes()).hexdigest() != take["sha256"]
                    ):
                        raise ValueError("Recording path/hash validation failed")
                    started = time.monotonic()
                    row = {
                        "case_id": case["id"],
                        "scenario": case["scenario"],
                        "take_id": take["take_id"],
                        "expected": case["expected_text"],
                        "evidence": "recorded laptop input / live university ASR / mock effects",
                        "expected_task": case["expected_task"],
                    }
                    try:
                        pcm = await asyncio.to_thread(decode_clip, clip)
                        row["decode_ms"] = round((time.monotonic() - started) * 1000, 2)
                        audio.clear()
                        request_started = time.monotonic()
                        if args.session:
                            if scenario != case["scenario"]:
                                await session.exit()
                                session.activate()
                                scenario = case["scenario"]
                            await (await session.submit(pcm))
                            snap = display.snapshot()
                            user = next(
                                (
                                    m
                                    for m in snap["messages"]
                                    if m["turn_id"] == session.turn and m["role"] == "user"
                                ),
                                None,
                            )
                            text = user["text"] if user else ""
                            row["tasks"] = [
                                t for t in snap["tasks"] if t["turn_id"] == session.turn
                            ]
                            row["assistant"] = [
                                m
                                for m in snap["messages"]
                                if m["turn_id"] == session.turn and m["role"] == "assistant"
                            ]
                            row["session_id"], row["turn_id"] = session.session_id, session.turn
                            row["turn_errors"] = [
                                {
                                    k: e[k]
                                    for k in ("error_code", "error_role", "error_stage")
                                    if k in e
                                }
                                for e in session.events
                                if e["session_id"] == session.session_id
                                and e["turn_id"] == session.turn
                                and e["event"] == "turn_error"
                            ]
                            if args.live_tts and audio:
                                wav_path = output.with_name(
                                    output.stem + "-" + case["id"] + "-reply.wav"
                                )
                                with wav_path.open("xb") as stream:
                                    with wave.open(stream, "wb") as writer:
                                        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                                        writer.writeframes(audio)
                                row["reply_wav"] = str(wav_path)
                        else:
                            text = await speech.transcribe(
                                TurnKey(session_id=manifest["batch_id"], turn_id=len(rows) + 1), pcm
                            )
                            row["asr_mcp_ms"] = round(
                                (time.monotonic() - request_started) * 1000, 2
                            )
                            row["session_id"], row["turn_id"] = manifest["batch_id"], len(rows) + 1
                        errors, count = word_edits(case["expected_text"], text)
                        row.update(
                            transcript=text,
                            word_edits=errors,
                            reference_words=count,
                            word_error_rate=errors / count if count else None,
                            exact_words=errors == 0,
                            input_audio_ms=round(len(pcm) / 32, 2),
                            elapsed_ms=round((time.monotonic() - started) * 1000, 2),
                            status="completed",
                        )
                        if row.get("turn_errors"):
                            row["status"] = "downstream_error"
                        if case["expected_task"] and args.session:
                            row["routing_check"] = (
                                "matched"
                                if any(t["kind"] == case["expected_task"] for t in row["tasks"])
                                else "not_observed"
                            )
                        if (case["id"] == "music-fragment" or case["suite"] == "core") and row.get(
                            "tasks"
                        ):
                            row["routing_check"] = "unexpected_action"
                    except Exception as exc:
                        row.update(
                            status="error",
                            error_type=type(exc).__name__,
                            error_code=getattr(exc, "code", "replay_error"),
                        )
                    rows.append(row)
                    # Preserve completed checks even if a later provider disconnects.
                    write_json(
                        output,
                        {
                            "kind": "swift-recording-evaluation",
                            "batch_id": manifest["batch_id"],
                            "rows": rows,
                            "complete": False,
                        },
                    )
                if args.session:
                    trace = output.with_suffix(".jsonl")
                    write_trace(trace, session.events)
                    timing = summarize(session.events)
                else:
                    timing = None
    finally:
        await music.close()
        close = getattr(llm, "aclose", None)
        if close:
            await close()
    report = {
        "kind": "swift-recording-evaluation",
        "batch_id": manifest["batch_id"],
        "complete": True,
        "rows": rows,
        "timing": timing,
        "backends": {r: s.backend for r, s in backends.items()},
        "request_limits": {
            r: s.request_limit for r, s in backends.items() if s.backend == "external"
        },
        "effects": "mock-only; no Spotify, GPIO, calls, messages, Onit or arbitrary file tasks",
        "limitations": "Direct ASR MCP input bypasses wake/VAD. Session replay preserves request order, but is not acoustic duplex, live interruption, AEC or Pi verification. Mock reasoning cannot qualify general tool selection. Complex agent requests are captured for later bounded harness tests.",
    }
    write_json(output, report)
    print(
        json.dumps(
            {
                "report": str(output),
                "clips": len(rows),
                "exact_word_matches": sum(r.get("exact_words", False) for r in rows),
                "errors": sum(r["status"] == "error" for r in rows),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    replay = commands.add_parser("evaluate")
    replay.add_argument("batch", type=Path)
    replay.add_argument(
        "--backend-config", type=Path, default=Path("var/cache/voice-backends-asr-check.json")
    )
    replay.add_argument("--secrets-file", type=Path, default=Path("var/cache/voice-backends.env"))
    replay.add_argument("--llm-secrets-file", type=Path)
    replay.add_argument("--max-requests", type=int, required=True)
    replay.add_argument("--case", action="append")
    replay.add_argument("--session", action="store_true")
    replay.add_argument("--live-reasoning", action="store_true")
    replay.add_argument("--live-tts", action="store_true")
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        if not 1024 <= args.port <= 65535:
            parser.error("Choose a loopback port from 1024 to 65535")
        app = web.Application(client_max_size=MAX_UPLOAD)
        add_recording_routes(app, Path.cwd())

        async def home(request):
            local_request(request)
            raise web.HTTPFound("/recording-lab")

        app.router.add_get("/", home)
        print(
            f"Swift recording lab: http://127.0.0.1:{args.port}/recording-lab (no inference during capture)"
        )
        try:
            web.run_app(app, host="127.0.0.1", port=args.port)
        except OSError:
            parser.exit(
                2,
                f"Could not open loopback port {args.port}. If Swift is running, stop it in its own terminal first, or choose another forwarded --port.\n",
            )
    else:
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
