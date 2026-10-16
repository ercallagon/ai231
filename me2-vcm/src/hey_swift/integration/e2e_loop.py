"""Wake -> command VCM -> action loop for Raspberry Pi field testing.

This is intentionally a thin integration harness:
  * wake detection, audio capture, and VCM preprocessing come from canonical
    ``src/hey_swift`` modules;
  * action execution comes from this integration bundle's ``ActionRouter``;
  * a promoted VCM export is supplied explicitly with ``--vcm-onnx``.

The loop returns to idle listening after every accepted/rejected command.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


PATCH_VERSION = "pi-flow-v7-demo-controls-20261016"
LOG_FORMAT = "text"


class MusicCaptureUnavailable(RuntimeError):
    """Known playback could not be quieted; do not classify it as user speech."""


def _emit(event):
    """Human-readable runtime events; JSON remains available for diagnostics."""
    if LOG_FORMAT == "json":
        print(json.dumps(event), flush=True)
        return
    kind = event.get("event", event.get("status", "info"))
    if kind == "command":
        vcm, capture, action = (event.get(k, {}) for k in ("vcm", "capture", "action"))
        print(f"\nCommand guess: {vcm.get('intent')} | confidence {vcm.get('confidence', 0):.1%} | accepted: {vcm.get('accepted')}", flush=True)
        if vcm.get("slots"):
            print(f"  Model slots: {vcm['slots']} | action slot: {event.get('prediction', {}).get('slot_value')}", flush=True)
        print(f"  Capture: {capture.get('reason')} | speech detected: {capture.get('speech_started')} | duration: {capture.get('capture_seconds')} s", flush=True)
        print(f"  Audio: retained {capture.get('audio_seconds')} s | detected speech span {capture.get('detected_speech_seconds')} s | model limit {capture.get('model_window_seconds')} s | model input {capture.get('model_audio_seconds')} s", flush=True)
        if vcm.get("rejection_reason"):
            print(f"  Rejection reason: {vcm['rejection_reason']}", flush=True)
        print(f"  Action: {action.get('status')} — {action.get('message')}", flush=True)
        latency, timing = event.get("latency", {}), action.get("timing", {})
        print(f"  Decision after speech: {latency.get('speech_end_to_decision_seconds')} s (detector-estimated speech end)", flush=True)
        print(f"  Action after speech: {latency.get('speech_end_to_service_seconds')} s | capture setup: {latency.get('capture_setup_seconds')} s", flush=True)
        print(f"  Timing: model {latency.get('vcm_seconds')} s | service {timing.get('service_seconds', '?')} s | response {timing.get('response_seconds', '?')} s | total after speech {latency.get('speech_end_to_action_seconds')} s", flush=True)
        if action.get("speech"):
            print(f"  Speech output: {action['speech'].get('status')}", flush=True)
        diagnostic = action.get("data", {}).get("spotify_diagnostic")
        if diagnostic:
            print(f"  Spotify: {diagnostic.get('message')} ({diagnostic.get('profile')}; phase: {diagnostic.get('phase', 'unknown')})", flush=True)
    elif kind == "action_result":
        print(f"Action {event.get('status')} ({event.get('service_seconds', 0):.3f} s): {event.get('message')}", flush=True)
    elif kind == "wake_score":
        print(f"Listening… wake score {event.get('p_wake', 0):.1%}, peak {event.get('peak_p_wake', 0):.1%}", flush=True)
    elif kind == "wake":
        print(f"Wake detected ({event.get('prob', 0):.1%}).", flush=True)
    elif kind == "ready":
        print(f"Ready [{event.get('runtime_version')}] | mic {event.get('device')} at {event.get('capture_rate')} Hz | live actions: {event.get('live_actions')}", flush=True)
    elif kind == "microphone_recovery":
        print(f"Microphone {event.get('status')}: {event.get('expected_name')} | attempt {event.get('attempt')} | retry {event.get('retry_seconds')} s | {event.get('error')}", flush=True)
    elif kind == "scheduled_action":
        print(f"Scheduled {event.get('intent')}: {event.get('status')} — {event.get('message')} | {event.get('data', {}).get('error', '')}", flush=True)
    elif kind == "cue_levels":
        print(f"Cue: gain {event.get('applied_gain_db')} dB | source RMS {event.get('source_rms_dbfs')} dBFS | peak limited {event.get('peak_limited')} | sink {event.get('sink')}", flush=True)
    else:
        details = event.get("message") or event.get("status") or ""
        volume = f" | volume {event['volume_percent']:g}%" if "volume_percent" in event else ""
        print(f"{kind.capitalize()}: {details}{volume}", flush=True)


INTEGRATION_ROOT = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *INTEGRATION_ROOT.parents):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate.resolve()
    return Path.cwd().resolve()


REPO_ROOT = _repo_root()


def _install_paths() -> None:
    sys.dont_write_bytecode = True
    for path in (REPO_ROOT / "src", REPO_ROOT):
        sp = str(path)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _coerce_device(value):
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


# Exact model-value -> existing action-service value translation. Unknown
# values remain invalid; never approximate/coerce an unsupported slot.
#
# TEMPERATURE and BRIGHTNESS accept BOTH the canonical string values emitted by
# string-labeled VCM exports (run_043 and later: "18", "20", ...) and the legacy
# integer values emitted by run_021-era exports (18, 20, ...). Each key is an
# exact literal match -- there is no numeric conversion, so "18 ", "18.0", 18.0
# and any other unsupported form stay invalid and fail closed. Keeping the
# integer keys preserves run_021 compatibility unchanged.
MODEL_ACTION_SLOTS = {
    "TIMER": {"10s": "10 seconds", "30s": "30 seconds", "1m": "1 minute"},
    "ALARM": {"6AM": "6:00 AM", "8AM": "8:00 AM", "9PM": "9:00 PM"},
    "TEMPERATURE": {
        "18": "18 degrees", "22": "22 degrees", "26": "26 degrees",
        18: "18 degrees", 22: "22 degrees", 26: "26 degrees",
    },
    "BRIGHTNESS": {
        "20": "20 percent", "60": "60 percent", "100": "100 percent",
        20: "20 percent", 60: "60 percent", 100: "100 percent",
    },
    "COLOR": {"red": "Red", "blue": "Blue", "green": "Green"},
    "CREATE_REMINDER": {"drink_water": "Drink water", "study": "Study", "exercise": "Exercise"},
}


def _make_prediction(vcm_result):
    from hey_swift.integration.router import Prediction

    slots = vcm_result.get("slots") or {}
    intent = str(vcm_result.get("intent", "OUT_OF_SCOPE"))
    slot_value = slots.get(intent)
    mapping = MODEL_ACTION_SLOTS.get(intent, {})
    if isinstance(slot_value, (str, int)) and type(slot_value) is not bool:
        slot_value = mapping.get(slot_value, slot_value)
    return Prediction(
        intent=str(vcm_result.get("intent", "OUT_OF_SCOPE")),
        slot_value=slot_value,
        accepted=bool(vcm_result.get("accepted", False)),
    )


def _dbfs(chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    return 20.0 * float(np.log10(rms + 1e-9))


def _capture_command_vad(device, args, *, rate=None, before_ready=None, stream_api=None):
    """Capture one command, stopping soon after end-of-speech.

    Uses an adaptive RMS gate instead of a fixed-duration recording. The first
    Calibrate ambient noise before the cue; reuse the verified sample rate.
    After the cue, restart the same command stream and announce readiness.
    Require sustained energy above the noise margin, preserve onset pre-roll,
    then stop after end silence and trim excessive trailing room audio.
    """
    import sounddevice as sd
    sd = stream_api or sd
    from hey_swift.audio.capture import ContinuousResampler, capture_supported_rate
    from hey_swift.wakeword.features import SAMPLE_RATE
    from hey_swift.audio.endpoint import SpeechFilter
    speech_filter = SpeechFilter(getattr(args, "endpoint_mode", "energy"))

    if rate is None:
        rate, _ = capture_supported_rate(device)
    frame = max(1, int(rate * args.vad_frame_ms / 1000.0))
    resampler = ContinuousResampler(rate, SAMPLE_RATE)
    min_samples = int(SAMPLE_RATE * args.min_command_seconds)
    start_samples = max(1, int(SAMPLE_RATE * args.speech_start_ms / 1000.0))
    pre_roll_samples = max(start_samples + frame * SAMPLE_RATE // rate,
                           int(SAMPLE_RATE * args.pre_roll_ms / 1000.0))
    max_samples = int(SAMPLE_RATE * args.command_seconds) + pre_roll_samples
    end_silence_samples = int(SAMPLE_RATE * args.end_silence_ms / 1000.0)
    start_timeout = float(args.speech_start_timeout)
    noise_seconds = float(args.vad_noise_seconds)

    handoff_started = time.monotonic()
    capture_t0 = handoff_started
    speech_started = False
    speech_start_t = None
    speech_end_t = None
    last_speech_t = None
    silence_samples = 0
    noise_floor_db = None
    chunks = []
    pre_roll = np.zeros(0, dtype=np.float32)
    all_before_speech = []
    noise_levels = []
    peak_input_db = -180.0
    peak_candidate_samples = 0
    candidate_samples = 0
    candidate_start_t = None
    last_speech_sample = 0
    buffered_samples = 0
    speech_deadline = None
    speech_start_sample = None
    overflow = False

    with sd.RawInputStream(samplerate=rate, blocksize=frame, dtype="float32",
                           channels=1, device=device) as stream:
        # Calibrate before the cue/prompt so users can speak as soon as ready.
        calibration_samples = 0
        while calibration_samples < int(rate * noise_seconds):
            data, lost = stream.read(frame)
            overflow = overflow or bool(lost)
            raw = np.frombuffer(data, dtype=np.float32).reshape(-1)
            noise_levels.append(_dbfs(raw))
            calibration_samples += raw.size
        noise_floor_db = float(np.percentile(noise_levels, 30)) if noise_levels else -60.0
        if before_ready is not None:
            # Hold one microphone handle; discard cue audio before restarting.
            stream.abort()
            before_ready()
            stream.start()
        capture_t0 = time.monotonic()
        ready_delay = capture_t0 - handoff_started
        _emit({"status": "capturing", "message": "Say your command now."})
        while True:
            now = time.monotonic()
            elapsed = now - capture_t0
            if not speech_started and elapsed >= start_timeout:
                audio = np.concatenate(all_before_speech) if all_before_speech else np.zeros(0, dtype=np.float32)
                return audio[:max_samples], {
                    "reason": "speech_start_timeout",
                    "_wake_ready_seconds": round(ready_delay, 3),
                    "capture_seconds": round(time.monotonic() - capture_t0, 3),
                    "speech_started": False,
                    "overflow": overflow,
                    "start_sustain_ms": args.speech_start_ms,
                    "noise_floor_db": round(float(noise_floor_db), 2),
                    "noise_calibration_min_db": round(min(noise_levels), 2) if noise_levels else None,
                    "noise_calibration_max_db": round(max(noise_levels), 2) if noise_levels else None,
                    "peak_input_db": round(peak_input_db, 2),
                    "peak_onset_sustain_ms": round(peak_candidate_samples / SAMPLE_RATE * 1000, 1),
                    "threshold_db": round(max(float(args.vad_min_speech_db),
                                               float(noise_floor_db) + float(args.vad_margin_db)), 2),
                }
            if speech_started and now >= speech_deadline:
                break

            data, _overflow = stream.read(frame)
            overflow = overflow or bool(_overflow)
            raw = np.frombuffer(data, dtype=np.float32).reshape(-1)
            chunk = resampler.push(raw)
            if chunk.size == 0:
                continue

            now = time.monotonic()
            elapsed = now - capture_t0
            level_db = _dbfs(chunk)
            peak_input_db = max(peak_input_db, level_db)
            threshold_db = max(float(args.vad_min_speech_db),
                               float(noise_floor_db) + float(args.vad_margin_db))
            voiced = speech_filter.voiced(chunk)
            is_speech = level_db >= threshold_db and voiced

            if not speech_started:
                all_before_speech.append(chunk)
                pre_roll = np.concatenate([pre_roll, chunk])[-pre_roll_samples:]
                if is_speech:
                    if candidate_samples == 0:
                        candidate_start_t = now - chunk.size / SAMPLE_RATE
                    candidate_samples += chunk.size
                    peak_candidate_samples = max(peak_candidate_samples, candidate_samples)
                else:
                    candidate_samples = 0
                    candidate_start_t = None
                if candidate_samples >= start_samples:
                    speech_started = True
                    speech_start_t = candidate_start_t
                    last_speech_t = now
                    speech_deadline = speech_start_t + float(args.command_seconds)
                    # Pre-roll already contains the current frame; do not duplicate.
                    chunks.append(pre_roll.copy())
                    buffered_samples = pre_roll.size
                    speech_start_sample = max(0, buffered_samples - candidate_samples)
                    last_speech_sample = buffered_samples
                continue

            chunks.append(chunk)
            buffered_samples += chunk.size
            if is_speech:
                last_speech_t = now
                silence_samples = 0
                speech_end_t = now
                last_speech_sample = buffered_samples
            else:
                silence_samples += chunk.size

            total = buffered_samples
            if total >= max_samples:
                break
            if total >= min_samples and silence_samples >= end_silence_samples:
                speech_end_t = last_speech_t or now
                break

    wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    # End silence controls waiting, not the waveform passed to the classifier.
    # Keep a short natural tail, removing excess room/music audio from inference.
    if speech_started:
        wav = wav[:last_speech_sample + int(SAMPLE_RATE * 0.12)]
    if wav.size > max_samples:
        wav = wav[:max_samples]
    capture_stop = time.monotonic()
    speech_end_t = speech_end_t or last_speech_t or capture_stop
    return wav, {
        "_wake_ready_seconds": round(ready_delay, 3),
        "reason": "end_silence" if silence_samples >= end_silence_samples else "max_duration",
        "capture_seconds": round(capture_stop - capture_t0, 3),
        "audio_seconds": round(wav.size / SAMPLE_RATE, 3),
        "speech_started": bool(speech_started),
        "overflow": overflow,
        "speech_start_sample": speech_start_sample,
        "speech_end_sample": last_speech_sample,
        "speech_end_estimate": "last voiced frame read time; not acoustic ground truth",
        "endpoint_mode": speech_filter.mode,
        "endpoint_wait_seconds": round(max(0, capture_stop - speech_end_t), 3),
        "speech_start_offset": round((speech_start_t - capture_t0), 3) if speech_start_t else None,
        "speech_wait_seconds": round(speech_start_t - capture_t0, 3) if speech_start_t else None,
        "command_capture_seconds": round(capture_stop - speech_start_t, 3) if speech_start_t else None,
        "speech_end_offset": round((speech_end_t - capture_t0), 3) if speech_end_t else None,
        "noise_floor_db": round(float(noise_floor_db), 2) if noise_floor_db is not None else None,
        "threshold_db": round(max(float(args.vad_min_speech_db),
                                  float(noise_floor_db or args.vad_min_speech_db) + float(args.vad_margin_db)), 2),
        "_speech_end_monotonic": speech_end_t,
    }


def _play_ack(playback, path: Path | None, *, gain_db=0., sink=None) -> None:
    if playback is None:
        _emit({"event": "ack", "status": "disabled"})
        return
    if path is None or not path.exists():
        _emit({"event": "ack", "status": "unavailable", "message": "Wake chime file is missing."})
        return
    try:
        from hey_swift.audio.cue import prepared_cue
        if sys.platform.startswith("linux"):
            # Match the working speech output: PipeWire default sink, then ALSA.
            from hey_swift.audio.speech import play_wav
            with prepared_cue(path, gain_db) as (cue, metadata):
                play_wav(cue, timeout=3, sink=sink)
                _emit({"event": "cue_levels", "sink": sink or "default", **metadata})
        else:
            with prepared_cue(path, gain_db) as (cue, metadata):
                playback.play_ack(str(cue))
        _emit({"event": "ack", "status": "ok"})
    except Exception:
        _emit({"event": "ack", "status": "unavailable",
                          "message": "Wake chime failed; use the capture prompt."})


@contextmanager
def _duck_music(router, duck_percent=0):
    """Mute active music during capture; explicit restore for music actions."""
    service = router.services.get("music")
    backend = getattr(service, "backend", None)
    active = getattr(backend, "active", None)
    setter = None
    original = None
    restored = False
    try:
        if active == "offline":
            player = backend.offline
            original = float(player.command("get_property", "volume", timeout=1))
            setter = lambda value: player.command("set_property", "volume", value, timeout=1)
        elif active == "spotify":
            player = backend.online
            original = float(player.ctl("now", as_json=True, timeout=1)["volume"])
            setter = lambda value: player.ctl("volume", str(int(round(value))), timeout=1)
        if setter is not None:
            setter(original * duck_percent / 100)
            _emit({"event": "music_duck", "backend": active,
                              "volume_percent": original * duck_percent / 100})
    except Exception as exc:
        _emit({"event": "music_duck", "status": "unavailable"})
        raise MusicCaptureUnavailable("Music quieting failed; command capture skipped.") from exc
    def restore():
        nonlocal restored
        if restored:
            return
        restored = True
        if setter is not None and original is not None:
            try:
                setter(original)
                _emit({"event": "music_restore", "backend": active,
                       "volume_percent": original})
            except Exception:
                _emit({"event": "music_restore", "status": "unavailable"})
    try:
        yield restore
    finally:
        restore()


class WakeScoreLog:
    """Summarize verbose wake scores without changing detector cadence."""

    def __init__(self, interval_seconds=2.0):
        self.interval = interval_seconds
        self.last_report = time.monotonic()
        self.peak = 0.0
        self.samples = 0

    def observe(self, probability):
        self.peak = max(self.peak, float(probability))
        self.samples += 1
        now = time.monotonic()
        if now - self.last_report < self.interval:
            return
        _emit({"event": "wake_score", "status": "listening",
                          "p_wake": round(float(probability), 4),
                          "peak_p_wake": round(self.peak, 4),
                          "samples": self.samples})
        self.last_report = now
        self.peak = 0.0
        self.samples = 0


def _listen_for_wake(sd, mic, rate, hop, detector, resampler, focus, args, on_idle=lambda: None):
    from hey_swift.audio.recovery import MicrophoneUnavailable, MicrophoneRecoveryExhausted
    while True:
        recent = np.zeros(0, dtype=np.float32)
        score_log = WakeScoreLog(args.wake_log_interval)
        try:
            device = mic.resolve()
            with sd.RawInputStream(samplerate=rate, blocksize=hop, dtype="float32",
                                   channels=1, device=device) as stream:
                while True:
                    data, overflow = stream.read(hop)
                    mic.healthy()
                    if on_idle():
                        # Controls may block on player IPC; reopen to discard buffered audio.
                        detector.reset(); resampler.reset()
                        break
                    if focus.suppress_input() or overflow:
                        detector.reset(); resampler.reset()
                        continue
                    chunk = resampler.push(np.frombuffer(data, dtype=np.float32).reshape(-1))
                    if getattr(args, "wake_audio_log_dir", None):
                        recent = np.concatenate((recent, chunk))[-32000:]
                    events, probability = detector.process(chunk)
                    if args.verbose:
                        score_log.observe(probability)
                    if events:
                        detected = time.monotonic()
                        _emit({"event": "wake", "prob": round(float(events[0]["prob"]), 4)})
                        if getattr(args, "wake_audio_log_dir", None):
                            _save_debug_audio(args.wake_audio_log_dir, "wake", recent,
                                              {"probability": float(events[0]["prob"]), "microphone": mic.name})
                        return detected
        except MicrophoneRecoveryExhausted:
            raise
        except (sd.PortAudioError, MicrophoneUnavailable) as exc:
            detector.reset(); resampler.reset()
            mic.recover(exc)


def _drain_scheduled_events(router, cursor):
    devices = router.services.get("devices")
    events = getattr(devices, "events", [])
    while cursor < len(events):
        event = events[cursor].to_dict()
        _emit({"event": "scheduled_action", **event})
        cursor += 1
    return cursor


def _save_debug_audio(directory, kind, wav, metadata):
    import uuid
    import soundfile as sf
    root = Path(directory).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        stem = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        sf.write(root / f"{stem}.wav", wav, 16000, subtype="FLOAT")
        (root / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (root / f"{stem}.wav").chmod(0o600)
        (root / f"{stem}.json").chmod(0o600)
    except (OSError, ValueError) as exc:
        _emit({"event": "audio_debug", "status": "unavailable", "message": "Audio debug storage failed; inference is unchanged."})
        return
    _emit({"event": "audio_debug", "message": "Opt-in audio saved; may contain personal speech.", "path": str(root / f"{stem}.wav")})

def run(args) -> int:
    global LOG_FORMAT
    LOG_FORMAT = args.log_format
    _install_paths()

    import sounddevice as sd
    from hey_swift.audio.capture import ContinuousResampler, capture_supported_rate
    from hey_swift.audio.playback import make_playback
    from hey_swift.vcm.inference import VCMInference
    from hey_swift.wakeword.features import SAMPLE_RATE
    from hey_swift.wakeword.inference import WakeWordDetector
    from hey_swift.integration.runtime import build_router, prepare_workspace, _common_speech_phrases
    from hey_swift.audio.focus import AudioFocus
    from hey_swift.audio.endpoint import prepare_command_audio
    from hey_swift.audio.recovery import StableMicrophone, MicrophoneUnavailable, MicrophoneRecoveryExhausted
    from hey_swift.integration.call_guard import wait_for_call_end
    focus = AudioFocus()

    device = _coerce_device(args.device)
    if getattr(args, "pipewire_source", None):
        from hey_swift.audio.pipewire_capture import PipeWireCapture
        sd = PipeWireCapture(args.pipewire_source)
    mic = StableMicrophone(sd, device, attempts=getattr(args, "mic_recovery_attempts", 5),
                           delay=getattr(args, "mic_recovery_delay", 1.), emit=_emit)
    device = mic.resolve()
    wake_config = str(Path(args.wake_config).resolve())
    vcm_config = str(Path(args.vcm_config).resolve())
    hardware_config = str(Path(args.hardware_config).resolve())
    wake_onnx = str(Path(args.wake_onnx).resolve()) if args.wake_onnx else None
    vcm_onnx = str(Path(args.vcm_onnx).resolve())
    labels = str(Path(args.vcm_labels).resolve()) if args.vcm_labels else None
    ack_path = Path(args.ack).resolve() if args.ack else None

    detector = WakeWordDetector(wake_config, onnx_path=wake_onnx)
    if args.threshold is not None:
        detector.gate = type(detector.gate)(
            threshold=float(args.threshold),
            sustain=detector.gate.sustain,
            cooldown_seconds=detector.gate.cooldown_seconds,
            rms_floor_db=detector.gate.rms_floor_db,
            hop_seconds=detector.gate.hop_seconds,
        )
    vcm = VCMInference(config=vcm_config, onnx_path=vcm_onnx, labels_path=labels)
    playback = None if args.no_ack else make_playback(hardware_config)

    prepare_workspace()
    rate, hop = (48000, 4800) if getattr(args, "pipewire_source", None) else capture_supported_rate(device)
    resampler = ContinuousResampler(rate, SAMPLE_RATE)
    _emit({
        "status": "ready",
        "runtime_version": PATCH_VERSION,
        "device": device,
        "microphone_name": mic.name,
        "microphone_hostapi": mic.hostapi,
        "capture_backend": "pipewire" if getattr(args, "pipewire_source", None) else "sounddevice",
        "capture_rate": rate,
        "model_rate": SAMPLE_RATE,
        "wake_threshold": detector.gate.threshold,
        "vcm_onnx": vcm_onnx,
        "live_actions": bool(args.live),
        "music_listen_volume_percent": getattr(args, "music_listen_volume_percent", 50),
        "cue_gain_db": getattr(args, "cue_gain_db", 0.),
        "end_silence_ms": getattr(args, "end_silence_ms", None),
        "speech_start_timeout": getattr(args, "speech_start_timeout", None),
        "speech_start_ms": getattr(args, "speech_start_ms", None),
        "noise_calibration_seconds": getattr(args, "vad_noise_seconds", None),
        "vad_margin_db": getattr(args, "vad_margin_db", None),
    })

    with build_router(live=args.live, hardware=args.hardware, speak=args.speak,
                      music_volume_limit=getattr(args, "music_listen_volume_percent", 50),
                      music_start_volume=getattr(args, "music_start_volume_percent", None),
                      offline=getattr(args, "offline", False),
                      speak_no_speech=getattr(args, "speak_no_speech", False)) as router:
        from hey_swift.integration.terminal import TerminalMusicControls, HELP as TERMINAL_HELP
        controls = TerminalMusicControls(router, emit=_emit)
        benchmark_context = {}
        scheduled_cursor = 0
        def poll_controls():
            # Serialize player changes with TTS duck/restore hooks.
            with focus.lock:
                return controls.poll()
        def deliver_scheduled():
            nonlocal scheduled_cursor
            scheduled_cursor = _drain_scheduled_events(router, scheduled_cursor)
            return poll_controls()
        speech_duck = [None]
        def before_speech():
            focus.before_playback()
            speech_duck[0] = _duck_music(router, args.music_duck_percent)
            speech_duck[0].__enter__()
        def after_speech():
            try:
                if speech_duck[0] is not None:
                    speech_duck[0].__exit__(None, None, None)
                    speech_duck[0] = None
            finally:
                focus.after_playback()
        if router.speech is not None:
            router.speech.set_playback_hooks(before_speech, after_speech, lock=focus.lock)

        def on_action(result, seconds):
            _emit({
                "event": "action_result", "status": result["status"],
                "intent": result["intent"], "message": result["message"],
                "data": result.get("data", {}),
                "service_seconds": round(seconds, 3)})

        router.on_action = on_action
        if router.speech is not None:
            try:
                _emit({"status": "warming_speech", "message": "Loading local voice before listening."})
                router.speech.prepare()
                router.speech.prepare_phrases(_common_speech_phrases(
                    getattr(args, "music_listen_volume_percent", 50)))
                _emit({"status": "speech_ready"})
            except Exception:
                _emit({"status": "speech_unavailable", "message": "Speech initialization failed; actions remain available."})
        music = router.services.get("music")
        if args.live and music is not None and not getattr(args, "offline", False) and os.environ.get("VCM_SPOTIFY_ENABLED", "1") != "0":
            _emit({"event": "warming_music", "message": "Checking this checkout's Spotify daemon; no playback requested."})
            try:
                music.backend.online.prepare()
                _emit({"event": "music_ready", "message": "Spotify control is responding; activation/playback will be checked on Play music."})
            except Exception:
                _emit({"event": "music_unavailable", "message": "Spotify startup check failed; Play music will retry, then use local music only when safe."})
        if sys.stdin.isatty():
            _emit({"event": "terminal_control", "message": TERMINAL_HELP})
        while True:
            scheduled_cursor = _drain_scheduled_events(router, scheduled_cursor)
            _emit({"status": "listening", "message": "Say Hey Swift, then wait for the chime."})
            # Leave the stream context before playback/capture: abort alone
            # keeps the ALSA device open and prevents command capture on Pi.
            wake_detected_at = _listen_for_wake(sd, mic, rate, hop, detector, resampler, focus, args, deliver_scheduled)
            try:
                capture_ready_at = [None]
                with focus.capture(), _duck_music(router, args.music_duck_percent) as restore_music:
                    def prepare_cue():
                        _play_ack(playback, ack_path, gain_db=getattr(args, "cue_gain_db", 0.),
                                  sink=getattr(args, "cue_sink", None))
                        time.sleep(args.playback_tail_seconds)
                        capture_ready_at[0] = time.monotonic()
                    detector.reset()
                    resampler.reset()
                    with focus.capture():
                        wav, capture_meta = _capture_command_vad(
                            mic.resolve(), args, rate=rate, before_ready=prepare_cue, stream_api=sd)
                    _emit({"status": "processing"})
                    vcm_t0 = time.monotonic()
                    processed_samples = 0
                    if capture_meta.get("speech_started", False) and wav.size:
                        model_wav, rejection = prepare_command_audio(wav, capture_meta,
                            getattr(getattr(vcm, "feat_cfg", None), "window_seconds", 2.0),
                            preserve_full=getattr(vcm, "frontend", "legacy_center_crop") != "legacy_center_crop")
                        first, last = capture_meta.get("speech_start_sample"), capture_meta.get("speech_end_sample")
                        capture_meta.update({"model_window_seconds": getattr(getattr(vcm, "feat_cfg", None), "window_seconds", 2.0),
                                             "detected_speech_seconds": (round((last-first)/SAMPLE_RATE, 3)
                                                if first is not None and last is not None else None),
                                             "model_audio_seconds": round(len(model_wav)/SAMPLE_RATE, 3) if model_wav is not None else None})
                        if rejection:
                            vcm_result = {"intent": "OUT_OF_SCOPE", "confidence": 0., "accepted": False,
                                          "slots": {}, "rejection_reason": rejection}
                        else:
                            processed_samples = model_wav.size
                            vcm_result = vcm.predict(model_wav)
                    else:
                        vcm_result = {"intent": "OUT_OF_SCOPE", "confidence": 0.0,
                                      "accepted": False, "oos": True, "slots": {},
                                      "slot_confidence": None,
                                      "rejection_reason": "no_sustained_speech"}
                    vcm_done = time.monotonic()
                    prediction = _make_prediction(vcm_result)
                    decision_at = time.monotonic()
                    speech_end = capture_meta.get("_speech_end_monotonic")
                    speech_end_to_decision = (
                        decision_at - speech_end
                        if capture_meta.get("speech_started") and speech_end is not None
                        else None)
                    if vcm_result.get("rejection_reason") and getattr(args, "rejected_audio_log_dir", None):
                        _save_debug_audio(args.rejected_audio_log_dir, "rejected", wav,
                                          {"capture": capture_meta, "vcm": vcm_result})
                    benchmark_context.clear()
                    benchmark_context.update({
                        "intent": (prediction.intent if prediction.accepted
                                   else "OUT_OF_SCOPE"),
                        "slot": prediction.slot_value if prediction.accepted else None,
                        "infer_ms": round((vcm_done - vcm_t0) * 1000.0, 1),
                        "audio_ms": round(processed_samples / SAMPLE_RATE * 1000.0, 1),
                        "speech_end_to_decision_ms": (
                            round(speech_end_to_decision * 1000.0, 3)
                            if speech_end_to_decision is not None else None),
                        "speech_end_estimate": capture_meta.get("speech_end_estimate"),
                        "decision_latency_scope": "detector_last_voiced_frame_read_to_prediction_ready_before_services",
                    })
                    if args.benchmark_log:
                        print(json.dumps({"event": "benchmark_decision", **benchmark_context,
                            "capture_audio_ms": round(wav.size / SAMPLE_RATE * 1000.0, 1),
                            "accepted": prediction.accepted,
                            "rejection_reason": vcm_result.get("rejection_reason"),
                            "end_silence_ms": args.end_silence_ms,
                            "threads": getattr(vcm, "threads", 1)}), flush=True)
                    # Restore before music effects so volume commands apply to
                    # normal volume; otherwise stay quiet through spoken feedback.
                    if prediction.accepted and prediction.intent in {
                        "PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"}:
                        restore_music()
                    action_result = router.dispatch(prediction, rejection_reason=vcm_result.get("rejection_reason"))
                    action_done = time.monotonic()
                service_finished = action_result.pop("_service_finished_monotonic", action_done)
                speech_end = capture_meta.pop("_speech_end_monotonic", None)
                latency = {
                    "wake_to_capture_ready_seconds": round(capture_ready_at[0] - wake_detected_at, 3),
                    "capture_setup_seconds": capture_meta.pop("_wake_ready_seconds", None),
                    "vcm_seconds": round(vcm_done - vcm_t0, 3),
                    "endpoint_wait_seconds": capture_meta.get("endpoint_wait_seconds"),
                    "speech_end_estimate": capture_meta.get("speech_end_estimate"),
                    "speech_end_to_decision_seconds": (
                        round(speech_end_to_decision, 6)
                        if speech_end_to_decision is not None else None),
                    "action_seconds": round(action_done - vcm_done, 3),
                    "speech_end_to_service_seconds": (
                        round(service_finished - speech_end, 3)
                        if speech_end is not None else None),
                    "speech_end_to_action_seconds": (
                        round(action_done - speech_end, 3) if speech_end is not None else None
                    ),
                }
                _emit({
                    "event": "command",
                    "capture": capture_meta,
                    "latency": latency,
                    "vcm": vcm_result,
                    "prediction": prediction.__dict__,
                    "action": action_result,
                })
                if args.once:
                    return 0
                if prediction.intent == "CALL" and action_result.get("status") in {"dialing", "unknown"}:
                    # Do not capture call/ring/hangup audio or play queued TTS into it.
                    with focus.capture():
                        if router.speech is not None:
                            router.speech.pause_notifications()
                        wait_for_call_end(router.services["calling"], action_result, emit=_emit,
                                          maximum=getattr(args, "call_guard_seconds", 300.),
                                          tail=getattr(args, "call_end_tail_seconds", 3.),
                                          on_idle=poll_controls)
                        if router.speech is not None:
                            router.speech.pause_notifications(False)
                scheduled_cursor = _drain_scheduled_events(router, scheduled_cursor)
                mic.healthy()
            except MicrophoneRecoveryExhausted:
                raise
            except MusicCaptureUnavailable:
                _emit({"event": "command_aborted", "message": "Music could not be quieted; no command capture or action. Check the player before trying again."})
            except (sd.PortAudioError, MicrophoneUnavailable) as exc:
                _emit({"event": "command_aborted", "message": "Microphone interrupted; no command replay or action retry."})
                mic.recover(exc)
            finally:
                detector.reset()
                resampler.reset()
            # A new wake stream is opened next iteration; buffered cue/command
            # audio from the old stream cannot leak into wake detection.


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speak-no-speech", action="store_true", help="Speak empty-capture feedback; default is console only")
    parser.add_argument("--vcm-onnx", required=True, help="Promoted VCM ONNX export.")
    parser.add_argument("--vcm-labels", help="Labels JSON. Defaults to the ONNX sibling labels file.")
    parser.add_argument("--vcm-config", default=str(REPO_ROOT / "configs/vcm.yaml"))
    parser.add_argument("--wake-onnx", default=str(REPO_ROOT / "models/wakeword/wakeword_exp15_mined_lp.onnx"))
    parser.add_argument("--wake-config", default=str(REPO_ROOT / "configs/wakeword.yaml"))
    parser.add_argument("--hardware-config", default=str(REPO_ROOT / "configs/hardware.yaml"))
    parser.add_argument("--ack", default=str(REPO_ROOT / "audio/wake_chime.wav"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--pipewire-source", help="Explicit shared PipeWire USB source node.name; alternative to --device.")
    parser.add_argument("--wake-audio-log-dir", help="Opt-in two-second wake-event recordings for false-wake diagnosis.")
    parser.add_argument("--rejected-audio-log-dir", help="Opt-in full rejected-command audio and duration metadata.")
    parser.add_argument("--mic-recovery-attempts", type=int, default=5)
    parser.add_argument("--mic-recovery-delay", type=float, default=1.)
    parser.add_argument("--call-guard-seconds", type=float, default=300.)
    parser.add_argument("--call-end-tail-seconds", type=float, default=3.,
                        help="Settling delay after verified call end; does not hang up an active call.")
    parser.add_argument("--cue-gain-db", type=float, default=0., help="Cue-only gain; peak-limited. Does not modify the WAV.")
    parser.add_argument("--cue-sink", help="Optional PipeWire target node.name for the cue only.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--command-seconds", type=float, default=5.0)
    parser.add_argument("--min-command-seconds", type=float, default=0.5)
    parser.add_argument("--speech-start-timeout", type=float, default=3.0)
    parser.add_argument("--speech-start-ms", type=float, default=120.0,
                        help="Consecutive above-threshold audio required to start a command (default: 120 ms).")
    parser.add_argument("--end-silence-ms", type=float, default=800.0,
                        help="Silence before endpoint (default: 800 ms; longer pauses may split a command).")
    parser.add_argument("--pre-roll-ms", type=float, default=200.0)
    parser.add_argument("--vad-frame-ms", type=float, default=40.0)
    parser.add_argument("--vad-noise-seconds", type=float, default=0.16)
    parser.add_argument("--vad-margin-db", type=float, default=10.0)
    parser.add_argument("--vad-min-speech-db", type=float, default=-45.0)
    parser.add_argument("--playback-tail-seconds", type=float, default=0.08)
    parser.add_argument("--music-duck-percent", type=float, default=0,
                        help="Percent of previous music volume during capture; default 0 mutes it.")
    parser.add_argument("--music-listen-volume-percent", type=int, default=50,
                        help="Maximum music player volume (default: 50%%; reduce if wake detection suffers).")
    parser.add_argument("--music-start-volume-percent", type=int, default=None,
                        help="Initial music volume (default: 30%%, or the lower configured limit).")
    parser.add_argument("--no-ack", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Skip internet weather and Spotify; use bundled local music.")
    parser.add_argument("--endpoint-mode", choices=["energy", "webrtc"], default="energy",
                        help="Optional speech-aware WebRTC filter; energy preserves tested defaults.")
    parser.add_argument("--log-format", choices=("text", "json"), default="text",
                        help="Readable text by default; json retains structured events.")
    parser.add_argument("--benchmark-log", action="store_true",
                        help="Emit flat JSON decision records for the class live benchmark.")
    parser.add_argument("--verbose", action="store_true", help="Include periodic wake-score summaries.")
    parser.add_argument("--wake-log-interval", type=float, default=2.0,
                        help="Seconds between verbose wake-score summaries (default: 2; 0: every chunk).")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--live", action="store_true", help="Execute real actions.")
    parser.add_argument("--hardware", choices=("mock", "pi"), default="mock")
    parser.add_argument("--speak", action="store_true")
    args = parser.parse_args(argv)
    if args.pipewire_source and args.device is not None:
        parser.error("Choose --pipewire-source OR --device, not both")
    if args.mic_recovery_attempts < 1:
        parser.error("microphone recovery attempts must be positive")
    for name in ("mic_recovery_delay", "call_guard_seconds", "call_end_tail_seconds"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(args.cue_gain_db) or not -12 <= args.cue_gain_db <= 12:
        parser.error("cue gain must be finite and between -12 and 12 dB")
    if not np.isfinite(args.wake_log_interval) or args.wake_log_interval < 0:
        parser.error("--wake-log-interval must be finite and non-negative")
    if not np.isfinite(args.speech_start_ms) or args.speech_start_ms <= 0:
        parser.error("--speech-start-ms must be finite and positive")
    if not np.isfinite(args.music_duck_percent) or not 0 <= args.music_duck_percent <= 100:
        parser.error("--music-duck-percent must be between 0 and 100")
    if not 0 <= args.music_listen_volume_percent <= 100:
        parser.error("--music-listen-volume-percent must be between 0 and 100")
    if args.music_start_volume_percent is not None and not 0 <= args.music_start_volume_percent <= args.music_listen_volume_percent:
        parser.error("--music-start-volume-percent must be between zero and the music limit")
    for name in ("command_seconds", "min_command_seconds", "speech_start_timeout",
                 "end_silence_ms", "pre_roll_ms", "vad_frame_ms"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("vad_noise_seconds", "vad_margin_db", "playback_tail_seconds"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.min_command_seconds > args.command_seconds:
        parser.error("minimum command duration cannot exceed maximum")
    try:
        return run(args)
    except KeyboardInterrupt:
        _emit({"status": "stopped"})
        return 0
    except (RuntimeError, OSError) as exc:
        _emit({"status": "stopped", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
