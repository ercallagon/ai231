"""M1 inventory, review-only AEC config, wake/VAD replay, PCM stdin and scoring.

No device stream is opened and no service/configuration is changed by these
commands. An AEC config artifact is not evidence of working echo cancellation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import wave
import sys
import time
import tempfile
import os
from contextlib import nullcontext
from pathlib import Path

from .artifacts import output_path


def query(command):
    if not shutil.which(command[0]):
        return {"available": False}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return {
            "available": True,
            "exit_code": result.returncode,
            "stdout": result.stdout[:16384],
            "stderr": result.stderr[:2048],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": type(exc).__name__}


def inventory():
    files = {}
    for name in (
        "/proc/device-tree/model",
        "/proc/asound/cards",
        "/proc/asound/pcm",
        "/proc/meminfo",
    ):
        path = Path(name)
        if path.is_file():
            value = path.read_text(errors="replace").replace("\0", "").strip()
            if name.endswith("meminfo"):
                value = "\n".join(
                    line
                    for line in value.splitlines()
                    if line.startswith(("MemTotal:", "MemAvailable:"))
                )
            files[name] = value[:4096]
    return {
        "mode": "read-only-inventory",
        "system": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "pi_detected": "Raspberry Pi" in files.get("/proc/device-tree/model", ""),
        "audio_device_directory": Path("/dev/snd").exists(),
        "files": files,
        "commands": {
            " ".join(command): query(command)
            for command in (
                ["arecord", "-l"],
                ["aplay", "-l"],
                ["pw-cli", "--version"],
                ["pw-record", "--version"],
                ["wpctl", "--version"],
                ["wpctl", "status", "-n"],
                ["pactl", "--version"],
            )
        },
        "verdict": "not_qualified",
        "limitations": "No capture/playback, supported-rate probe, ownership or acoustic trial; "
        "this inventory alone cannot establish sufficiency or AEC.",
    }


def aec_config(capture_node: str, playback_node: str, rate: int = 48000):
    for node in (capture_node, playback_node):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,240}", node) or node.startswith("me3.aec."):
            raise ValueError("supply a verified physical node.name, without configuration syntax")
    if capture_node == playback_node or rate not in (16000, 48000):
        raise ValueError("distinct physical nodes and a supported AEC rate are required")
    return f'''# REVIEW ARTIFACT ONLY: not installed, loaded, or Pi-verified.
# Confirm target node identities and native capabilities before authorized use.
# Do not combine blindly with another hardware/browser AEC stage.
context.modules = [
  {{ name = libpipewire-module-echo-cancel
    args = {{
      library.name = aec/libspa-aec-webrtc
      audio.rate = {rate}
      capture.props = {{
        node.name = me3.aec.capture
        target.object = "{capture_node}"
      }}
      source.props = {{ node.name = me3.aec.source }}
      sink.props = {{ node.name = me3.aec.sink }}
      playback.props = {{
        node.name = me3.aec.playback
        target.object = "{playback_node}"
      }}
    }}
  }}
]
'''


def build_wake(args):
    from .wake import OpenWakeWord, SherpaWakeWord, WakeGate

    wake_paths = [args.wake_model, args.mel_model, args.embedding_model]
    if any(wake_paths) and not all(wake_paths):
        raise ValueError("supply all three explicit wake/mel/embedding model paths")
    kws_assets = getattr(args, "kws_assets", None)
    keywords = getattr(args, "keywords_file", None)
    if bool(kws_assets) != bool(keywords):
        raise ValueError("supply both Sherpa assets directory and keyword file")
    if kws_assets and any(wake_paths):
        raise ValueError("select one wake backend per replay")
    wake = (
        WakeGate(OpenWakeWord(*wake_paths), threshold=args.wake_threshold)
        if all(wake_paths)
        else None
    )
    if kws_assets:
        suffix = "epoch-12-avg-2-chunk-16-left-64.int8.onnx"
        wake = WakeGate(
            SherpaWakeWord(
                *(
                    kws_assets / f"{component}-{suffix}"
                    for component in ("encoder", "decoder", "joiner")
                ),
                kws_assets / "tokens.txt",
                keywords,
                threshold=args.kws_threshold,
                boost=args.kws_boost,
            )
        )
    return wake


def replay(args):
    from .audio import SileroVAD
    from .preprocessing import PCMPreprocessor

    if args.wav and args.output.resolve() == args.wav.resolve():
        raise ValueError("report cannot overwrite input evidence")
    if args.processed_wav and args.output.resolve() == args.processed_wav.resolve():
        raise ValueError("report and processed WAV must use distinct paths")
    if args.processed_wav and args.wav and args.wav.resolve() == args.processed_wav.resolve():
        raise ValueError("processed output cannot overwrite input evidence")
    wake = build_wake(args)
    from .frontend import FrontendMonitor

    monitor = FrontendMonitor(wake, SileroVAD(args.vad_threshold))
    live = getattr(args, "command", "replay") == "monitor"
    host = inventory()
    mode = "Pi-local-inference" if host["pi_detected"] else "HPC-live-file-inference"
    source_kind = "pcm-stdin" if live else "wav-file"
    processed = bytearray()
    digest = hashlib.sha256()
    started = time.perf_counter()
    last_publish = started
    status = "running"

    def report():
        elapsed = time.perf_counter() - started
        return {
            "mode": mode,
            "source_kind": source_kind,
            "audio_origin": args.audio_origin,
            "run_status": status,
            "host_architecture": host["architecture"],
            "input_sha256": digest.hexdigest(),
            "hash_scope": "consumed PCM bytes, excluding WAV header",
            "input_rate": rate,
            "input_channels": channels,
            "model_rate": 16000,
            "input_stats": pre.input_stats.report(),
            "output_stats": pre.output_stats.report(),
            **monitor.report(),
            "vad_threshold": args.vad_threshold,
            "wake_backend": wake.detector.backend if wake else "not_configured",
            "wake_provenance": wake.detector.provenance if wake else None,
            "aec": "not_run",
            "noise_suppression": "not_run",
            "elapsed_seconds": elapsed,
            "processing_rtf": elapsed / (pre.input_stats.samples / channels / rate)
            if not live and pre.input_stats.samples
            else None,
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "soxr", "pipecat-ai")
            },
            "limitations": "Input-consumed offsets; no acoustic latency, AEC or accuracy claim. "
            "Diagnostic sessions only: no ASR, tasks or playback. "
            "PCM stdin has no device identity, capture timestamp or overrun detection. "
            "Partial model frames are unscored. Pi execution is not acoustic qualification. "
            "Compute percentiles cover at most the most recent 10000 calls per model.",
        }

    source_context = nullcontext(sys.stdin.buffer) if live else wave.open(str(args.wav), "rb")
    with source_context as source:
        if live:
            rate, channels = args.rate, args.channels
            limit = int(rate * args.max_seconds)
        else:
            rate, channels = source.getframerate(), source.getnchannels()
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("expected uncompressed PCM16 WAV")
            if not 0 < source.getnframes() <= rate * 60:
                raise ValueError(
                    "readiness replay expects 0..60 seconds; split longer trials explicitly"
                )
            limit = source.getnframes()
        pre = PCMPreprocessor(rate, channels)
        consumed, sequence = 0, 0
        if live:
            publish(args.output, report())
        try:
            while consumed < limit:
                frames = min(max(1, rate // 100), limit - consumed)
                pcm = source.read(frames * channels * 2) if live else source.readframes(frames)
                if not pcm:
                    break
                digest.update(pcm)
                normalized = pre.push(pcm, sequence)
                if args.processed_wav:
                    processed.extend(normalized)
                monitor.push(normalized)
                sequence += 1
                consumed += len(pcm) // (channels * 2)
                if live and time.perf_counter() - last_publish >= 1:
                    publish(args.output, report())
                    last_publish = time.perf_counter()
        except KeyboardInterrupt:
            status = "stopped"
        except Exception:
            status = "error"
            monitor.finish()
            if live:
                publish(args.output, report())
            raise
        normalized = pre.finish()
        monitor.push(normalized)
        if args.processed_wav:
            processed.extend(normalized)
        monitor.finish()
        if status == "running":
            status = "limit_reached" if live and consumed >= limit else "eof"
    if args.processed_wav:
        with wave.open(str(args.processed_wav), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", ""))
            output.writeframes(processed)
    return report()


def publish(path, report):
    """Atomic replace of an explicitly owned current report, not evidence archival."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--output", type=output_path, default="var/results/audio-inventory.json")
    scoring = sub.add_parser("score", help="score a completed report against exhaustive labels")
    scoring.add_argument("report", type=Path)
    scoring.add_argument("--labels", type=Path, required=True)
    scoring.add_argument("--output", type=output_path, default="var/results/frontend-score.json")
    aec = sub.add_parser("aec-config")
    aec.add_argument("--capture-node", required=True)
    aec.add_argument("--playback-node", required=True)
    aec.add_argument("--rate", type=int, choices=[16000, 48000], default=48000)
    aec.add_argument("--output", type=output_path, default="var/results/pipewire-echo-cancel.conf")
    rec = sub.add_parser("replay")
    rec.add_argument("wav", type=Path)
    mon = sub.add_parser(
        "monitor", help="bounded raw PCM16 stdin diagnostic; opens no audio device"
    )
    mon.set_defaults(wav=None)
    mon.add_argument(
        "--rate", type=int, choices=[8000, 16000, 24000, 32000, 44100, 48000], required=True
    )
    mon.add_argument("--channels", type=int, choices=[1, 2], default=1)
    mon.add_argument("--max-seconds", type=int, choices=range(1, 601), default=60)
    for target in (rec, mon):
        add_replay_options(target)
    args = parser.parse_args()
    if args.command == "score":
        from .scoring import score

        if args.output.resolve() in (args.report.resolve(), args.labels.resolve()):
            raise ValueError("score output cannot overwrite report or labels")
        publish(args.output, score(args.report, args.labels))
    elif args.command == "aec-config":
        args.output.write_text(aec_config(args.capture_node, args.playback_node, args.rate))
    else:
        result = inventory() if args.command == "inventory" else replay(args)
        publish(args.output, result)
    print(f"{args.command}: {args.output}")


def add_replay_options(rec):
    rec.add_argument("--vad-threshold", type=float, default=0.7)
    rec.add_argument(
        "--audio-origin", choices=["unknown", "synthetic", "recorded"], default="unknown"
    )
    rec.add_argument("--processed-wav", type=output_path)
    rec.add_argument("--wake-model", type=Path)
    rec.add_argument("--mel-model", type=Path)
    rec.add_argument("--embedding-model", type=Path)
    rec.add_argument("--wake-threshold", type=float, default=0.5)
    rec.add_argument(
        "--kws-assets", type=Path, help="explicit English GigaSpeech int8 release directory"
    )
    rec.add_argument("--keywords-file", type=Path, help="tokenized Sherpa keyword file")
    rec.add_argument("--kws-threshold", type=float, default=0.25)
    rec.add_argument("--kws-boost", type=float, default=1.0)
    rec.add_argument("--output", type=output_path, default="var/results/readiness-replay.json")


if __name__ == "__main__":
    main()
