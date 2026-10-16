"""Serialized offline speech with an injectable backend and playback hooks."""

from collections import OrderedDict
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import threading
import hashlib
import json
import math
import os
import queue
import time
from typing import Protocol
import wave


class SpeechUnavailable(Exception):
    """Speech setup, synthesis, or playback failed."""


class SpeechBackend(Protocol):
    def speak(self, text: str) -> None: ...


class MockSpeech:
    def __init__(self):
        self.messages = []

    def speak(self, text: str) -> None:
        self.messages.append(text)


def _pipewire_command(path, *, volume, sink=None, channel):
    properties = {
        "application.name": f"Hey Swift {channel.title()}",
        "application.id": f"org.hey-swift.{channel}",
        "media.role": "Notification",
        "state.restore-props": False,
        "state.restore-target": False,
    }
    return ["pw-play", "--volume", f"{volume:g}",
            "--properties", json.dumps(properties), "--target", sink or "auto",
            str(path)]


def play_wav(path: Path, *, timeout=60, sink=None) -> None:
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    elif sys.platform.startswith("linux"):
        if shutil.which("pw-play"):
            target = speech_playback_sink(sink if sink is not None else "")
            subprocess.run(_pipewire_command(path, volume=1., sink=target, channel="cue"),
                           check=True, timeout=timeout)
        else:
            if sink:
                raise SpeechUnavailable("A cue output target requires pw-play.")
            subprocess.run(["aplay", "-q", str(path)], check=True, timeout=timeout)
    else:
        raise SpeechUnavailable("No playback adapter configured for this platform.")


def speech_playback_volume(value=None):
    raw = os.environ.get("VCM_SPEECH_CHANNEL_VOLUME", "0.5") if value is None else value
    try:
        volume = float(raw)
    except (TypeError, ValueError) as exc:
        raise SpeechUnavailable("TTS playback volume must be a number from 0 to 1.") from exc
    if not math.isfinite(volume) or not 0 <= volume <= 1:
        raise SpeechUnavailable("TTS playback volume must be a number from 0 to 1.")
    return volume


def speech_playback_sink(value=None):
    sink = os.environ.get("VCM_SPEECH_SINK", "") if value is None else value
    sink = str(sink).strip()
    if any(character in sink for character in ("\x00", "\n", "\r")):
        raise SpeechUnavailable("TTS output target is invalid.")
    return sink or None


def play_speech_wav(path: Path, *, timeout=60, volume=None) -> None:
    """Play speech with per-stream gain; leave cue and device volume untouched."""
    if sys.platform.startswith("linux"):
        gain = speech_playback_volume(volume)
        sink = speech_playback_sink()
        if shutil.which("pw-play"):
            subprocess.run(_pipewire_command(path, volume=gain, sink=sink, channel="speech"),
                           check=True, timeout=timeout)
            return
        if shutil.which("paplay"):
            paplay_volume = round(gain * 65536)
            command = ["paplay", f"--volume={paplay_volume}"]
            if sink:
                command.append(f"--device={sink}")
            command.append(str(path))
            subprocess.run(command,
                           check=True, timeout=timeout)
            return
        if gain != 1 or sink:
            raise SpeechUnavailable("TTS-only volume/routing needs pw-play or paplay; no global audio setting was changed.")
    play_wav(path, timeout=timeout)


class PiperSpeech:
    """Load one pretrained voice lazily and reuse it on CPU."""

    def __init__(self, model_path, *, player=play_speech_wav, phrases_dir=None):
        self.model_path = Path(model_path)
        self._player = player
        self._voice = None
        self._cache = OrderedDict()
        self._cache_limit = 64
        self._cache_dir = None
        self.phrases_dir = Path(phrases_dir) if phrases_dir else None
        self._precomputed = {}
        if self.phrases_dir and (self.phrases_dir / "manifest.json").is_file():
            try:
                manifest = json.loads((self.phrases_dir / "manifest.json").read_text())
                voice_matches = manifest["voice_sha256"] == hashlib.sha256(self.model_path.read_bytes()).hexdigest()
                config_hash = manifest.get("voice_config_sha256")
                config_matches = not config_hash or config_hash == hashlib.sha256(
                    Path(str(self.model_path) + ".json").read_bytes()).hexdigest()
                if voice_matches and config_matches and isinstance(manifest["phrases"], dict):
                    self._precomputed = manifest["phrases"]
            except (OSError, ValueError, KeyError, TypeError):
                # A stale phrase cache must not prevent local actions from starting.
                self._precomputed = {}

    def prepare(self):
        if self._voice is None:
            if not self.model_path.is_file() or not Path(str(self.model_path) + ".json").is_file():
                raise SpeechUnavailable("Piper requires a voice .onnx file and its .onnx.json configuration.")
            from piper import PiperVoice
            self._voice = PiperVoice.load(str(self.model_path), use_cuda=False)
        if self._cache_dir is None:
            self._cache_dir = tempfile.TemporaryDirectory(prefix="swift-speech-")

    def _audio_path(self, text):
        self._last_cache_hit = False
        entry = self._precomputed.get(text)
        if isinstance(entry, dict) and isinstance(entry.get("file"), str):
            path = self.phrases_dir / entry["file"]
            if (path.resolve().parent == self.phrases_dir.resolve() and path.is_file()
                    and hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("sha256")):
                self._last_cache_hit = True
                return path
        path = self._cache.get(text)
        if path is not None and path.is_file():
            self._cache.move_to_end(text)
            self._last_cache_hit = True
            return path
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = Path(self._cache_dir.name) / f"{digest}.wav"
        with wave.open(str(path), "wb") as wav:
            self._voice.synthesize_wav(text, wav)
        self._cache[text] = path
        while len(self._cache) > self._cache_limit:
            _, old_path = self._cache.popitem(last=False)
            old_path.unlink(missing_ok=True)
        return path

    def prepare_phrases(self, phrases):
        """Synthesize finite common responses before the first live command."""
        self.prepare()
        for phrase in dict.fromkeys(p.strip() for p in phrases if p and p.strip()):
            self._audio_path(phrase)

    def speak(self, text: str) -> None:
        try:
            begin = time.monotonic()
            self.prepare()
            path = self._audio_path(text)
            cached = self._last_cache_hit
            prepared = time.monotonic()
            with wave.open(str(path), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if self._player is play_speech_wav:
                self._player(path, timeout=max(5., duration + 5.))
            else:
                self._player(path)
            return {"prepare_seconds": round(prepared - begin, 3),
                    "playback_seconds": round(time.monotonic() - prepared, 3),
                    "audio_seconds": round(duration, 3), "cached": cached}
        except SpeechUnavailable:
            raise
        except Exception as exc:
            raise SpeechUnavailable("Unable to synthesize or play speech.") from exc

    def close(self):
        self._cache.clear()
        if self._cache_dir is not None:
            self._cache_dir.cleanup()
            self._cache_dir = None
        self._voice = None


class SpeechService:
    """One shared instance prevents overlapping responses.

    Calls block until playback ends. The runtime may call from one worker queue.
    before_playback/after_playback hooks can pause detection and clear/resume
    microphone capture; the service does not manipulate the microphone itself.
    """

    def __init__(self, backend: SpeechBackend, *, before_playback=None, after_playback=None):
        self._backend = backend
        self._before = before_playback
        self._after = after_playback
        self._lock = threading.RLock()
        self._notifications = queue.Queue(maxsize=8)
        self._worker = None
        self._worker_lock = threading.Lock()
        self._closed = False
        self.notification_errors = []
        self.notifications_paused = False

    def pause_notifications(self, paused=True):
        with self._lock:
            self.notifications_paused = paused

    def enqueue(self, text):
        """Do not block timer workers on microphone ownership or playback."""
        with self._worker_lock:
            if self._closed:
                raise SpeechUnavailable("Speech service is closed.")
            try:
                self._notifications.put_nowait(text)
            except queue.Full as exc:
                raise SpeechUnavailable("Scheduled speech queue is full.") from exc
            if self._worker is None:
                self._worker = threading.Thread(target=self._deliver_notifications, daemon=True)
                self._worker.start()

    def _deliver_notifications(self):
        while True:
            text = self._notifications.get()
            try:
                if text is None:
                    return
                with self._lock:
                    if not self.notifications_paused:
                        self.speak(text)
            except Exception:
                self.notification_errors.append("Scheduled speech unavailable.")
                self.notification_errors[:] = self.notification_errors[-8:]
            finally:
                self._notifications.task_done()

    def prepare(self):
        with self._lock:
            prepare = getattr(self._backend, "prepare", None)
            if prepare is not None:
                prepare()

    def prepare_phrases(self, phrases):
        with self._lock:
            prepare = getattr(self._backend, "prepare_phrases", None)
            if prepare is not None:
                prepare(phrases)

    def close(self):
        with self._worker_lock:
            self._closed = True
            while True:
                try:
                    self._notifications.get_nowait()
                    self._notifications.task_done()
                except queue.Empty:
                    break
            if self._worker is not None:
                self._notifications.put_nowait(None)
        if self._worker is not None:
            self._worker.join()
        with self._lock:
            close = getattr(self._backend, "close", None)
            if close is not None:
                close()

    def set_playback_hooks(self, before, after, *, lock=None):
        with self._lock:
            self._before, self._after = before, after
            if lock is not None:
                self._lock = lock

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with self._lock:
            try:
                if self._before:
                    self._before()
                return self._backend.speak(text)
            finally:
                if self._after:
                    self._after()
