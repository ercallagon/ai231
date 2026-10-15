"""Explicit shared USB-source capture through pw-record, not exclusive ALSA."""
import json
import queue
import subprocess
import threading


class PipeWireCaptureError(OSError):
    pass


class PipeWireCapture:
    PortAudioError = PipeWireCaptureError

    def __init__(self, source):
        self.source = source

    def query_devices(self, selection=None, kind=None):
        try:
            result = subprocess.run(["pw-dump"], capture_output=True, text=True, check=True, timeout=3)
            nodes = [item.get("info", {}).get("props", {}) for item in json.loads(result.stdout)]
            matches = [node for node in nodes if node.get("node.name") == self.source
                       and node.get("media.class") == "Audio/Source"]
            if len(matches) != 1:
                raise PipeWireCaptureError("Configured PipeWire source missing or ambiguous; refusing default microphone")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise PipeWireCaptureError("Cannot resolve configured USB PipeWire source; inspect pw-dump") from exc
        info = {"name": self.source, "hostapi": 0, "max_input_channels": 1}
        return info if kind else [info]

    def query_hostapis(self, index):
        return {"name": "PipeWire explicit source"}

    def RawInputStream(self, **kwargs):
        self.query_devices()
        return PipeWireInputStream(self.source, **kwargs)


class PipeWireInputStream:
    def __init__(self, source, *, samplerate, blocksize, dtype, channels, device=None):
        if dtype != "float32" or channels != 1:
            raise ValueError("PipeWire capture supports mono float32 only")
        self.source, self.rate, self.blocksize = source, samplerate, blocksize
        self._queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._paused = False
        self._overflow = False
        self._error = None
        self.process = None
        self.thread = None

    def __enter__(self):
        try:
            self.process = subprocess.Popen(["pw-record", "--target", self.source,
                "--rate", str(self.rate), "--channels", "1", "--channel-map", "MONO", "--format", "f32", "--raw",
                "--properties", json.dumps({"node.name": "hey-swift.capture", "node.dont-fallback": True,
                                            "node.dont-reconnect": True}), "-"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            self.thread = threading.Thread(target=self._pump, daemon=True)
            self.thread.start()
            return self
        except OSError as exc:
            self.__exit__(None, None, None)
            raise PipeWireCaptureError("Cannot start pw-record; check pipewire-bin and user audio session") from exc

    def _pump(self):
        try:
            needed = self.blocksize * 4
            while not self._stop.is_set():
                block = bytearray()
                while len(block) < needed and not self._stop.is_set():
                    data = self.process.stdout.read(needed - len(block))
                    if not data:
                        raise PipeWireCaptureError("pw-record ended; configured USB source may be unavailable")
                    block.extend(data)
                with self._lock:
                    if self._paused or self._stop.is_set():
                        continue
                    if self._queue.full():
                        self._queue.get_nowait()
                        self._overflow = True
                    self._queue.put_nowait(bytes(block))
        except Exception as exc:
            self._error = exc

    def read(self, frames):
        if frames != self.blocksize:
            raise ValueError("Read size must match stream blocksize")
        try:
            data = self._queue.get(timeout=3)
        except queue.Empty as exc:
            raise PipeWireCaptureError("PipeWire input stalled; inspect USB source and audio graph") from (self._error or exc)
        with self._lock:
            overflow = self._overflow
            self._overflow = False
        return data, overflow

    def _discard(self):
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def abort(self):
        # Keep draining the shared source during the cue; never buffer its audio.
        with self._lock:
            self._paused = True
            self._discard()
            self._overflow = False

    def start(self):
        with self._lock:
            self._discard()
            self._paused = False

    def __exit__(self, *exc):
        self._stop.set()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
            if self.thread is not None:
                self.thread.join(timeout=1)
            self.process.stdout.close()
