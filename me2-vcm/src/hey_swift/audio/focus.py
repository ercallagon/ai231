"""Half-duplex ownership shared by command capture and scheduled speech."""
from contextlib import contextmanager
import threading
import time


class AudioFocus:
    def __init__(self, tail_seconds=.16):
        self._lock = threading.RLock()
        self._playing = threading.Event()
        self._resume_at = 0.
        self.tail_seconds = tail_seconds

    @property
    def lock(self):
        return self._lock

    def before_playback(self):
        self._lock.acquire()
        self._playing.set()

    def after_playback(self):
        self._resume_at = time.monotonic() + self.tail_seconds
        self._playing.clear()
        self._lock.release()

    def suppress_input(self):
        return self._playing.is_set() or time.monotonic() < self._resume_at

    @contextmanager
    def capture(self):
        with self._lock:
            remaining = self._resume_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            yield
