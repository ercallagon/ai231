"""Optional WebRTC speech filter; energy-only behavior remains the default."""
import numpy as np


class SpeechFilter:
    def __init__(self, mode="energy"):
        self.mode = mode
        self._vad = None
        self._pending = np.zeros(0, dtype=np.float32)
        if mode == "webrtc":
            try:
                import webrtcvad
            except ImportError as exc:
                raise RuntimeError("Install optional webrtcvad-wheels to use --endpoint-mode webrtc") from exc
            self._vad = webrtcvad.Vad(2)
        elif mode != "energy":
            raise ValueError("Unknown endpoint mode")

    def voiced(self, chunk):
        if self._vad is None:
            return True
        self._pending = np.concatenate((self._pending, chunk))
        votes = []
        while self._pending.size >= 160:
            frame = self._pending[:160]
            self._pending = self._pending[160:]
            pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
            votes.append(self._vad.is_speech(pcm, 16000))
        return bool(votes) and sum(votes) * 2 >= len(votes)


def prepare_command_audio(wav, capture, window_seconds, rate=16000, preserve_full=False):
    """Fit known speech bounds without discarding words; reject incomplete audio."""
    wav = np.asarray(wav, dtype=np.float32)
    limit = int(window_seconds * rate)
    if capture.get("overflow"):
        return None, "capture_overflow"
    if capture.get("reason") == "max_duration":
        return None, "command_incomplete"
    if wav.size <= limit:
        return wav, None
    if preserve_full:
        return None, "command_too_long"
    first, last = capture.get("speech_start_sample"), capture.get("speech_end_sample")
    if first is None or last is None or not 0 <= first < last <= len(wav) or last - first > limit:
        return None, "command_too_long"
    lower, upper = max(0, last - limit), min(first, len(wav) - limit)
    start = max(lower, min(upper, first - (limit - (last - first)) // 2))
    return wav[start:start + limit], None
