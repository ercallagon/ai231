"""Bounded microphone recovery without falling back to a different input."""
import time


class MicrophoneUnavailable(RuntimeError):
    pass


class MicrophoneRecoveryExhausted(MicrophoneUnavailable):
    pass


class StableMicrophone:
    def __init__(self, sd, selection, *, attempts=5, delay=1., emit=lambda event: None):
        self.sd, self.attempts, self.delay, self.emit = sd, attempts, delay, emit
        selected = sd.query_devices(selection, "input")
        if selected["max_input_channels"] < 1:
            raise MicrophoneUnavailable("Selected device has no microphone input")
        self.name = selected["name"]
        self.hostapi = sd.query_hostapis(selected["hostapi"])["name"]
        self.failures = 0
        self.resolve()

    def inputs(self):
        return [{"index": i, "name": info["name"],
                 "hostapi": self.sd.query_hostapis(info["hostapi"])["name"],
                 "channels": info["max_input_channels"]}
                for i, info in enumerate(self.sd.query_devices())
                if info["max_input_channels"] > 0]

    def resolve(self):
        matches = [entry for entry in self.inputs()
                   if entry["name"] == self.name and entry["hostapi"] == self.hostapi]
        if len(matches) != 1:
            raise MicrophoneUnavailable("Selected microphone is missing or ambiguous; refusing another input")
        return matches[0]["index"]

    def healthy(self):
        self.failures = 0

    def recover(self, error):
        self.failures += 1
        try:
            inputs = self.inputs()
        except Exception:
            inputs = []
        exhausted = self.failures > self.attempts
        wait = min(4., self.delay * 2 ** min(self.failures - 1, 3))
        self.emit({"event": "microphone_recovery", "status": "exhausted" if exhausted else "waiting",
                   "message": "Microphone unavailable; same-input recovery only. Close competing capture apps; inspect USB/ALSA.",
                   "expected_name": self.name, "expected_hostapi": self.hostapi,
                   "attempt": self.failures, "retry_seconds": 0 if exhausted else wait,
                   "error": str(error), "inputs": inputs})
        if exhausted:
            raise MicrophoneRecoveryExhausted(
                "Microphone recovery exhausted. Reconnect the same USB mic and restart; do not change input blindly.") from error
        time.sleep(wait)
