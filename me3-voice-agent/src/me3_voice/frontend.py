"""Wake/VAD diagnostic controller, independent of ASR, tools and playback.

Times are consumed model-input offsets. No acoustic clock, AEC or microphone
driver is implied. The bounded report contains ranges, never audio or text.
"""

from collections import deque
from time import perf_counter

from .preprocessing import PCMFrames


class FrontendMonitor:
    def __init__(
        self, wake, vad, *, silence_frames=20, idle_frames=313, max_frames=937, session_policy=True
    ):
        if min(silence_frames, idle_frames, max_frames) < 1:
            raise ValueError("positive diagnostic policy bounds required")
        self.wake, self.vad = wake, vad
        self.session_policy = session_policy
        self.silence_frames, self.idle_frames, self.max_frames = (
            silence_frames,
            idle_frames,
            max_frames,
        )
        self.frames = PCMFrames(512)
        self.wake_frames = PCMFrames(wake.frame_samples if wake else 1280)
        self.vad_count = self.voiced_count = self.wake_count = 0
        self.wake_peak = None
        self.events = deque(maxlen=1000)
        self.segments = deque(maxlen=1000)
        self.event_count = self.segment_count = self.activations = 0
        self.state = "wake_listening" if wake else "vad_only"
        self.latest_voiced = False
        self.last_voice = self.last_activity = 0
        self.segment_start = self.utterance_start = None
        self.closed = False
        self.tail = self.wake_tail = 0
        self.timing = {name: deque(maxlen=10000) for name in ("wake", "vad")}

    def event(self, name, samples, **fields):
        self.event_count += 1
        self.events.append({"event": name, "audio_seconds": samples / 16000, **fields})

    def _onset(self, end):
        # Range includes up to 320 ms available pre-roll; no audio is exported.
        self.utterance_start = max(0, end - 512 - 5120)
        self.state = "speech"
        self.event("utterance_start", end, pre_roll_start_seconds=self.utterance_start / 16000)

    def push(self, pcm):
        if self.closed:
            raise ValueError("monitor already finished")
        if len(pcm) % 2 or len(pcm) > 32000:
            raise ValueError("complete PCM16 samples, at most one second required")
        # At most 4 ms between policy updates, below either model frame size.
        # This keeps rearming identical for tiny callbacks and large file reads.
        signals = []
        for offset in range(0, len(pcm), 128):
            signals.extend(self._push(pcm[offset : offset + 128]))
        return signals

    def _push(self, pcm):
        pending = []
        signals = []
        for frame in self.frames.push(pcm):
            started = perf_counter()
            voiced = bool(self.vad.voiced(frame))
            self.timing["vad"].append((perf_counter() - started) * 1000)
            self.vad_count += 1
            self.voiced_count += int(voiced)
            pending.append((self.vad_count * 512, "vad", voiced))
            signals.append(
                {"kind": "vad", "samples": self.vad_count * 512, "voiced": voiced, "pcm": frame}
            )
        if self.wake:
            for frame in self.wake_frames.push(pcm):
                started = perf_counter()
                event = self.wake.feed(frame)
                self.timing["wake"].append((perf_counter() - started) * 1000)
                self.wake_count += 1
                pending.append((self.wake_count * self.wake.frame_samples, "wake", event))
                signals.append(
                    {"kind": "wake", "samples": self.wake_count * self.wake.frame_samples, **event}
                )
                if event["score"] is not None:
                    self.wake_peak = max(self.wake_peak or 0, event["score"])
        for end, kind, value in sorted(pending, key=lambda item: (item[0], item[1])):
            if kind == "wake":
                if value["triggered"]:
                    self.activations += 1
                    self.event("wake", end, phrase=value["phrase"], score=value["score"])
                    self.state = "listening" if self.session_policy else "external_session"
                    self.last_activity = end
                    if self.session_policy and self.latest_voiced:
                        self._onset(end)
                continue
            if value:
                if self.segment_start is None:
                    self.segment_start = end - 512
                self.last_voice = self.last_activity = end
                if self.state == "listening":
                    self._onset(end)
            elif self.segment_start is not None:
                self.segment_count += 1
                self.segments.append(
                    {"start": self.segment_start / 16000, "end": (end - 512) / 16000}
                )
                self.segment_start = None
            self.latest_voiced = value
            if self.state == "speech":
                reason = None
                if end - self.last_voice >= self.silence_frames * 512:
                    reason = "silence"
                elif end - self.utterance_start >= self.max_frames * 512:
                    reason = "cap"
                if reason:
                    self.event(
                        "endpoint",
                        end,
                        reason=reason,
                        range_start_seconds=self.utterance_start / 16000,
                    )
                    self.utterance_start = None
                    self.state = "listening"
                    self.last_activity = end
            if (
                self.session_policy
                and self.state == "listening"
                and end - self.last_activity >= self.idle_frames * 512
            ):
                self.event("session_timeout", end)
                self.wake.end_session()
                self.state = "wake_listening"
        return sorted(signals, key=lambda item: (item["samples"], item["kind"]))

    def finish(self):
        if not self.closed:
            self.tail = len(self.frames.finish()) // 2
            self.wake_tail = len(self.wake_frames.finish()) // 2
            self.closed = True

    def report(self):
        import numpy as np

        segments = list(self.segments)
        if self.segment_start is not None:
            segments.append(
                {
                    "start": self.segment_start / 16000,
                    "end": self.vad_count * 512 / 16000,
                    "open": not self.closed,
                }
            )
        return {
            "state": self.state,
            "finished": self.closed,
            "vad_frames": self.vad_count,
            "voiced_frames": self.voiced_count,
            "vad_segments": segments[-1000:],
            "vad_segments_truncated": self.segment_count > len(self.segments)
            or len(segments) > 1000,
            "vad_tail_samples_unscored": self.tail
            if self.closed
            else len(self.frames.pending) // 2,
            "wake_frames": self.wake_count,
            "wake_peak_score": self.wake_peak,
            "activation_count": self.activations,
            "wake_events": [{**e, "triggered": True} for e in self.events if e["event"] == "wake"],
            "frontend_events": list(self.events),
            "events_truncated": self.event_count > len(self.events),
            "wake_tail_samples_unscored": self.wake_tail
            if self.closed
            else len(self.wake_frames.pending) // 2,
            "incomplete_utterance": self.utterance_start is not None,
            "policy": {
                "pre_roll_ms": 320,
                "endpoint_silence_ms": self.silence_frames * 32,
                "inactivity_ms": self.idle_frames * 32,
                "max_utterance_ms": self.max_frames * 32,
            },
            "compute_ms": {
                name: {
                    "count_window": len(values),
                    "p50": float(np.percentile(values, 50)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                    "max": max(values) if values else None,
                }
                for name, values in self.timing.items()
            },
        }
