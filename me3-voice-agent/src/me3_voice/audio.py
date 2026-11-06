"""File replay, activation pre-roll and endpoint policy; no audio device is opened."""

import wave
import math
from collections import deque
from pathlib import Path
from typing import Protocol

from .contracts import FRAME_BYTES, FRAME_SECONDS, MAX_AUDIO_BYTES, RATE


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (
            1,
            2,
            RATE,
            "NONE",
        ):
            raise ValueError("recording must be uncompressed mono PCM16 at 16000 Hz")
        if wav.getnframes() > MAX_AUDIO_BYTES // 2:
            raise ValueError("recording exceeds 30 seconds; split it explicitly")
        return wav.readframes(wav.getnframes())


class VAD(Protocol):
    def voiced(self, pcm: bytes) -> bool: ...


class SileroVAD:
    def __init__(self, confidence: float = 0.7):
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("VAD confidence must be finite and between zero and one")
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        self.analyzer = SileroVADAnalyzer(sample_rate=RATE)
        self.analyzer.set_sample_rate(RATE)
        self.confidence = confidence

    def voiced(self, pcm: bytes) -> bool:
        if len(pcm) != FRAME_BYTES:
            raise ValueError("Silero requires 512 samples; pad only at file boundary")
        return self.analyzer.voice_confidence(pcm).item() >= self.confidence


class ReplayInput:
    """Wake events are explicitly injected, never inferred from a mock transcript.

    320 ms pre-roll, 640 ms trailing silence and 30 s cap. One emitted utterance
    may include the wake phrase; stripping it belongs to a future verified engine.
    """

    def __init__(self, vad: VAD, pre_roll_frames=10, silence_frames=20, max_frames=937):
        if min(pre_roll_frames, silence_frames, max_frames) < 1:
            raise ValueError("audio frame bounds must be positive")
        self.vad = vad
        self.ring = deque(maxlen=pre_roll_frames)
        self.silence_frames, self.max_frames = silence_frames, max_frames
        self.active = False
        self.utterance: list[bytes] = []
        self.has_speech = False
        self.quiet = 0
        self.frames = 0
        self.latest_voiced = False
        self.events = deque(maxlen=1000)
        self.event_count = 0

    def record_event(self, event):
        self.event_count += 1
        self.events.append(event)

    def activate(self, event="wake_injected", seed_speech=False):
        if self.active:
            return
        self.active = True
        self.record_event({"event": event, "audio_seconds": self.frames * FRAME_SECONDS})

        if seed_speech and self.latest_voiced and self.ring:
            self.utterance = list(self.ring)
            self.ring.clear()
            self.has_speech = True
            self.record_event({"event": "vad_onset", "audio_seconds": self.frames * FRAME_SECONDS})

    def reset(self):
        self.active = False
        self.ring.clear()
        self.utterance.clear()
        self.has_speech, self.quiet = False, 0
        self.latest_voiced = False

    def feed(self, frame: bytes) -> bytes | None:
        if len(frame) != FRAME_BYTES:
            raise ValueError("expected 32 ms PCM frame")
        return self.feed_scored(frame, self.vad.voiced(frame))

    def feed_scored(self, frame, voiced, *, allow_endpoint=True):
        if len(frame) != FRAME_BYTES or type(voiced) is not bool:
            raise ValueError("expected complete PCM frame and boolean VAD decision")
        self.frames += 1
        self.latest_voiced = voiced
        if not self.active:
            self.ring.append(frame)
            return None
        if not self.has_speech:
            if not voiced:
                self.ring.append(frame)
                return None
            self.utterance = list(self.ring)
            self.ring.clear()
            self.has_speech = True
            self.record_event({"event": "vad_onset", "audio_seconds": self.frames * FRAME_SECONDS})
        self.utterance.append(frame)
        self.quiet = 0 if voiced else self.quiet + 1
        if (self.quiet >= self.silence_frames and allow_endpoint) or len(
            self.utterance
        ) >= self.max_frames:
            result = b"".join(self.utterance)
            self.record_event(
                {
                    "event": "endpoint",
                    "audio_seconds": self.frames * FRAME_SECONDS,
                    "reason": "silence" if self.quiet >= self.silence_frames else "cap",
                }
            )
            self.utterance.clear()
            self.has_speech, self.quiet = False, 0
            return result
        return None


class ConversationInput:
    """Wire file VAD onsets to cancellation without losing the incoming utterance.

    Feed from one serial capture consumer. No echo cancellation is provided: a
    future device source must supply echo-controlled PCM before this boundary.
    """

    def __init__(
        self,
        source: ReplayInput,
        session,
        turn_analyzer=None,
        *,
        barge_min_voiced_frames=1,
        barge_window_frames=8,
    ):
        if not 1 <= barge_min_voiced_frames <= barge_window_frames <= 32:
            raise ValueError("invalid barge-in confirmation window")
        self.source, self.session = source, session
        self.turn_analyzer = turn_analyzer
        self.barge_min_voiced_frames = barge_min_voiced_frames
        self.barge_voiced = deque(maxlen=barge_window_frames)
        self.input_failed = False

    def activate(self, event="wake_injected", seed_speech=False):
        self.input_failed = False
        self.candidate_started_at = self.session.clock()
        self.session.activate(event)
        self.source.activate(event, seed_speech)
        if seed_speech and self.source.has_speech:
            self.session.event("vad_candidate_started")

    async def feed(self, frame, voiced=None):
        from .session import State

        if self.session.state == State.IDLE and self.source.active:
            self.source.reset()
        had_speech = self.source.has_speech
        if voiced is None:
            voiced = bool(self.source.vad.voiced(frame))
        allow_endpoint = True
        if self.turn_analyzer and self.source.active:
            from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

            self.turn_analyzer.append_audio(frame, voiced)
            quiet = 0 if voiced else self.source.quiet + 1
            if self.source.has_speech and quiet >= self.source.silence_frames:
                allow_endpoint = quiet >= max(64, self.source.silence_frames)
                if not allow_endpoint and (quiet - self.source.silence_frames) % 10 == 0:
                    self.session.event("turn_analysis_started")
                    try:
                        state, _ = await self.turn_analyzer.analyze_end_of_turn()
                        allow_endpoint = state == EndOfTurnState.COMPLETE
                    except Exception:
                        # Bounded silence fallback; never promote an inference
                        # failure into evidence that the speaker finished.
                        self.session.event("turn_analysis_error")
                    finally:
                        self.session.event("turn_analysis_finished")
                if quiet >= max(64, self.source.silence_frames):
                    self.session.event("turn_silence_fallback")
        result = self.source.feed_scored(frame, voiced, allow_endpoint=allow_endpoint)
        candidate_started = not had_speech and (self.source.has_speech or result is not None)
        if candidate_started:
            self.candidate_started_at = self.session.clock()
            deadline = self.session.engagement_deadline
            if deadline is None or self.candidate_started_at < deadline:
                self.session.candidate_started_at = self.candidate_started_at
            self.session.event("vad_candidate_started")
        if result and self.turn_analyzer:
            self.turn_analyzer.clear()
        if not self.source.active:
            return result
        # During reasoning or playback, one VAD-positive frame can be leakage. Keep
        # capturing the candidate, but do not cancel output until it persists.
        # The source retains pre-roll and every candidate frame for a real barge-in.
        needs_confirmation = (
            self.session.state in {State.SPEAKING, State.THINKING} or self.session.music_playing
        )
        if needs_confirmation and not self.session.input_open:
            self.barge_voiced.append(voiced)
            voiced_frames = sum(self.barge_voiced)
            if voiced_frames < self.barge_min_voiced_frames:
                if result:
                    self.barge_voiced.clear()
                    self.session.candidate_started_at = None
                    self.session.event(
                        "barge_candidate_rejected",
                        voiced_frames=voiced_frames,
                        window_frames=self.barge_voiced.maxlen,
                    )
                return None
            if candidate_started or self.barge_voiced:
                self.session.event(
                    "vad_candidate_confirmed",
                    voiced_frames=voiced_frames,
                    window_frames=self.barge_voiced.maxlen,
                    required_voiced_frames=self.barge_min_voiced_frames,
                )
            self.barge_voiced.clear()
        else:
            self.barge_voiced.clear()
            if candidate_started:
                self.session.event("vad_candidate_confirmed")
        recognition = self.session.asr_task
        if (
            had_speech
            and recognition
            and recognition.done()
            and not recognition.cancelled()
            and recognition.exception() is not None
        ):
            await self.session.interrupt()
            self.input_failed = True
            self.session.event("input_discarded_after_provider_error")
        if self.input_failed:
            if result:
                self.input_failed = False
            return result
        try:
            if (self.source.has_speech or result) and not self.session.input_open:
                self.session.event("vad_onset_file")
                await self.session.begin_input(
                    started_at=getattr(self, "candidate_started_at", None)
                )
                await self.session.push_input(result if result else b"".join(self.source.utterance))
            elif had_speech and self.session.input_open:
                await self.session.push_input(frame)
            if voiced is True and self.session.key:
                self.session.event("last_voiced_frame_consumed_file")
            if result:
                await self.session.commit_input()
        except Exception as exc:
            if self.session.key:
                self.session.event(
                    "turn_error",
                    error=type(exc).__name__,
                    error_code=getattr(exc, "code", "input_error"),
                    error_role="asr",
                    error_stage="asr",
                )
            await self.session.interrupt()
            # Drain this failed utterance to its endpoint; never retry its audio
            # automatically. A subsequent user turn may use a recovered service.
            self.input_failed = result is None
            self.session.event("input_discarded_after_provider_error")
        return result
