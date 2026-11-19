"""Native conversation composition; no Session, custom tool loop or scheduler.

Callers own transport activation and SpeechClient lifetime. LLM and transports
are supplied framework processors; importing this module never opens devices or
contacts a provider. Engagement/UI/task migration belongs to later phases.
"""

from dataclasses import dataclass

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from .native_speech import McpSTTService, McpTTSService


class ASRConfirmedUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Require recognized text before superseding the current response.

    Interruptions are disabled here because Pipecat broadcasts an interruption
    in both directions, which would abort the ASR stream carrying this turn.
    NativeApplication cancels the old assistant branch after this event.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_evidence = None

    async def process_frame(self, frame):
        if isinstance(frame, InterimTranscriptionFrame):
            if not frame.text.strip():
                return ProcessFrameResult.CONTINUE
            self.start_evidence = "asr_interim"
            await self.trigger_user_turn_started(enable_interruptions=False)
            return ProcessFrameResult.STOP
        if isinstance(frame, TranscriptionFrame):
            if not frame.text.strip():
                return ProcessFrameResult.CONTINUE
            # A final-only ASR provider reaches the aggregator after the text
            # was appended. Do not reset the aggregator or abort its ASR stream.
            self.start_evidence = "asr_final"
            await self.trigger_user_turn_started(enable_interruptions=False)
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


@dataclass(frozen=True)
class NativePipelineParts:
    """Handles only; owns no state machine, tasks, queues or conversation copy."""

    pipeline: Pipeline
    stt: McpSTTService
    tts: McpTTSService
    aggregators: LLMContextAggregatorPair


def make_native_pipeline(
    speech,
    llm,
    output,
    *,
    context=None,
    ingress=(),
    after_tts=(),
    session_id=None,
    batch_ms=32,
    silence_seconds=0.5,
    on_event=None,
    on_partial=None,
    user_idle_seconds=0,
):
    if not 0 <= silence_seconds <= 3:
        raise ValueError("silence_seconds must be 0..3")
    aggregators = LLMContextAggregatorPair(
        context if context is not None else LLMContext(),
        user_params=LLMUserAggregatorParams(
            user_idle_timeout=user_idle_seconds,
            empty_user_turn=None,
            user_turn_strategies=UserTurnStrategies(
                start=[ASRConfirmedUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=silence_seconds)],
            ),
        ),
    )
    stt = McpSTTService(
        speech,
        session_id=session_id,
        batch_ms=batch_ms,
        on_event=on_event,
        on_partial=on_partial,
    )
    tts = McpTTSService(speech, session_id=session_id, on_event=on_event)
    # Exactly one native context pair, one turn controller, and one TTS/output
    # scheduling path. Ingress must supply VAD events (real analyzer or replay).
    pipeline = Pipeline(
        [*ingress, stt, aggregators.user(), llm, tts, *after_tts, output, aggregators.assistant()]
    )
    return NativePipelineParts(pipeline, stt, tts, aggregators)
