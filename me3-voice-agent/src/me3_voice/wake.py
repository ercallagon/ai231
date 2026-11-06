"""Wake policy and an optional pretrained openWakeWord ONNX adapter.

Model assets must be explicitly supplied under var/cache; no downloads, training,
provider calls or unreviewed phrase substitution occur during construction.
"""

from pathlib import Path
import hashlib
import importlib.metadata
from dataclasses import dataclass

import numpy as np

WAKE_SAMPLES = 1280  # openWakeWord's documented 80 ms frames at 16 kHz.


@dataclass(frozen=True)
class WakeDetection:
    phrase: str
    score: float | None = None


class OpenWakeWord:
    frame_samples = WAKE_SAMPLES
    backend = "openwakeword-onnx"

    def __init__(self, model_path: Path, mel_path: Path, embedding_path: Path):
        for path in (model_path, mel_path, embedding_path):
            if not path.is_file() or path.suffix != ".onnx":
                raise ValueError("supply explicit existing wake/mel/embedding ONNX files")
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord backend is not installed; see M1 compatibility gate"
            ) from exc
        self.model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(mel_path),
            embedding_model_path=str(embedding_path),
            device="cpu",
            ncpu=1,
        )
        self.phrase = model_path.stem
        self.provenance = {
            "version": importlib.metadata.version("openwakeword"),
            "asset_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (model_path, mel_path, embedding_path)
            },
            "execution_providers": {
                name: session.get_providers() for name, session in self.model.models.items()
            },
        }

    def score(self, pcm: bytes) -> float:
        if len(pcm) != WAKE_SAMPLES * 2:
            raise ValueError("wake backend expects 80 ms PCM16 mono at 16 kHz")
        scores = self.model.predict(np.frombuffer(pcm, dtype="<i2"))
        return max(float(score) for score in scores.values())

    def detect(self, pcm: bytes) -> WakeDetection:
        return WakeDetection(self.phrase, self.score(pcm))


class SherpaWakeWord:
    """Explicit local assets/keyword file, CPU only; no downloads or training.

    KeywordSpotter emits phrase events, not openWakeWord probabilities. Its
    decoder threshold/boost are configured separately from the gate's score.
    """

    frame_samples = 320
    backend = "sherpa-onnx-kws"

    def __init__(
        self,
        encoder: Path,
        decoder: Path,
        joiner: Path,
        tokens: Path,
        keywords: Path,
        *,
        threshold=0.25,
        boost=1.0,
    ):
        paths = (encoder, decoder, joiner, tokens, keywords)
        if any(not path.is_file() for path in paths):
            raise ValueError("supply explicit existing Sherpa model/token/keyword paths")
        if any(path.suffix != ".onnx" for path in paths[:3]):
            raise ValueError("Sherpa inference models must be ONNX files")
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("invalid Sherpa decoder threshold")
        if not np.isfinite(boost) or not 0 < boost <= 10:
            raise ValueError("invalid Sherpa decoder boost")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("install ME3's optional kws extra") from exc
        self.model = sherpa_onnx.KeywordSpotter(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            keywords_file=str(keywords),
            num_threads=1,
            provider="cpu",
            keywords_threshold=threshold,
            keywords_score=boost,
            max_active_paths=4,
        )
        self.stream = self.model.create_stream()
        self.provenance = {
            "version": importlib.metadata.version("sherpa-onnx"),
            "asset_sha256": {
                label: hashlib.sha256(path.read_bytes()).hexdigest()
                for label, path in zip(
                    ("encoder", "decoder", "joiner", "tokens", "keywords"), paths
                )
            },
            "provider": "cpu",
            "threads": 1,
            "threshold": threshold,
            "boost": boost,
            "max_active_paths": 4,
            "reset_policy": "after every detected phrase",
        }

    def detect(self, pcm: bytes) -> WakeDetection | None:
        if len(pcm) != self.frame_samples * 2:
            raise ValueError("Sherpa expects 20 ms PCM16 mono at 16 kHz")
        self.stream.accept_waveform(
            16000, np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        )
        phrase = None
        while self.model.is_ready(self.stream):
            self.model.decode_stream(self.stream)
            result = self.model.get_result(self.stream)
            if result:
                phrase = result
                self.model.reset_stream(self.stream)
        return WakeDetection(phrase) if phrase else None


class WakeGate:
    """One activation per session; cooldown uses stream samples, not wall clock.

    No VAD veto on the wake frame: an independent VAD can lag the wake phrase.
    Wake-score tuning must use separate development/qualification recordings.
    """

    def __init__(self, detector, threshold=0.5, cooldown_samples=16000):
        if not 0 < threshold <= 1 or cooldown_samples < 0:
            raise ValueError("invalid wake threshold/cooldown")
        self.detector, self.threshold = detector, threshold
        self.cooldown_samples = cooldown_samples
        self.samples = 0
        self.ready_at = 0
        self.active = False
        self.frame_samples = detector.frame_samples

    def feed(self, pcm):
        if len(pcm) != self.frame_samples * 2:
            raise ValueError("wake policy expects a complete backend frame")
        self.samples += self.frame_samples
        detection = self.detector.detect(pcm)
        score = detection.score if detection else None
        if score is not None and (not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("invalid wake score")
        detected = detection is not None and (score is None or score >= self.threshold)
        triggered = not self.active and self.samples >= self.ready_at and detected
        if triggered:
            self.active = True
        return {
            "score": score,
            "triggered": triggered,
            "audio_seconds": self.samples / 16000,
            "phrase": detection.phrase if detection else None,
            "backend": self.detector.backend,
        }

    def end_session(self):
        self.active = False
        self.ready_at = self.samples + self.cooldown_samples
