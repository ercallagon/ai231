import numpy as np
import pytest

from me3_voice.preprocessing import PCMFrames, PCMPreprocessor
from me3_voice.readiness import aec_config
from me3_voice.wake import WAKE_SAMPLES, WakeDetection, WakeGate


def convert(pcm, rate, chunks):
    pre = PCMPreprocessor(rate)
    parts = []
    offset = 0
    for index, size in enumerate(chunks):
        parts.append(pre.push(pcm[offset : offset + size], index))
        offset += size
    assert offset == len(pcm)
    parts.append(pre.finish())
    assert pre.finish() == b""
    return b"".join(parts), pre


def test_stream_resampling_matches_whole_signal_across_callback_boundaries():
    rate = 48000
    t = np.arange(rate) / rate
    values = np.rint(10000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
    pcm = values.tobytes()
    whole, _ = convert(pcm, rate, [len(pcm)])
    streamed, pre = convert(pcm, rate, [960] * 100)
    assert streamed == whole
    assert len(streamed) == 32000
    assert pre.input_stats.samples == 48000 and pre.output_stats.samples == 16000


def test_downsample_filters_high_frequency_instead_of_aliasing_into_speech():
    rate = 48000
    t = np.arange(rate) / rate
    rms = []
    for frequency in (1000, 12000):
        pcm = np.rint(10000 * np.sin(2 * np.pi * frequency * t)).astype("<i2").tobytes()
        converted, _ = convert(pcm, rate, [len(pcm)])
        signal = np.frombuffer(converted, "<i2").astype(float)[1000:-1000]
        rms.append(np.sqrt(np.mean(signal**2)))
    assert rms[1] < rms[0] * 0.01


def test_diagnostics_expose_clipping_and_stereo_downmix_without_gain_hiding_it():
    pre = PCMPreprocessor(16000, channels=2)
    pcm = np.array([[32767, 32767], [-32768, -32768]] * 100, dtype="<i2").tobytes()
    result = pre.push(pcm, 0) + pre.finish()
    assert len(result) == 400
    assert pre.input_stats.report()["clipped_fraction"] == 1
    assert pre.input_stats.report()["peak"] == 1
    with pytest.raises(ValueError, match="finished"):
        pre.push(pcm, 1)


def test_capture_gaps_and_model_frame_assembly_preserve_sample_accounting():
    pre = PCMPreprocessor(48000)
    with pytest.raises(ValueError, match="sequence"):
        pre.push(b"\0\0", 1)
    with pytest.raises(ValueError, match="PCM16"):
        pre.push(b"\0", 0)
    frames = PCMFrames(512)
    original = b"\1\0" * 1300
    output = frames.push(original[:600]) + frames.push(original[600:])
    tail = frames.finish()
    assert b"".join(output) + tail == original
    assert len(output) == 2 and len(tail) == (1300 - 1024) * 2


def test_wake_once_per_session_and_stream_clock_cooldown():
    class Scores:
        frame_samples = WAKE_SAMPLES
        backend = "test"

        def detect(self, pcm):
            return WakeDetection("test", 0.9)

    gate = WakeGate(Scores(), cooldown_samples=2 * WAKE_SAMPLES)
    frame = bytes(WAKE_SAMPLES * 2)
    assert gate.feed(frame)["triggered"]
    assert not gate.feed(frame)["triggered"]
    gate.end_session()
    assert not gate.feed(frame)["triggered"]
    assert gate.feed(frame)["triggered"]


def test_phrase_events_have_no_fabricated_score_and_keep_session_cooldown():
    class Phrases:
        frame_samples = 320
        backend = "sherpa-test"

        def detect(self, pcm):
            return WakeDetection("HEY_SWIFT")

    gate = WakeGate(Phrases(), cooldown_samples=640)
    frame = bytes(640)
    first = gate.feed(frame)
    assert first["triggered"] and first["score"] is None
    assert first["phrase"] == "HEY_SWIFT" and first["audio_seconds"] == 0.02
    assert not gate.feed(frame)["triggered"]
    gate.end_session()
    assert not gate.feed(frame)["triggered"]
    assert gate.feed(frame)["triggered"]
    with pytest.raises(ValueError, match="complete"):
        gate.feed(b"\0\0")


def test_real_sherpa_silence_when_optional_assets_are_present():
    from pathlib import Path
    from me3_voice.wake import SherpaWakeWord

    directory = Path("var/cache/wake/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01")
    keywords = Path("var/cache/wake/hey-swift-keywords.txt")
    if not directory.is_dir() or not keywords.is_file():
        pytest.skip("explicit optional Sherpa assets/keywords absent")
    pytest.importorskip("sherpa_onnx")
    suffix = "epoch-12-avg-2-chunk-16-left-64.int8.onnx"
    gate = WakeGate(
        SherpaWakeWord(
            *(directory / f"{name}-{suffix}" for name in ("encoder", "decoder", "joiner")),
            directory / "tokens.txt",
            keywords,
        )
    )
    assert not any(gate.feed(bytes(640))["triggered"] for _ in range(100))
    assert gate.detector.provenance["provider"] == "cpu"


def test_aec_artifact_requires_explicit_physical_targets_and_rejects_injection():
    config = aec_config("alsa_input.usb-test.mono-fallback", "alsa_output.usb-test.analog-stereo")
    assert 'target.object = "alsa_input.usb-test.mono-fallback"' in config
    assert 'target.object = "alsa_output.usb-test.analog-stereo"' in config
    assert "libpipewire-module-echo-cancel" in config
    for capture in ('bad" }', "me3.aec.source"):
        with pytest.raises(ValueError):
            aec_config(capture, "physical-output")
    with pytest.raises(ValueError):
        aec_config("same-node", "same-node")


def test_real_openwakeword_onnx_silence_when_selected_assets_are_present():
    from pathlib import Path
    from me3_voice.wake import OpenWakeWord

    directory = Path("var/cache/wake/openwakeword-v0.5.1")
    paths = [
        directory / name
        for name in ("hey_jarvis_v0.1.onnx", "melspectrogram.onnx", "embedding_model.onnx")
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("selected optional pretrained ONNX assets are absent")
    pytest.importorskip("openwakeword")
    detector = OpenWakeWord(*paths)
    gate = WakeGate(detector)
    events = [gate.feed(bytes(WAKE_SAMPLES * 2)) for _ in range(40)]
    assert not any(event["triggered"] for event in events)
    assert all(
        providers == ["CPUExecutionProvider"]
        for providers in detector.provenance["execution_providers"].values()
    )


def test_replay_refuses_report_overwriting_input_evidence(tmp_path):
    from argparse import Namespace
    from me3_voice.readiness import replay

    recording = tmp_path / "evidence.wav"
    recording.write_bytes(b"preserve original evidence")
    args = Namespace(
        wav=recording,
        output=recording,
        processed_wav=None,
        wake_model=None,
        mel_model=None,
        embedding_model=None,
    )
    with pytest.raises(ValueError, match="overwrite input evidence"):
        replay(args)
    assert recording.read_bytes() == b"preserve original evidence"
