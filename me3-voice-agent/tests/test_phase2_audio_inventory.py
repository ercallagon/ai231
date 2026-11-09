"""Private corpus integrity and acquisition admission contracts; no device access."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase2"))
from audio_inventory import inventory


def test_saved_take_integrity_does_not_imply_transcript_or_provider_consent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path("var/recordings/guided/example")
    root.mkdir(parents=True)
    clip = root / "voice.wav"
    with wave.open(str(clip), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    data = {
        "kind": "swift-guided-recordings",
        "suite": "timing",
        "consent": {"project_storage": True, "other_providers": False},
        "cases": [{"id": "pause", "expected_text": "Read this prompt."}],
        "takes": [
            {
                "case_id": "pause",
                "file": "voice.wav",
                "complete": True,
                "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(data))
    result = inventory([Path("var/recordings")])
    assert result["guided_original_count"] == 1
    assert result["guided_original_seconds"] == 1
    take = result["files"][0]["guided"]
    assert take["manifest_integrity"]["hash_matches"]
    assert not take["manifest_integrity"]["expected_text_is_human_verified_transcript"]
    assert take["consent"]["other_providers"] is False
    clip.write_bytes(b"changed and invalid audio")
    result = inventory([Path("var/recordings")])
    assert not result["files"][0]["guided"]["manifest_integrity"]["hash_matches"]
    assert "probe_error" in result["files"][0]
    clip.unlink()
    result = inventory([Path("var/recordings")])
    assert result["guided_original_count"] == 0
    assert not result["manifests"][0]["takes"][0]["present"]


@pytest.mark.parametrize("profile", ["speech", "negative"])
def test_corpus_capture_requires_explicit_opt_in_even_without_reference(tmp_path, profile):
    script = Path(__file__).resolve().parents[1] / "scripts/phase2/operator_native.py"
    output = Path.cwd() / "var/scratch/phase2-corpus-disabled-must-not-exist"
    assert not output.exists()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--profile",
            profile,
            "--source",
            "me3.phase2.aec.source",
            "--sink",
            "me3.phase2.aec.sink",
            "--raw-mic",
            "physical.mic",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Capture disabled" in result.stdout
    assert not output.exists()


def test_playback_profile_cannot_capture_without_known_reference():
    script = Path(__file__).resolve().parents[1] / "scripts/phase2/operator_native.py"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--profile",
            "overlap",
            "--source",
            "me3.phase2.aec.source",
            "--sink",
            "me3.phase2.aec.sink",
            "--raw-mic",
            "physical.mic",
            "--output",
            "var/scratch/never-opened",
            "--capture",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "require a local reference WAV" in result.stderr


@pytest.mark.parametrize("enabled", [True, False])
async def test_installed_native_vad_strategy_carries_interruption_policy(enabled):
    from pipecat.frames.frames import VADUserStartedSpeakingFrame
    from pipecat.turns.user_start import VADUserTurnStartStrategy

    strategy = VADUserTurnStartStrategy(enable_interruptions=enabled)
    observed = []

    @strategy.event_handler("on_user_turn_started")
    async def started(_strategy, params):
        observed.append(params.enable_interruptions)

    await strategy.process_frame(VADUserStartedSpeakingFrame())
    assert observed == [enabled]
