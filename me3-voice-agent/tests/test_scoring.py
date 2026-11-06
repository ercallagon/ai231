import json

import pytest

from me3_voice.dashboard import normalize
from me3_voice.scoring import score


def fixture(tmp_path, windows):
    report = tmp_path / "replay.json"
    labels = tmp_path / "labels.json"
    report.write_text(
        json.dumps(
            {
                "mode": "HPC-live-file-inference",
                "input_sha256": "test-pcm-hash",
                "audio_origin": "recorded",
                "finished": True,
                "run_status": "eof",
                "hash_scope": "consumed PCM bytes, excluding WAV header",
                "output_stats": {"samples": 16000},
                "vad_frames": 31,
                "vad_segments": [{"start": 0.032, "end": 0.128}],
                "wake_events": [
                    {"audio_seconds": 0.2},
                    {"audio_seconds": 0.21},
                    {"audio_seconds": 0.8},
                ],
            }
        )
    )
    labels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "input_sha256": "test-pcm-hash",
                "audio_origin": "recorded",
                "split": "development",
                "wake_windows": windows,
                "speech_ranges": [{"start": 0, "end": 0.096}],
            }
        )
    )
    return report, labels


def test_matching_counts_misses_duplicates_and_frame_confusion(tmp_path):
    report, labels = fixture(tmp_path, [{"start": 0.1, "end": 0.3}, {"start": 0.5, "end": 0.6}])
    result = score(report, labels)
    assert (result["wake_detected"], result["wake_missed"], result["false_activations"]) == (
        1,
        1,
        2,
    )
    assert (result["vad_tp"], result["vad_fp"], result["vad_fn"], result["vad_tn"]) == (2, 1, 1, 27)
    assert result["vad_precision"] == result["vad_recall"] == 2 / 3
    assert result["negative_exposure_hours"] is None
    kind, evidence, view = normalize(result)
    assert kind == "frontend_score" and evidence == "HPC-live"
    assert view["replay_sha256"] == result["replay_sha256"]


def test_negative_exposure_and_reject_mismatched_annotations(tmp_path):
    report, labels = fixture(tmp_path, [])
    result = score(report, labels)
    assert result["false_activations_per_hour"] == 10800
    data = json.loads(labels.read_text())
    data["input_sha256"] = "other-recording"
    labels.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="matching"):
        score(report, labels)


def test_reject_partial_or_truncated_replay_and_overlapping_labels(tmp_path):
    report, labels = fixture(tmp_path, [{"start": 0.1, "end": 0.4}, {"start": 0.3, "end": 0.5}])
    with pytest.raises(ValueError, match="nonoverlapping"):
        score(report, labels)
    fixture(tmp_path, [])
    data = json.loads(report.read_text())
    data["events_truncated"] = True
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="untruncated"):
        score(report, labels)
