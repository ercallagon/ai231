"""Cue-only digital gain, peak-limited without changing the original asset."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf


@contextmanager
def prepared_cue(path, gain_db=0.):
    path = Path(path)
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Cue contains empty/nonfinite audio")
    peak = float(np.max(np.abs(audio)))
    requested = 10 ** (gain_db / 20)
    gain = min(requested, .98 / peak) if gain_db > 0 and peak else requested
    metadata = {"asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "requested_gain_db": gain_db, "applied_gain_db": round(20 * np.log10(gain), 3),
                "source_peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
                "source_rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(audio ** 2))), 1e-12)), 3),
                "peak_limited": gain < requested, "seconds": len(audio) / rate}
    if gain_db == 0:
        yield path, metadata
    else:
        with tempfile.TemporaryDirectory(prefix="swift-cue-") as directory:
            output = Path(directory) / "cue.wav"
            sf.write(output, audio * gain, rate, subtype="PCM_16")
            yield output, metadata
