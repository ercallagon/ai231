#!/usr/bin/env python3
"""Build a precomputed augmented-feature store for the VCM (one-time, offline).

WHY
---
On-the-fly `full` augmentation (speed + noise + gain + spec_aug) inside the
dataloader cost ~42 min/epoch at 34k rows. This script bakes the augmentation
in ONCE, offline, into a compact NumPy store so training reads fixed tensors.

SINGLE-PROCESS BY DESIGN (no multiprocessing/spawn)
----------------------------------------------------
An 8-worker spawn Pool deadlocked on this NFS box (futex_wait storm, 0 CPU).
The transform is a single-process loop; at ~10-20 ms/clip it is fast enough
for pilot sizes (5k -> ~2 min) and avoids the deadlock class entirely.

PROGRESS REPORTING
------------------
Every ~100 clips: count/total, %, elapsed, clips/s, ETA, RSS.

DETERMINISTIC AUGMENTATION
--------------------------
Per-row RNG: np.random.default_rng(seed * 1000 + row_position). The same row
in the same manifest always gets the same augmented features, so the store is
reproducible bit-for-bit and later scaling runs (10k/15k/34k) built from the
same source manifest stay directly comparable to the pilot.

SEGREGATION (hard rule from the user)
-------------------------------------
* HF source-of-truth `data/vcm/raw/` is READ-ONLY here (never written).
* Committed manifests under `data/vcm/manifests/` are untouched.
* Everything produced lives in its OWN directory:
      data/vcm/precomputed_aug/<store_tag>/
        mels.npy    float32 (N, 1, 80, 198)
        index.json  one entry per row, manifest order
        meta.json   recipe, counts, seed, sha256, provenance
        README.md   human documentation

Usage:
    python3 scripts/build_precomputed_aug.py \
        --manifest data/vcm/pilot/run_011_pilot_5k/train_19intent.csv \
        --out-tag run_011_pilot_5k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hey_swift.vcm import dataset as ds  # noqa: E402
from hey_swift.vcm.augmentation import AugConfig, augment_mel  # noqa: E402
from hey_swift.vcm.features import FeatureConfig, _mel_filterbank  # noqa: E402

PRECOMP_ROOT = REPO_ROOT / "data" / "vcm" / "precomputed_aug"
CACHE_DIR = REPO_ROOT / "work" / "mfcc_cache"
PROCESSED_WAV = REPO_ROOT / "data" / "vcm" / "processed" / "wav"

AUG_PARAMS = dict(
    speed=(0.9, 1.0, 1.1),
    noise_prob=0.3,
    snr_db=(10.0, 25.0),
    gain_db=(-6.0, 6.0),
)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return -1.0


def _vectorized_mel(wav: np.ndarray, fb: np.ndarray, cfg: FeatureConfig,
                    target_frames: int) -> np.ndarray:
    """Waveform -> (1, n_mels, T) log-mel, vectorized. Matches MelFeatures."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    n_fft, hop = cfg.n_fft, cfg.hop_length
    if len(wav) >= n_fft:
        n_avail = 1 + (len(wav) - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(
            wav, shape=(n_avail, n_fft),
            strides=(wav.strides[0], wav.strides[0]),
        )
    else:
        frames = np.zeros((1, n_fft), dtype=np.float32)
        frames[0, :len(wav)] = wav
    window = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = fb @ spec.T
    log_mel = np.log(mel + 1e-5)
    t = log_mel.shape[1]
    if t >= target_frames:
        start = (t - target_frames) // 2
        log_mel = log_mel[:, start:start + target_frames]
    else:
        pad = target_frames - t
        log_mel = np.pad(log_mel, ((0, 0), (pad // 2, pad - pad // 2)), mode="edge")
    return log_mel[np.newaxis, :, :].astype(np.float32)


def _load_wav_local(ap: str) -> np.ndarray:
    """Load a clip: extracted-WAV fast path, else HF-parquet fallback.

    The parquet fallback pins the row-group index ONCE (module-global) and
    resolves the id-map path relative to the REPO ROOT. The original
    dataset._rg_index resolved work/tmp/file_to_parquet.json relative to the
    CWD, which missed under a relative working dir and fell back to a full
    recursive scan of 1,038 row groups x 16 files (hours, per clip).
    """
    cand = PROCESSED_WAV / ap
    if cand.is_file():
        raw = cand.read_bytes()
    else:
        raw = _load_from_parquet(ap)
    import io
    import wave
    if raw[:4] == b"RIFF":
        w = wave.open(io.BytesIO(raw))
        n = w.getnframes()
        pcm = w.readframes(n)
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width == 2:
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            x = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported sample width {width}")
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return x
    import soundfile as sf
    x, _ = sf.read(io.BytesIO(raw), dtype="float32")
    return np.asarray(x, dtype=np.float32).reshape(-1)


_RG_INDEX: dict | None = None
_RAW_ROOT = REPO_ROOT / "data" / "vcm" / "raw"
_ID_MAP = REPO_ROOT / "work" / "tmp" / "file_to_parquet.json"


def _rg_index() -> dict:
    """Pin file-id -> (parquet_path, row_group) ONCE.

    Uses the prebuilt work/tmp/file_to_parquet.json map (file-id -> parquet
    path, relative to repo root), then scans each parquet's row groups'
    `file` column once (~30-60 s for 16 files). Paths are pinned to the
    REPO ROOT so the index is CWD-independent (the original
    dataset._build_rg_index resolved the map relative to the CWD and missed
    under a relative working dir, falling back to a full recursive scan).
    """
    global _RG_INDEX
    if _RG_INDEX is not None:
        return _RG_INDEX
    import pyarrow.parquet as pq
    idx: dict = {}
    if _ID_MAP.is_file():
        fmap = json.loads(_ID_MAP.read_text())
        by_pq: dict[str, list[str]] = {}
        for fid, pq_rel in fmap.items():
            by_pq.setdefault(str(REPO_ROOT / pq_rel), []).append(str(fid))
        for pq_path, fids in by_pq.items():
            pf = pq.ParquetFile(pq_path)
            for rg in range(pf.metadata.num_row_groups):
                names = pf.read_row_group(rg, columns=["file"]).to_pydict()["file"]
                for nm in names:
                    idx.setdefault(str(nm), (pq_path, rg))
        print(f"[build] rg-index from id-map: {len(idx)} entries "
              f"across {len(by_pq)} parquet files", flush=True)
    if not idx:
        import glob
        for pq_path in sorted(glob.glob(str(_RAW_ROOT / "**" / "*.parquet"), recursive=True)):
            pf = pq.ParquetFile(pq_path)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["file"]).to_pydict()
                for fid in tbl["file"]:
                    idx.setdefault(str(fid), (pq_path, rg))
        print(f"[build] rg-index built by scan: {len(idx)} entries", flush=True)
    _RG_INDEX = idx
    return idx


def _load_from_parquet(ap: str) -> bytes:
    import pyarrow.parquet as pq
    hit = _rg_index().get(str(ap))
    if hit is None:
        raise FileNotFoundError(f"row id {ap!r} not in rg-index")
    pq_path, rg = hit
    tbl = pq.ParquetFile(pq_path).read_row_group(rg, columns=["file", "audio"]).to_pydict()
    for i, fid in enumerate(tbl["file"]):
        if str(fid) == str(ap):
            return ds.decode_audio(tbl["audio"][i])
    raise FileNotFoundError(f"row id {ap!r} not in row group {rg} of {pq_path}")


def _augment_waveform(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """speed_perturb -> add_noise -> apply_gain (waveform domain)."""
    w = w.copy()
    # speed perturbation (linear-interp stride, same as augmentation.perturb_speed)
    factor = float(rng.choice(AUG_PARAMS["speed"]))
    if factor != 1.0:
        n_out = int(len(w) / factor)
        xi = np.linspace(0, len(w) - 1, n_out)
        ii = np.floor(xi).astype(int)
        ij = np.minimum(ii + 1, len(w) - 1)
        fr = xi - ii
        w = (w[ii] * (1 - fr) + w[ij] * fr).astype(np.float32)
    # additive white noise at random SNR
    if rng.random() < AUG_PARAMS["noise_prob"]:
        snr = float(rng.uniform(*AUG_PARAMS["snr_db"]))
        sig_pow = np.mean(w ** 2) + 1e-9
        noise = rng.standard_normal(len(w)).astype(np.float32)
        noise *= np.sqrt(sig_pow / (10 ** (snr / 10.0)))
        w = (w + noise).astype(np.float32)
    # linear gain
    db = float(rng.uniform(*AUG_PARAMS["gain_db"]))
    w = (w * (10 ** (db / 20.0))).astype(np.float32)
    return w


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-reproduce", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()
    if not args.legacy_reproduce:
        ap.error("Legacy-only waveform store. Corrected trials use prepare_vcm_protocol.py --cache plus train-only online SpecAugment.")
    if pd.read_csv(args.manifest, usecols=["audio_path"])["audio_path"].str.startswith("hf:").any():
        ap.error("Source-qualified manifests must never use legacy filename-based caches")

    t_start = time.time()
    out_dir = PRECOMP_ROOT / args.out_tag
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite historical store: {out_dir}")
    out_dir.mkdir(parents=True)

    feat_cfg = FeatureConfig()  # 16k, n_fft 400, hop 160, 80-mel, 2.0s -> (1,80,198)
    target_frames = 1 + (feat_cfg.window_samples - feat_cfg.n_fft) // feat_cfg.hop_length
    fb = _mel_filterbank(feat_cfg.sample_rate, feat_cfg.n_fft, feat_cfg.n_mels,
                         feat_cfg.f_min, feat_cfg.f_max)

    man = Path(args.manifest)
    df = pd.read_csv(man)
    print(f"[build] manifest={man}")
    print(f"[build] rows={len(df)}  target_frames={target_frames}  "
          f"deterministic_seed={args.seed}", flush=True)

    # ---- MFCC cache (clean, unaugmented mels for the standard clips) --------
    cache = None
    cache_idx = {}
    if (CACHE_DIR / "index.json").exists() and (CACHE_DIR / "mels.npy").exists():
        cache_idx = json.loads((CACHE_DIR / "index.json").read_text())
        cache = np.load(CACHE_DIR / "mels.npy", mmap_mode="r")
        print(f"[build] MFCC cache: {len(cache_idx)} clips", flush=True)

    n = len(df)
    mels = np.empty((n, 1, feat_cfg.n_mels, target_frames), dtype=np.float32)
    index = []
    n_cache = n_decode = n_fail = 0
    t0 = time.time()

    # ---- PRE-DECODE PHASE ---------------------------------------------------
    # Decode all non-cached clips UPFRONT (serial, id-map fast path) so the
    # transform loop below is purely in-memory. This eliminates the mid-loop
    # decode stalls that killed the 3 prior builds (they died at ~92% where
    # the OOS decode-heavy rows sit). With the _rg_index path fix, each decode
    # is a single parquet row-group read (~5-10 ms), so 399 clips -> ~5 s.
    wav_preload: dict[int, np.ndarray | None] = {}
    if cache is not None:
        decode_positions = [
            i for i in range(n)
            if str(df.iloc[i]["audio_path"]) not in cache_idx
        ]
    else:
        decode_positions = list(range(n))
    if decode_positions:
        t_pd = time.time()
        print(f"[build] pre-decoding {len(decode_positions)} non-cached clips "
              f"(serial, id-map fast path)...", flush=True)
        for k, pos in enumerate(decode_positions):
            apath = str(df.iloc[pos]["audio_path"])
            try:
                wav_preload[pos] = _load_wav_local(apath)
            except Exception:
                wav_preload[pos] = None
            if (k + 1) % 50 == 0 or k + 1 == len(decode_positions):
                el = time.time() - t_pd
                print(f"[build]   pre-decode {k + 1}/{len(decode_positions)} "
                      f"el={el:.1f}s rss={_rss_gb():.1f}GB", flush=True)
        n_ok = sum(1 for v in wav_preload.values() if v is not None and len(v) > 0)
        print(f"[build] pre-decode done: {n_ok}/{len(decode_positions)} OK "
              f"in {time.time() - t_pd:.1f}s", flush=True)

    for i in range(n):
        row = df.iloc[i]
        apath = str(row["audio_path"])
        rng = np.random.default_rng(args.seed * 1000 + i)  # deterministic per row
        try:
            if cache is not None and apath in cache_idx:
                # cached clean mel: spec_aug only (no waveform-level aug)
                mel = np.array(cache[cache_idx[apath]], dtype=np.float32)
                mel = augment_mel(mel, AugConfig(enabled=True), rng)
                src = "cache"
                n_cache += 1
            else:
                wav = wav_preload.get(i)
                n_decode += 1
                if wav is None or len(wav) == 0:
                    mel = np.zeros((1, feat_cfg.n_mels, target_frames), dtype=np.float32)
                    n_fail += 1
                    src = "decoded_empty"
                else:
                    w = _augment_waveform(wav, rng)
                    mel = _vectorized_mel(w, fb, feat_cfg, target_frames)
                    mel = augment_mel(mel, AugConfig(enabled=True), rng)
                    del wav_preload[i]  # free memory as we go
                    src = "decoded"
            mels[i] = mel
            index.append({"i": i, "audio_path": apath,
                          "intent": str(row["intent"]), "src": src})
        except Exception as exc:  # keep going; record the failure
            n_fail += 1
            mels[i] = 0.0
            index.append({"i": i, "audio_path": apath,
                          "intent": str(row["intent"]),
                          "src": f"FAIL:{type(exc).__name__}:{exc}"})

        if (i + 1) % args.progress_every == 0 or i + 1 == n:
            el = time.time() - t0
            cps = (i + 1) / el
            eta = (n - i - 1) / cps if cps > 0 else -1
            print(f"[build] {i + 1}/{n} ({100 * (i + 1) / n:.1f}%) "
                  f"el={el:.0f}s {cps:.1f} clips/s eta={eta:.0f}s "
                  f"rss={_rss_gb():.1f}GB decode={n_decode} fail={n_fail}",
                  flush=True)

    # ---- persist -------------------------------------------------------------
    mels_path = out_dir / "mels.npy"
    np.save(mels_path, mels)
    (out_dir / "index.json").write_text(json.dumps(index))

    meta = {
        "out_tag": args.out_tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(man),
        "manifest_sha256": _sha256(man),
        "n_rows": n,
        "shape": list(mels.shape),
        "dtype": "float32",
        "src_counts": {"cache": n_cache, "decoded": n_decode, "fail": n_fail},
        "deterministic": {
            "seed": args.seed,
            "per_row_rng": "np.random.default_rng(seed * 1000 + row_position)",
            "note": ("same manifest + seed => bit-identical store; scaling runs "
                     "reuse the same seed/recipe so rows stay comparable"),
        },
        "augmentation_recipe": {
            "decoded_clips": ("speed_perturb -> add_noise -> apply_gain -> spec_aug"),
            "cached_clips": "spec_aug only (clean mel from work/mfcc_cache)",
            "speed_perturb": list(AUG_PARAMS["speed"]),
            "noise_prob": AUG_PARAMS["noise_prob"],
            "snr_db": list(AUG_PARAMS["snr_db"]),
            "gain_db": list(AUG_PARAMS["gain_db"]),
            "spec_aug": {"time_mask": 2, "time_width": 20,
                         "freq_mask": 2, "freq_width": 24},
        },
        "mfcc_params": {
            "sample_rate": feat_cfg.sample_rate, "n_fft": feat_cfg.n_fft,
            "hop_length": feat_cfg.hop_length, "n_mels": feat_cfg.n_mels,
            "window_seconds": feat_cfg.window_seconds,
            "f_min": feat_cfg.f_min, "f_max": feat_cfg.f_max,
            "target_frames": target_frames,
        },
        "elapsed_s": round(time.time() - t_start, 1),
        "mels_sha256": _sha256(mels_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    readme = f"""# Precomputed augmented store: {args.out_tag}

Built {meta['created_utc']} by `scripts/build_precomputed_aug.py` (single-process,
deterministic).

- Rows: {n} (1:1 with `{man}`)
- Shape: {tuple(mels.shape)} float32 (~{mels.nbytes / 1e6:.0f} MB)
- Source: cache={n_cache} decoded={n_decode} fail={n_fail}
- Per-row RNG: `default_rng({args.seed} * 1000 + row_position)` — reproducible.

HF `data/vcm/raw/` untouched (read-only). See `meta.json` for the full recipe.
"""
    (out_dir / "README.md").write_text(readme)

    print(f"[build] DONE in {time.time() - t_start:.0f}s -> {out_dir}")
    print(f"[build] mels.npy {mels.shape} {mels.nbytes / 1e6:.0f} MB "
          f"sha256={meta['mels_sha256'][:16]}...")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
