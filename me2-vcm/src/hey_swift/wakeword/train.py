"""Reproducible wake-word training (faithful exp15 recipe).

Faithful port of ``me2-wakeword-swift/scripts/train_v5.py`` (the exp15 trainer).
This is a REPRODUCIBILITY run: it re-executes the historical exp15 recipe from
the canonical repository against the canonical dataset snapshot. It is NOT an
optimization experiment.

Recipe (frozen from historical exp15 ``config.json``):
    model    WakeGRU 2-class, 204,738 params, input (B, 80, 100)
    seed     20260926
    opt      AdamW(lr=3e-4, weight_decay=1e-4)
    sched    cosine annealing to floor 0.0 over `epochs`
    loss     cross_entropy (wake_weight=1.0, label_smoothing=0.0)
    batch    32 ; epochs 60 ; early-stop patience 20
    ckpt     best val = max(recall, f1) among thresholds with FA <= fa_cap (0.05)
    aug      gain[-12,6]dB, noise SNR[8,30]dB (white/pink/brown/babble/city),
             RIR RT60[0.15,0.6]; then prob 0.85; LP 0.45x[3,5]kHz;
             SNR-ext 0.35x[0,10]dB; VTLP 0.3; SpecAug 0.5 (t35,f20,n2); shift 12

Outputs are immutable under ``experiments/wakeword/<tag>/``. Smoke/debug runs
belong under ``work/`` (see the artifact lifecycle in docs/).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import dataset as ds
from .features import (N_MELS, SAMPLE_RATE, TARGET_FRAMES,
                       load_wav_mono_16k, mel_from_pcm)
from .model import WakeArchSpec, build_model, count_params

EXP_ROOT = Path("experiments/wakeword")

# --- Historical exp15 defaults (override via config 'training:' block) ------
DEFAULTS = dict(
    seed=20260926, lr=3e-4, epochs=60, patience=20, batch=32, wd=1e-4,
    cosine_floor=0.0, wake_weight=1.0, label_smooth=0.0, fa_cap=0.05,
    workers=4,
    # augmentation
    aug_prob=0.85, snr_ext_prob=0.35, snr_ext_lo=0.0, snr_ext_hi=10.0,
    lp_prob=0.45, lp_lo=3000.0, lp_hi=5000.0, vtlp_prob=0.3, spec_prob=0.5,
    shift_frames=12,
    # audio-aug ranges (from historical config.json)
    gain_db=(-12.0, 6.0), snr_db=(8.0, 30.0), rt60=(0.15, 0.6),
    noise_cats=("white", "pink", "brown", "babble", "city"),
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# audio augmentation (faithful to train_swift2.py)
# --------------------------------------------------------------------------- #
def _pink(n: int, rng) -> np.ndarray:
    white = rng.standard_normal(n)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    a0, a1, a2, a3, a4, a5, a6 = 0.0104, 0.0052, 0.0003, 0.0001, 0.0001, 0.0001, 0.0001
    out = np.zeros(n)
    for i in range(n):
        w = white[i]
        b0 = 0.99886 * b0 + w * a0
        b1 = 0.99332 * b1 + w * a1
        b2 = 0.96900 * b2 + w * a2
        b3 = 0.86650 * b3 + w * a3
        b4 = 0.55000 * b4 + w * a4
        b5 = -0.7616 * b5 - w * a5
        b6 = -0.0169 * b6 - w * a6
        out[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + 0.5362 * w
    return out


def _make_noise(n: int, kind: str, rng) -> np.ndarray:
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = _pink(n, rng)
    elif kind == "brown":
        x = np.cumsum(rng.standard_normal(n))
        x -= x.mean()
    else:  # babble/city proxy: broadband + transients
        x = rng.standard_normal(n) * 0.8 + _pink(n, rng) * 0.4
        for _ in range(int(rng.integers(1, 4))):
            pos = int(rng.integers(0, max(1, n - 200)))
            dur = int(rng.integers(30, 200))
            env = np.exp(-np.arange(dur) / (dur / 4.0))
            x[pos:pos + dur] += rng.uniform(0.5, 1.5) * env * rng.standard_normal(dur)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def _make_rir(n: int, rt60: float, rng) -> np.ndarray:
    decay = np.exp(-np.log(10) * np.arange(n) / (rt60 * SAMPLE_RATE))
    ir = rng.standard_normal(n) * decay
    ne = int(0.05 * SAMPLE_RATE)
    ir[:ne] += rng.standard_normal(ne) * np.exp(-np.arange(ne) / (0.02 * SAMPLE_RATE)) * 0.6
    return ir / (np.sqrt(np.mean(ir ** 2)) + 1e-12)


def audio_augment(pcm: np.ndarray, rng, prob: float = 0.85,
                  gain_db=(-12.0, 6.0), snr_db=(8.0, 30.0),
                  rt60=(0.15, 0.6), noise_cats=("white", "pink", "brown")) -> np.ndarray:
    """noise + reverb + gain on a 16 kHz mono clip (returns float32)."""
    if rng.random() > prob:
        return pcm
    g0, g1 = gain_db
    pcm = pcm * (10 ** (rng.uniform(g0, g1) / 20.0))
    if rng.random() < 0.85:
        snr = float(rng.uniform(*snr_db))
        kind = rng.choice(list(noise_cats))
        noise = _make_noise(len(pcm), kind, rng)
        sp = np.sqrt(np.mean(pcm ** 2) + 1e-12)
        tgt = sp / (10 ** (snr / 20.0))
        pcm = pcm + noise * (tgt / (np.sqrt(np.mean(noise ** 2)) + 1e-12))
    if rng.random() < 0.80:
        rt = float(rng.uniform(*rt60))
        ir = _make_rir(int(0.3 * SAMPLE_RATE), rt, rng)
        from scipy.signal import fftconvolve
        pcm = fftconvolve(pcm, ir)[:len(pcm)]
    peak = np.abs(pcm).max()
    if peak > 1.0:
        pcm = pcm / peak
    return pcm.astype(np.float32)


def lowpass_butter(pcm: np.ndarray, cutoff_hz: float, sr: int = SAMPLE_RATE,
                   order: int = 2) -> np.ndarray:
    """2nd-order Butterworth low-pass (consumer-mic simulation)."""
    from scipy.signal import butter, lfilter
    nyq = 0.5 * sr
    b, a = butter(order, min(cutoff_hz, nyq * 0.99) / nyq, btype="low")
    return lfilter(b, a, pcm)


# --------------------------------------------------------------------------- #
# PyTorch dataset + augmentation (faithful to train_v5.WakeDS)
# --------------------------------------------------------------------------- #
def _build_torch_dataset(items, augment, spec, vtlp_prob, shift_frames,
                         aug_prob, snr_ext_prob, snr_ext_range,
                         lp_prob, lp_range, seed):
    import torch
    from torch.utils.data import Dataset

    class WakeDS(Dataset):
        def __init__(self):
            self.items = items
            self.augment = augment
            self.spec = spec
            self.vtlp_prob = vtlp_prob
            self.shift_frames = shift_frames
            self.aug_prob = aug_prob
            self.snr_ext_prob = snr_ext_prob
            self.snr_ext_range = snr_ext_range
            self.lp_prob = lp_prob
            self.lp_range = lp_range
            self._cache = {}
            self._rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.items)

        def _feat(self, path):
            if path not in self._cache:
                pcm = load_wav_mono_16k(path)
                f = mel_from_pcm(np.asarray(pcm, dtype=np.float32))
                self._cache[path] = torch.from_numpy(f).squeeze(0).float()
            return self._cache[path]

        def _pcm(self, path):
            if path not in self._cache:
                self._cache[path] = load_wav_mono_16k(path)
            return self._cache[path]

        def __getitem__(self, i):
            path, label = self.items[i]
            if self.augment:
                pcm = np.array(self._pcm(path), dtype=np.float32, copy=True)
                if self._rng.random() < self.aug_prob:
                    pcm = audio_augment(pcm, self._rng, prob=1.0,
                                        gain_db=GLOBAL_GAIN, snr_db=GLOBAL_SNR,
                                        rt60=GLOBAL_RT60, noise_cats=GLOBAL_NOISE)
                    if self.lp_prob > 0 and self._rng.random() < self.lp_prob:
                        cut = float(self._rng.uniform(*self.lp_range))
                        pcm = lowpass_butter(pcm, cut, sr=SAMPLE_RATE)
                    if self._rng.random() < self.snr_ext_prob:
                        kind = self._rng.choice(list(GLOBAL_NOISE))
                        noise = _make_noise(len(pcm), kind, self._rng)
                        snr = float(self._rng.uniform(*self.snr_ext_range))
                        sp = np.sqrt(np.mean(pcm ** 2) + 1e-12)
                        tgt = sp / (10 ** (snr / 20.0))
                        pcm = pcm + noise * (tgt / (np.sqrt(np.mean(noise ** 2)) + 1e-12))
                        peak = np.abs(pcm).max()
                        if peak > 1.0:
                            pcm = pcm / peak
                f = mel_from_pcm(np.asarray(pcm, dtype=np.float32))
                x = torch.from_numpy(f).squeeze(0).float()
                # time-shift roll
                sh = int(self._rng.integers(-self.shift_frames, self.shift_frames + 1))
                if sh:
                    x = torch.roll(x, shifts=sh, dims=1)
                if self.vtlp_prob > 0 and self._rng.random() < self.vtlp_prob:
                    factor = float(self._rng.uniform(0.9, 1.1))
                    x = _vtlp(x, factor)
                if self.spec is not None:
                    x = self.spec(x)
            else:
                x = self._feat(path)
            return x, torch.tensor(label, dtype=torch.long)

    return WakeDS()


GLOBAL_GAIN = GLOBAL_SNR = GLOBAL_RT60 = GLOBAL_NOISE = None  # set in run()


def _vtlp(x, factor):
    import torch
    if abs(factor - 1.0) < 1e-3:
        return x
    F_ = x.size(1)
    M = int(round(F_ * factor))
    grid = torch.linspace(0, F_ - 1, M, device=x.device)
    idx0 = grid.long().clamp(0, F_ - 1)
    idx1 = (idx0 + 1).clamp(0, F_ - 1)
    frac = (grid - idx0.float()).unsqueeze(0)
    y = x[:, idx0] * (1 - frac) + x[:, idx1] * frac
    if M != F_:
        g2 = torch.linspace(0, M - 1, F_, device=x.device)
        j0 = g2.long().clamp(0, M - 1)
        j1 = (j0 + 1).clamp(0, M - 1)
        f2 = (g2 - j0.float()).unsqueeze(0)
        y = y[:, j0] * (1 - f2) + y[:, j1] * f2
    return y


class SpecAugment:
    def __init__(self, apply_prob=0.7, t_mask=35, f_mask=20, n_masks=2):
        self.apply_prob, self.t_mask, self.f_mask, self.n_masks = (
            apply_prob, t_mask, f_mask, n_masks)

    def __call__(self, x):
        import torch
        if torch.rand(1).item() > self.apply_prob:
            return x
        for _ in range(self.n_masks):
            w = int(torch.randint(1, self.t_mask + 1, (1,)).item())
            s = int(torch.randint(0, max(x.size(1) - w, 1), (1,)).item())
            x = x.clone()
            x[:, s:s + w] = 0.0
            w = int(torch.randint(1, self.f_mask + 1, (1,)).item())
            s = int(torch.randint(0, max(x.size(0) - w, 1), (1,)).item())
            x[s:s + w, :] = 0.0
        return x


# --------------------------------------------------------------------------- #
# evaluation helpers (faithful to train_v5)
# --------------------------------------------------------------------------- #
def probs_of(model, dl, device):
    import torch
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        x = x.to(device)
        with torch.no_grad():
            p = torch.softmax(model(x), dim=1)[:, 0].cpu().numpy()
        ps.append(p)
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def sweep(p, y, lo=0.01, hi=0.99, step=0.01):
    """Full threshold sweep. y==0 is wake."""
    out = []
    th = lo
    while th <= hi + 1e-9:
        pred = p > th
        isw = y == 0
        tp = int((pred & isw).sum())
        fp = int((pred & ~isw).sum())
        fn = int((~pred & isw).sum())
        tn = int((~pred & ~isw).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fa = fp / (fp + tn) if (fp + tn) else 0.0
        fr = fn / (fn + tp) if (fn + tp) else 0.0
        out.append(dict(thr=round(float(th), 2), precision=prec, recall=rec,
                        f1=f1, fa=fa, fr=fr, tp=tp, fp=fp, fn=fn, tn=tn))
        th += step
    return out


def temporal_metrics(p, y, thr=0.5, consec=2):
    """Sliding proxy: sort wake clips by index, require `consec` consecutive
    above threshold to count a fire."""
    isw = y == 0
    idx = np.where(isw)[0]
    fired = 0
    run = 0
    for i in idx:
        if p[i] >= thr:
            run += 1
            if run >= consec:
                fired += 1
                run = 0
        else:
            run = 0
    total = int(isw.sum())
    return {"temporal_fires": fired,
            "temporal_recall_proxy": fired / total if total else 0.0,
            "thr": thr, "consec": consec}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(tag: str, config: str = "configs/wakeword.yaml",
        seed: Optional[int] = None, device: Optional[str] = None,
        out_root: Optional[str] = None) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from collections import Counter

    global GLOBAL_GAIN, GLOBAL_SNR, GLOBAL_RT60, GLOBAL_NOISE
    cfg = yaml.safe_load(open(config, encoding="utf-8")) or {}
    tr = cfg.get("training", {}) or {}
    p = lambda k, d: tr.get(k, DEFAULTS[k])

    seed = int(seed if seed is not None else p("seed", DEFAULTS["seed"]))
    lr = float(p("lr", DEFAULTS["lr"]))
    epochs = int(p("epochs", DEFAULTS["epochs"]))
    patience = int(p("patience", DEFAULTS["patience"]))
    batch = int(p("batch", DEFAULTS["batch"]))
    wd = float(p("wd", DEFAULTS["wd"]))
    cosine_floor = float(p("cosine_floor", DEFAULTS["cosine_floor"]))
    wake_weight = float(p("wake_weight", DEFAULTS["wake_weight"]))
    label_smooth = float(p("label_smooth", DEFAULTS["label_smooth"]))
    fa_cap = float(p("fa_cap", DEFAULTS["fa_cap"]))
    workers = int(p("workers", DEFAULTS["workers"]))
    aug_prob = float(p("aug_prob", DEFAULTS["aug_prob"]))
    snr_ext_prob = float(p("snr_ext_prob", DEFAULTS["snr_ext_prob"]))
    snr_ext_range = (float(p("snr_ext_lo", 0.0)), float(p("snr_ext_hi", 10.0)))
    lp_prob = float(p("lp_prob", DEFAULTS["lp_prob"]))
    lp_range = (float(p("lp_lo", 3000.0)), float(p("lp_hi", 5000.0)))
    vtlp_prob = float(p("vtlp_prob", DEFAULTS["vtlp_prob"]))
    spec_prob = float(p("spec_prob", DEFAULTS["spec_prob"]))
    shift_frames = int(p("shift_frames", DEFAULTS["shift_frames"]))
    GLOBAL_GAIN = tuple(p("gain_db", DEFAULTS["gain_db"]))
    GLOBAL_SNR = tuple(p("snr_db", DEFAULTS["snr_db"]))
    GLOBAL_RT60 = tuple(p("rt60", DEFAULTS["rt60"]))
    GLOBAL_NOISE = tuple(p("noise_cats", DEFAULTS["noise_cats"]))

    set_seed(seed)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    out_dir = Path(out_root or EXP_ROOT) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    log("=" * 72)
    log(f"Hey Swift wake-word reproducibility training  tag={tag}")
    log(f"device     : {device}")
    if device.type == "cuda":
        log(f"gpu        : {torch.cuda.get_device_name(0)}")
    log(f"seed       : {seed}")
    log(f"cfg        : lr={lr} bs={batch} wd={wd} wake_w={wake_weight} "
        f"ls={label_smooth} vtlp={vtlp_prob} spec={spec_prob} shift={shift_frames} "
        f"cosine_floor={cosine_floor} fa_cap={fa_cap} "
        f"AUGPROB={aug_prob} SNREXT={snr_ext_prob}x{snr_ext_range} "
        f"LPPROB={lp_prob}x{lp_range}Hz")

    tr_items = ds.collect_split("train")
    va_items = ds.collect_split("validation")
    te_items = ds.collect_split("test")
    trc = Counter(l for _, l in tr_items)
    vac = Counter(l for _, l in va_items)
    tec = Counter(l for _, l in te_items)
    log(f"train  : wake={trc[0]}  not-wake={trc[1]}  total={len(tr_items)}")
    log(f"valid  : wake={vac[0]}  not-wake={vac[1]}  total={len(va_items)}")
    log(f"test   : wake={tec[0]}  not-wake={tec[1]}  total={len(te_items)}")

    spec = SpecAugment(apply_prob=spec_prob, t_mask=35, f_mask=20, n_masks=2)
    ds_tr = _build_torch_dataset(tr_items, True, spec, vtlp_prob, shift_frames,
                                 aug_prob, snr_ext_prob, snr_ext_range,
                                 lp_prob, lp_range, seed)
    ds_va = _build_torch_dataset(va_items, False, None, 0.0, 0, 0.0, 0.0,
                                 (0, 1), 0.0, (0, 0), seed)
    ds_te = _build_torch_dataset(te_items, False, None, 0.0, 0, 0.0, 0.0,
                                 (0, 1), 0.0, (0, 0), seed)
    kw = dict(num_workers=workers, pin_memory=(device.type == "cuda"))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, **kw)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, **kw)

    model = build_model(WakeArchSpec()).to(device)
    nparams = count_params(model)
    log(f"model    : WakeGRU n_classes=2  params={nparams}  size~{nparams * 4 / 1e6:.2f} MB fp32")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    def lr_lambda(ep):
        prog = ep / max(epochs, 1)
        return cosine_floor + (1 - cosine_floor) * 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    alpha_t = torch.tensor([wake_weight, 1.0]).to(device) if wake_weight != 1.0 else None

    def compute_loss(out, y):
        return nn.functional.cross_entropy(out, y, weight=alpha_t,
                                           label_smoothing=label_smooth)

    best_score = -1
    best_thr = 0.5
    bad = 0
    history = []
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        tloss = 0.0
        t0 = time.time()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = compute_loss(model(x), y)
            loss.backward()
            opt.step()
            tloss += loss.item()
        sched.step()
        vp, vy = probs_of(model, dl_va, device)
        sv = sweep(vp, vy)
        cand = [m for m in sv if m["fa"] <= fa_cap] or sv
        bestv = max(cand, key=lambda m: (m["recall"], m["f1"]))
        tm = temporal_metrics(vp, vy, thr=bestv["thr"])
        trp, try_ = probs_of(model, dl_tr, device)
        tr_pos = trp[try_ == 0]
        va_pos = vp[vy == 0]
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        log(f"ep{ep + 1:02d} tloss={tloss / len(dl_tr):.4f} | "
            f"VAL thr={bestv['thr']:.2f} P={bestv['precision']:.3f} R={bestv['recall']:.3f} "
            f"F1={bestv['f1']:.3f} FA={bestv['fa'] * 100:.2f}% FR={bestv['fr'] * 100:.1f}% | "
            f"TRpos={tr_pos.mean():.3f} VApos={va_pos.mean():.3f} | "
            f"tempR={tm['temporal_recall_proxy']:.3f} | lr={lr_now:.2e} {dt:.0f}s")
        history.append(dict(epoch=ep + 1, tloss=float(tloss / len(dl_tr)),
                            **{k: float(v) for k, v in bestv.items()},
                            temporal={k: float(v) for k, v in tm.items()},
                            tr_pos=float(tr_pos.mean()), va_pos=float(va_pos.mean())))
        score = (bestv["recall"], bestv["f1"])
        star = ""
        if best_score == -1 or score > best_score:
            best_score = score
            best_thr = bestv["thr"]
            bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=1))
            star = " *"
        else:
            bad += 1
            if bad >= patience:
                log(f"EARLY STOP at ep{ep + 1} (no improvement {bad})")
                break
        if star:
            log(f"   -> new best (score={score[0]:.3f},{score[1]:.3f}){star}")

    # ---------------- final eval: val + test full sweep ----------------
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    final = {}
    for name, dl in (("val", dl_va), ("test", dl_te)):
        p, y = probs_of(model, dl, device)
        sv = sweep(p, y)
        cand = [m for m in sv if m["fa"] <= fa_cap] or sv
        best = max(cand, key=lambda m: (m["recall"], m["f1"]))
        tm = temporal_metrics(p, y, thr=best["thr"])
        final[name] = dict(best=best, temporal=tm, sweep=sv)
        log(f"\n[{name}] BEST thr={best['thr']:.2f} P={best['precision']:.3f} "
            f"R={best['recall']:.3f} F1={best['f1']:.3f} FA={best['fa'] * 100:.2f}% "
            f"FR={best['fr'] * 100:.1f}%  (tp{best['tp']} fp{best['fp']} fn{best['fn']} tn{best['tn']})")
    (out_dir / "final_eval.json").write_text(json.dumps(final, indent=1))

    cfg_out = dict(exp=tag, seed=seed, device=str(device),
                   epochs_run=len(history), best_epoch=history[-1]["epoch"] if history else 0,
                   n_params=nparams,
                   train_classes=dict(trc), val_classes=dict(vac), test_classes=dict(tec),
                   elapsed_s=round(time.time() - t_start, 1),
                   recipe_source="me2-wakeword-swift/scripts/train_v5.py (exp15)")
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=1))
    (out_dir / "train.log").write_text("\n".join(log_lines) + "\n")
    log(f"\nDONE {tag}: {len(history)} epochs, {cfg_out['elapsed_s']}s, best_thr={best_thr}")
    return out_dir
