# Raspberry Pi 5 Deployment

> Historical research guide. Current frozen defaults, installation, rollback and
> reproduction commands are in README.md, docs/PI5.md and docs/BENCHMARK.md.
> Old artifact tags, package paths and readiness snapshots below are not current
> release instructions. Physical Pi 5 verification of the six-second profile is pending.

Standalone, CPU-only deployment of Hey Swift on a Raspberry Pi 5.

This page describes the older `deploy/rpi/` launcher. For the current complete
wake-word -> VCM -> action loop and preflighted bundle, use
[`docs/rpi-e2e.md`](rpi-e2e.md); that guide is authoritative for current runtime
dependencies and commands.

## What ships to the Pi

The Pi bundle is **lean**: inference code + small ONNX models + config. No torch,
no training data, no large checkpoints.

```
deploy/rpi/
├── README.md
├── hey-swift.service          # systemd auto-start unit
├── labels.json                # intent order + slot sets + τ (from export)
├── thresholds.json            # wake gates (threshold/sustain/cooldown/RMS)
├── models/
│   ├── wakeword.onnx          # ~204k params, NumPy-inference
│   └── vcm_a0_scratch.onnx    # DS-CNN A0, (1,80,197) -> 20 logits
└── run_live.py                # thin launcher (imports hey_swift.app)
```

## Prepare the bundle (on the dev machine)

```bash
# 1. Train + export (produces models/vcm/a0_scratch.onnx + labels.json)
./run.sh

# 2. Assemble the Pi bundle
python scripts/05_stage_rpi.py --tag a0_scratch   # copies ONNX + labels + config
```

## Install on the Pi

```bash
# 1. Copy the bundle
scp -r deploy/rpi/ pi@raspberrypi:~/hey-swift

# 2. System deps
ssh pi@raspberrypi
sudo apt update
sudo apt install -y python3-venv python3-lgpio libportaudio2 libsndfile1

# 3. Install the current inference runtime in a venv. The system-site setting
#    exposes apt's lgpio module; binary-only prevents accidental source builds.
cd ~/hey-swift
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-pi.txt

# 4. Run once manually
python3 -m hey_swift.cli run

# 5. Enable auto-start
sudo cp hey-swift.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hey-swift
journalctl -u hey-swift -f
```

## Verify

```bash
systemctl status hey-swift
# say "Hey Swift" then a command; watch journalctl for the routed action
```

## Resource notes (CPU-only)

- Both models are small (≤ ~205k params); inference is a few ms per 0.1 s hop
  on a Pi 5 (Cortex-A76). No CUDA, no swap pressure.
- The feature paths are pure NumPy; ONNX Runtime uses the CPU EP.
- RAM footprint is dominated by the OS + Python; expect < 300 MB for the app.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `lgpio not available` | venv cannot see apt module | recreate the venv with `--system-site-packages`; install `python3-lgpio` |
| No mic input | wrong device / no portaudio | check `audio_io.capture_device`; install `libportaudio2` |
| Wake never fires | threshold too high / RMS gate | lower `threshold` or `rms_floor_db` in `configs/wakeword.yaml` |
| Double-fires | cooldown too short | increase `cooldown_seconds` |
| I2C LCD not found | I2C disabled | `sudo raspi-config` → enable I2C |
