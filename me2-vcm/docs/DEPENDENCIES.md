# Pi locks and uv evaluation

Default installation uses Debian 13 ARM64 /usr/bin/python3 3.13, a new .venv
with --system-site-packages, and pip --require-hashes --only-binary=:all:.
Existing environments are refused. OS python3-lgpio stays apt-owned and is
intentionally absent from the Python locks. Installation and doctor actually
import lgpio; successful resolution alone is not ABI validation.

Committed locks:

- requirements-pi-py313-aarch64.lock: inference, audio/GPIO/DBus and evaluation.
- requirements-speech-py313-aarch64.lock: same stack plus optional Piper.

Both were resolved with uv 0.13.1, Python 3.13,
aarch64-unknown-linux-gnu, hashes and binary-only resolution. All locked
packages were downloaded using pip ARM64/CPython-3.13/abi3 platform tags and
hash verification; no source builds, voice downloads or target execution.
This verifies wheels, not Pi imports. LOCK_AUDIT.json records scope.

Recreate deliberately, never during scored evaluation:

```bash
uv pip compile requirements-pi.txt requirements-eval.txt --python-version 3.13 --python-platform aarch64-unknown-linux-gnu --generate-hashes --only-binary :all: --index-url https://pypi.org/simple --output-file requirements-pi-py313-aarch64.lock
uv pip compile requirements-speech.txt --python-version 3.13 --python-platform aarch64-unknown-linux-gnu --generate-hashes --only-binary :all: --index-url https://pypi.org/simple --output-file requirements-speech-py313-aarch64.lock
```

References: https://docs.astral.sh/uv/reference/platforms/ and
https://docs.astral.sh/uv/pip/compile/. uv venv supports system-site packages;
--no-managed-python prevents substituting an interpreter that cannot import
Debian's lgpio ABI. The host flag probe used /usr/bin/python3 3.12.3. A fresh
Pi 5 checkout was installed and `./swift doctor` verified Python 3.13
core/audio/ONNX/GPIO imports plus the OS-provided lgpio ABI. The optional uv
installer path was not separately exercised on Pi; doctor does not prove GPIO
actuation or optional Spotify, calling or messaging behavior.

**Decision:** commit hashed requirements locks and retain pip as the default;
the Pi 5 pip installation/import checks passed. uv remains optional via
`swift.py install --uv` when uv is
on PATH, in a separate new checkout/environment. Do not run uv sync over all
training extras: scoped requirements locks cover only Pi inference/evaluation.
A root uv.lock is unnecessary for this scoped installer. Record tool versions;
do not claim pip/uv target equivalence until both paths pass on the Pi.

Optional uv installation can live in an isolated tool environment:

```bash
/usr/bin/python3 -m venv work/uv-tools
work/uv-tools/bin/python -m pip install uv==0.13.1
PATH="$PWD/work/uv-tools/bin:$PATH" ./swift install --uv
```

Use this only in a separate checkout whose .venv does not exist. OS audio,
Bluetooth and GPIO permissions remain shared. No voices or songs are installed
automatically. Detectors retain their own scratch-trained weights.
