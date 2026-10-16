# macOS Companion

The recognizer runs on Pi 5; macOS can fetch frozen results or operate the laptop
acoustic benchmark. These commands are documented, not physically macOS-tested.

```bash
git clone --filter=blob:none --sparse https://github.com/ercallagon/ai231.git
cd ai231
git sparse-checkout set me2-vcm
cd me2-vcm
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-eval.txt
.venv/bin/python scripts/reproduce.py reproduce --fetch --output work/reproduction_unique
ssh ej@192.168.100.103
```

For acoustic testing, use a separate checkout of
`https://github.com/airimonda/vcm-benchmark`, its requirements and microphone
permissions for Terminal. Use `.venv/bin/python benchmark.py` and preserve its
revision/settings/results. Select the actual output speaker, not a headset's
hands-free profile. Follow [FINAL_PI_VALIDATION.md](FINAL_PI_VALIDATION.md);
never replace the speaker-to-mic run with injection under the same metric name.
`scp -r PATH ej@192.168.100.103:~/swift-clean/me2-vcm/work/UNIQUE_NAME`
transfers evidence; it does not install or modify private setup.

Use Python 3.11-3.13 (prefer 3.13); Python 3.14 is outside this project's declared
support. If `python3` is 3.14, explicitly use an installed `python3.13` for venv.
