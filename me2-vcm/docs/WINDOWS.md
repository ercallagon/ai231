# Windows reproduction and transfer

Use the existing project folder; commands below create only a Git checkout and
its environment. The working Pi installation is not replaced automatically.

PowerShell:

```powershell
git clone --filter=blob:none --sparse https://github.com/ercallagon/ai231.git ai231
Set-Location ai231
git sparse-checkout set me2-vcm
Set-Location me2-vcm
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe scripts/reproduce.py verify
.\.venv\Scripts\python.exe scripts/reproduce.py reproduce --fetch --output work\windows_reproduction_unique
```

If ai231 already exists, inspect that checkout and update it without overwriting
local edits. Reproduction downloads the pinned public data, verifies hashes and
never trains or fits thresholds. The test/holdout evaluation is previously exposed.
Do not commit fetched data or result caches.

For Pi testing, update the existing checkout using
[FINAL_PI_VALIDATION.md](FINAL_PI_VALIDATION.md), or install a separate one with
[QUICKSTART_PI.md](QUICKSTART_PI.md). Do not create a duplicate installation just
to update. GitHub authentication belongs in Git Credential Manager/browser,
never in a URL, committed configuration or chat. Dependency/data downloads
need internet initially; after caching, a shared hotspot/LAN is sufficient for
SSH and the acoustic benchmark. See BENCHMARK.md for the separate laptop tool.
The class benchmark has its own Windows environment and recordings; Pi commands
using `.venv/bin/python` must be entered in the Pi SSH terminal, not PowerShell.
