# Windows reproduction and transfer

Use the existing project folder; commands below create only a Git checkout and
its environment. The working Pi installation is not replaced automatically.

PowerShell:

```powershell
Set-Location 'C:\Users\EJ\Documents\ChatGPT\ME2-VCM 2'
git clone https://github.com/ercallagon/ai231.git ai231
Set-Location ai231\me2-vcm
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv\Scripts\python.exe scripts/reproduce.py verify
.\.venv\Scripts\python.exe scripts/reproduce.py reproduce --fetch --output work\windows_reproduction_unique
```

If ai231 already exists, inspect that checkout and update it without overwriting
local edits. Reproduction downloads the pinned public data, verifies hashes and
never trains or fits thresholds. The test/holdout evaluation is previously exposed.
Do not commit fetched data or result caches.

If HPC cannot push because GitHub authentication is unavailable, download the
agent-provided Git bundle under work/finalize_submit/ instead. It contains the
preserved repository history and the initial ME2 commit, with no credentials.
For a new checkout:

```powershell
git clone .\ME2_submission_ready.bundle ai231
Set-Location ai231
git remote set-url origin https://github.com/ercallagon/ai231.git
git fetch origin
git log --oneline origin/main..main
git diff --name-only origin/main...main
git push -u origin main
```

Verify that the difference adds only me2-vcm/. GitHub authentication belongs in
Git Credential Manager/browser, never in a URL, config committed to Git, or chat.
If the remote has advanced, stop on a non-fast-forward error; integrate that
history normally and never force-push over ME1.

For Pi testing, clone/update the published repository on Pi as in PI5.md. If
using the bundle offline, transfer it with your existing SSH connection, clone
it on Pi to ~/ai231, and follow the same instructions. Dependency/data downloads
need internet initially; after caching, a shared hotspot/LAN is sufficient for
SSH and the acoustic benchmark. See BENCHMARK.md for the separate laptop tool.
