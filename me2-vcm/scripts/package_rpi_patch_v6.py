#!/usr/bin/env python3
"""Build a checksummed selective patch against the verified v5 tar, never models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"work/rpi_transfer_run043_runtime_v5_20261010_04.tar.gz"
BASE_SHA="94d8603e8560bc1aab26c147f4ae5508717bfc15324d2e09a125b2a6cd0733e0"
FILES=[
    "src/hey_swift/integration/e2e_loop.py","src/hey_swift/integration/runtime.py",
    "src/hey_swift/integration/call_guard.py","src/hey_swift/audio/recovery.py",
    "src/hey_swift/audio/cue.py","src/hey_swift/audio/pipewire_capture.py",
    "src/hey_swift/audio/speech.py","src/hey_swift/actions/services/calling.py",
    "src/hey_swift/actions/services/device_actions.py","src/hey_swift/actions/services/music.py",
    "scripts/test_pi_runtime_v6.py","scripts/test_local_pi_updates.py","scripts/diagnose_pi_runtime.py",
    "docs/rpi-runtime-v6.md","docs/command-cheatsheet.md","docs/rpi-e2e.md","docs/rpi-runtime-v5.md",
    "deploy/rpi/README.md","LOCAL_PI_UPDATES.md",
]
PROTECTED=["configs/hardware.yaml","configs/wakeword.yaml","configs/actions.yaml",
    "configs/vcm_run_043_calibrated.yaml","models/wakeword/wakeword_exp15_mined_lp.onnx",
    "models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.onnx",
    "models/vcm/run_043_hf_grouped_v2b_bc8_sourceft_s42.labels.json",
    "integration/assets/voices/en_US-amy-medium.onnx","integration/assets/voices/en_US-amy-medium.onnx.json"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_paths(root, name):
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Name must be one nonempty path component")
    stage = root / "work/rpi_patch_staging" / name
    archive = root / "work" / (name + ".tar.gz")
    checksum = archive.with_name(archive.name + ".sha256")
    final = root / "deploy/rpi" / name
    if stage.exists() or archive.exists() or checksum.exists() or final.exists():
        raise FileExistsError("Use a fresh patch name; no overwriting")
    return stage, archive, checksum, final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",required=True)
    args=parser.parse_args()
    dest, archive, checksum, final = output_paths(ROOT, args.name)
    if digest(BASE)!=BASE_SHA: raise RuntimeError("Verified base archive hash mismatch")
    subprocess.run([sys.executable,"-B","scripts/test_pi_runtime_v6.py"],cwd=ROOT,check=True)
    with tarfile.open(BASE) as base:
        members={m.name.split("/",1)[1]:m for m in base.getmembers() if m.isfile()}
        before={rel:hashlib.sha256(base.extractfile(member).read()).hexdigest() for rel,member in members.items()}
    dest.mkdir(parents=True)
    rows=[]
    copies={rel:rel for rel in FILES}
    copies["RUN_043_RPI_README.md"]="docs/rpi-runtime-v6.md"
    for target,source in copies.items():
        output=dest/"payload"/target; output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,output)
        rows.append({"path":target,"sha256":digest(output),"before_sha256":before.get(target)})
    shutil.copy2(ROOT/"scripts/apply_rpi_runtime_patch.py",dest/"apply.py")
    shutil.copy2(ROOT/"docs/rpi-runtime-v6.md",dest/"README.md")
    manifest={"base_tar_sha256":BASE_SHA,"base_version":"pi-flow-v5-safe-capture-offline-20261010",
              "version":"pi-flow-v6-selective-recovery-20261010","files":rows,
              "protected":{rel:before[rel] for rel in PROTECTED},
              "excluded":"Models/configs/chimes/music/voice/account state/venv/training artifacts",
              "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    (dest/"PATCH_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (dest/"CHANGED_FILES.txt").write_text("\n".join(row["path"] for row in rows)+"\n")
    sums="".join(f"{digest(path)}  {path.relative_to(dest).as_posix()}\n"
                  for path in sorted(dest.rglob("*")) if path.is_file())
    (dest/"SHA256SUMS").write_text(sums)
    staged_archive = dest.parent / archive.name
    staged_checksum = dest.parent / checksum.name
    if staged_archive.exists() or staged_checksum.exists():
        raise FileExistsError("Staging archive already exists; use a fresh name")
    with tarfile.open(staged_archive,"w:gz") as tar: tar.add(dest,arcname=args.name)
    sha=digest(staged_archive)
    staged_checksum.write_text(f"{sha}  {archive.name}\n")
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Transfer archives stay in work; only the complete unpacked release is deployed.
    os.link(staged_archive, archive)
    os.link(staged_checksum, checksum)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError("Deployment directory already exists; refusing replacement")
    dest.rename(final)
    print(json.dumps({"deployment":str(final),"archive":str(archive),"sha256":sha,
                      "bytes":archive.stat().st_size,"changed_files":len(rows)},indent=2))


if __name__=="__main__": main()
