#!/usr/bin/env python3
"""Guarded code-only updater: baseline checks, backups, regression, rollback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def safe(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or relative.startswith("/"):
        raise ValueError(f"Unsafe patch path: {relative}")
    return path


def ensure_stopped(root):
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args=entry.read_bytes().replace(b"\0",b" ").decode(errors="replace")
            if "run_e2e_rpi.py" not in args and "hey_swift.integration.e2e_loop" not in args: continue
            cwd=(entry.parent/"cwd").resolve()
            if cwd.is_relative_to(root) or str(root) in args:
                raise RuntimeError("Stop this installation's E2E runtime before applying or rolling back")
        except (OSError,ValueError): continue


def validate(root, package):
    manifest=json.loads((package/"PATCH_MANIFEST.json").read_text())
    ensure_stopped(root)
    if (root/"V6_PATCH_SHA256SUMS").exists():
        raise FileExistsError("An earlier v6 update receipt exists; reconcile it before applying again")
    for row in manifest["files"]:
        rel=row["path"]
        if not (rel.startswith(("src/hey_swift/","scripts/","docs/","deploy/rpi/")) or rel in
                {"LOCAL_PI_UPDATES.md","RUN_043_RPI_README.md"}):
            raise ValueError(f"Forbidden payload destination: {rel}")
        if not rel.endswith((".py",".md")): raise ValueError("Only code/docs are allowed")
        source=safe(package/"payload",rel)
        if sha(source)!=row["sha256"]: raise ValueError(f"Corrupt payload: {rel}")
        dest=safe(root,rel)
        expected=row["before_sha256"]
        if expected is None:
            if dest.exists(): raise FileExistsError(f"New path already exists: {rel}")
        elif not dest.is_file() or sha(dest)!=expected:
            raise RuntimeError(f"Baseline/local edit conflict: {rel}; preserve it and request reconciliation")
    for rel,digest in manifest["protected"].items():
        if sha(safe(root,rel))!=digest: raise RuntimeError(f"Protected model/config mismatch: {rel}")
    return manifest


def replace_file(source, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,prefix=".swift-update-",delete=False) as handle:
        staged=Path(handle.name)
    shutil.copy2(source,staged)
    os.replace(staged,dest)


def post_validate(root):
    env={**os.environ,"PYTHONPATH":str(root/"src"),"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
    for rel in ("scripts/test_pi_runtime_v6.py","scripts/test_prepi_runtime_v5.py",
                "scripts/test_local_pi_updates.py","tests/test_rpi_slot_translation.py"):
        subprocess.run([sys.executable,"-B",rel],cwd=root,env=env,check=True)
    tag="run_043_hf_grouped_v2b_bc8_sourceft_s42"
    subprocess.run([sys.executable,"-B","scripts/preflight_rpi_bundle.py","--root",str(root),
        "--vcm-onnx",f"models/vcm/{tag}.onnx","--vcm-labels",f"models/vcm/{tag}.labels.json",
        "--vcm-config","configs/vcm_run_043_calibrated.yaml"],cwd=root,env=env,check=True)


def rollback(root, backup):
    ensure_stopped(root)
    records=json.loads((backup/"BACKUP_MANIFEST.json").read_text())
    for row in records["files"]:
        if row["before_sha256"] is None: continue
        current=safe(root,row["path"])
        source=safe(backup/"original",row["path"])
        if sha(source)!=row["before_sha256"]: raise RuntimeError("Backup hash mismatch")
        if not current.is_file() or sha(current) not in {row["sha256"],row["before_sha256"]}:
            raise RuntimeError(f"Rollback refuses subsequent local edits: {row['path']}")
    for row in records["files"]:
        if row["before_sha256"] is not None:
            replace_file(safe(backup/"original",row["path"]),safe(root,row["path"]))
    print("Original overwritten files restored. New files remain unused; nothing deleted.")


def apply(root, package):
    manifest=validate(root,package)
    name=f"runtime-v6_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    backup=root/"deploy/rpi/backups"/name
    backup.mkdir(parents=True,exist_ok=False,mode=0o700)
    chimes={rel:sha(root/rel) if (root/rel).is_file() else None
            for rel in ("audio/wake_chime.wav","audio/wake_chime_original.wav")}
    for row in manifest["files"]:
        if row["before_sha256"] is not None:
            target=safe(backup/"original",row["path"])
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(safe(root,row["path"]),target)
    (backup/"BACKUP_MANIFEST.json").write_text(json.dumps({**manifest,"chimes_before":chimes},indent=2)+"\n")
    print(f"BACKUP: {backup}",flush=True)
    try:
        for row in manifest["files"]:
            replace_file(safe(package/"payload",row["path"]),safe(root,row["path"]))
        post_validate(root)
        for rel,digest in manifest["protected"].items():
            if sha(safe(root,rel))!=digest: raise RuntimeError("Protected artifact changed during validation")
        for rel,digest in chimes.items():
            current=sha(root/rel) if (root/rel).is_file() else None
            if current!=digest: raise RuntimeError("Local chime changed during validation")
        checks="".join(f"{row['sha256']}  {row['path']}\n" for row in manifest["files"])
        checksum_file=root/"V6_PATCH_SHA256SUMS"
        with checksum_file.open("x") as handle: handle.write(checks)
        (backup/"APPLY_RESULT.json").write_text(json.dumps({"status":"tests_and_cpu_preflight_passed",
            "pi_physical_verification":"pending","protected":manifest["protected"],"chimes_unchanged":chimes},indent=2)+"\n")
    except Exception:
        rollback(root,backup)
        raise
    print("APPLIED: selective source update; physical Pi tests remain pending.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--package",default=str(Path(__file__).resolve().parent))
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--apply",action="store_true")
    mode.add_argument("--rollback")
    args=parser.parse_args(); root=Path(args.root).resolve(); package=Path(args.package).resolve()
    if not (root/"src/hey_swift/integration/e2e_loop.py").is_file(): raise RuntimeError("Not an existing Hey Swift installation")
    if args.rollback: rollback(root,Path(args.rollback).resolve())
    elif args.apply: apply(root,package)
    else: validate(root,package); print("CHECK PASS: baseline, payload and protected artifacts; no files changed.")


if __name__=="__main__": main()
