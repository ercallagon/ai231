"""Pinned metadata and verified embedded audio; no model fitting."""
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
from .slot_labels import _slot_label
from .taxonomy import load_taxonomy

ROOT=Path(__file__).resolve().parents[3]
REVISION="bfca0bbebfe17e00d6a685d1da573615d29ecdce"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def tax():
    return load_taxonomy(ROOT/"configs/vcm_selected.yaml")


def labels(rows,split):
    overlay=json.loads((ROOT/"data/unsupported_overlay.json").read_text()).get(split,{})
    y,ys,invalid=[],[],[]
    for r in rows:
        name=r["intent"]
        value=_slot_label(name,r.get("slot") or None,tax())
        bad=r["audio_path"] in overlay
        if bad and name=="TEMPERATURE":
            if value not in (None,3):
                raise ValueError("Conflicting unsupported-temperature target")
            value=3
        y.append(tax().oos_idx if name==tax().oos_class else tax().intent_to_idx[name])
        ys.append(-1 if value is None else value)
        invalid.append(bad)
    return np.array(y),np.array(ys),np.array(invalid)


def manifest(split):
    if split not in ("train","val","tune_oos","test","holdout","challenge"):
        raise ValueError("Unknown frozen split")
    path=ROOT/"data/manifests"/(split+".csv")
    expected=json.loads((ROOT/"data/PROVENANCE.json").read_text())["manifests"][split]["public_metadata_sha256"]
    if sha(path)!=expected:
        raise ValueError("Manifest hash changed")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def catalog():
    spec=importlib.util.spec_from_file_location("pinned_catalog",ROOT/"scripts/01_download_hf.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return {p:(size,digest) for p,size,digest in module.CORE_FILES+module.FULL_EXTRA_FILES}


def relative_parquet(row):
    prefix="data/vcm/raw/"
    if not row["parquet"].startswith(prefix):
        raise ValueError("Unexpected Parquet location")
    path=Path(row["parquet"][len(prefix):])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe dataset path")
    return path


def ensure_files(rows,data_root,fetch=False):
    files=catalog()
    for relative in sorted({relative_parquet(r) for r in rows}):
        size,digest=files[relative.as_posix()]
        path=Path(data_root)/relative
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(str(path)+"; --fetch downloads the pinned data")
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(".download")
            url="https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/resolve/"+REVISION+"/"+relative.as_posix()
            with urllib.request.urlopen(url,timeout=120) as source,temporary.open("xb") as target:
                for block in iter(lambda:source.read(1024*1024),b""):
                    target.write(block)
            if temporary.stat().st_size!=size or sha(temporary)!=digest:
                raise ValueError("Download hash mismatch; failed download preserved")
            temporary.rename(path)
        if path.stat().st_size!=size or sha(path)!=digest:
            raise ValueError("Dataset size/hash mismatch: "+relative.as_posix())


def waves(rows,data_root):
    import pyarrow.parquet as pq
    import soundfile as sf
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[relative_parquet(r),int(r["row_group"])].append((i,r))
    for (relative,group),items in groups.items():
        audio=pq.ParquetFile(Path(data_root)/relative).read_row_group(group,columns=["audio"])["audio"]
        for i,r in items:
            item=audio[int(r["row_offset"])].as_py()
            encoded=item["bytes"] if isinstance(item,dict) else item
            if isinstance(encoded,memoryview):
                encoded=encoded.tobytes()
            if not isinstance(encoded,bytes) or hashlib.sha256(encoded).hexdigest()!=r["audio_sha256"]:
                raise ValueError("Embedded audio hash mismatch")
            waveform,sr=sf.read(io.BytesIO(encoded),dtype="float32",always_2d=True)
            waveform=waveform.mean(1)
            if sr!=16000 or hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest()!=r["pcm_sha256"]:
                raise ValueError("Decoded audio contract mismatch")
            yield i,waveform
