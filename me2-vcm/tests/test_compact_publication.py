import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("compact", ROOT / "scripts/prepare_compact_publication.py")
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("path", ["src/hey_swift/__init__.py", "configs/public-runtime.env.example", "docs/public-reproduction.md", ".gitignore"])
def test_relevant_source_retained(path):
    assert module().git_candidate(path)


@pytest.mark.parametrize("path", ["data/vcm/protocols/hf_grouped_v2b/manifests/train_19intent.csv",
    "work/cache/mels.npy", "models/vcm/model.onnx", "models/checkpoints/best.pt", "src/__pycache__/x.py",
    "scripts/auto_queue_r010.sh", "integration/state/login.json", ".env"])
def test_intermediates_weights_and_large_metadata_not_git_source(path):
    assert not module().git_candidate(path)


def test_metadata_archive_deterministic_and_preserves_sources(tmp_path):
    tool = module()
    root = tmp_path / "source"; root.mkdir()
    source = root / "manifests/train.csv"; source.parent.mkdir(); source.write_text("id,intent\n1,TIMER\n")
    before = tool.digest(source)
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    tool.metadata_archive(root, first, [source]); tool.metadata_archive(root, second, [source])
    assert tool.digest(first) == tool.digest(second)
    assert tool.digest(source) == before
    with tarfile.open(first) as archive:
        assert archive.getnames() == ["manifests/train.csv", "METADATA_SHA256.json"]
        proof = json.load(archive.extractfile("METADATA_SHA256.json"))
        assert proof["manifests/train.csv"] == before
    with pytest.raises(FileExistsError): tool.metadata_archive(root, first, [source])
