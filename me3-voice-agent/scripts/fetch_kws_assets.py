"""Fetch selected official English Sherpa inference assets and tokenize a phrase.

Explicit opt-in network setup; runtime never downloads models. No training data.
Model README declares Apache License 2.0; preserve provenance and notices.
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from me3_voice.artifacts import output_path

NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
ASSETS = Path("var/cache/wake") / NAME
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/" + NAME + ".tar.bz2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch():
    ASSETS.mkdir(parents=True, exist_ok=True)
    archive = ASSETS.parent / (NAME + ".tar.bz2")
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read(40 * 1024 * 1024 + 1)
        if len(data) > 40 * 1024 * 1024:
            raise ValueError("archive exceeds limit")
        archive.write_bytes(data)
    manifest = {
        "url": URL,
        "archive_sha256": digest(archive),
        "checksum_origin": "computed on HTTPS download, not vendor signed",
        "files": {},
    }
    # Select file contents; never extract archive paths, links or bundled test data.
    with tarfile.open(archive, "r:bz2") as tf:
        for member in tf.getmembers():
            p = Path(member.name)
            if len(p.parts) != 2 or p.parts[0] != NAME or not member.isfile():
                continue
            if p.name not in ["README.md", "bpe.model", "tokens.txt"] and not p.name.endswith(
                ".int8.onnx"
            ):
                continue
            data = tf.extractfile(member).read()
            path = ASSETS / p.name
            if path.exists() and path.read_bytes() != data:
                raise ValueError("preserve changed existing asset")
            if not path.exists():
                path.write_bytes(data)
            manifest["files"][p.name] = {"sha256": digest(path), "bytes": len(data)}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", default="HEY SWIFT")
    parser.add_argument(
        "--keywords-output", type=output_path, default="var/cache/wake/hey-swift-keywords.txt"
    )
    args = parser.parse_args()
    if not 1 <= len(args.phrase) <= 80 or not all(
        c.isascii() and (c.isalpha() or c == " ") for c in args.phrase
    ):
        parser.error("phrase must contain 1..80 ASCII letters/spaces")
    if not args.phrase.strip():
        parser.error("empty phrase")
    if args.keywords_output.exists():
        parser.error("preserve existing keyword file; supply a new output path")
    fetch()
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(ASSETS / "bpe.model"))
    tokens = tokenizer.encode(args.phrase.upper(), out_type=str)
    known = {line.rsplit(" ", 1)[0] for line in (ASSETS / "tokens.txt").read_text().splitlines()}
    if tokenizer.unk_id() in tokenizer.encode(args.phrase.upper()) or not set(tokens).issubset(
        known
    ):
        parser.error("phrase has unknown model tokens")
    label = "_".join(args.phrase.upper().split())
    with args.keywords_output.open("x") as stream:
        stream.write(" ".join(tokens) + " @" + label + "\n")
    print(f"Inference assets: {ASSETS}; configured keyword file: {args.keywords_output}")


if __name__ == "__main__":
    main()
