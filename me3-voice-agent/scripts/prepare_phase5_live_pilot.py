"""Select up to four existing Phase 4 cases for a later, budget-capped live pilot.

This script only writes a case file. It never enables or sends provider requests.
"""

import argparse
import hashlib
import json
from pathlib import Path

from me3_voice.artifacts import output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("tests/fixtures/phase4_questions.json"))
    parser.add_argument("--case-ids", required=True, help="comma-separated existing case IDs (1..4)")
    parser.add_argument("--output", type=output_path, required=True)
    args = parser.parse_args()
    selected = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    if not 1 <= len(selected) <= 4 or len(set(selected)) != len(selected):
        parser.error("select one to four unique case IDs")
    raw = args.source.read_bytes()
    value = json.loads(raw)
    index = {case["id"]: case for case in value["cases"]}
    missing = [case_id for case_id in selected if case_id not in index]
    if missing:
        parser.error("unknown case ID(s): " + ",".join(missing))
    result = {
        "evidence": "PREPARED_NOT_EXECUTED",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "case_ids": selected,
        "cases": [index[case_id] for case_id in selected],
        "live_requests_sent": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("Prepared pilot cases; no live requests sent:", args.output)


if __name__ == "__main__":
    main()
