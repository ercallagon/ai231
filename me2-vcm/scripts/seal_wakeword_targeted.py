"""Seal completed regression evidence for the wakeword queue."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from hey_swift.vcm.protocol import sha256

stage = Path("work/wakeword_targeted_v2")
report = stage / "regression.xml"
root = ET.parse(report).getroot()
cases = list(root.iter("testcase"))
if len(cases) < 87 or list(root.iter("failure")) or list(root.iter("error")):
    raise RuntimeError("Incomplete or failing regression suite")
with (stage / "regression_passed.json").open("x") as stream:
    json.dump({"passed": True, "tests": len(cases), "sha256": sha256(report)}, stream, indent=2)
print(f"Sealed {len(cases)} regression cases")
