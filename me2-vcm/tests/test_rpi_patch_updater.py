"""Installer safeguards exercised against temporary, synthetic installations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "rpi_patch_updater", Path(__file__).resolve().parents[1] / "scripts/apply_rpi_runtime_patch.py")
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)


class PatchUpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "install"
        self.package = Path(self.temp.name) / "patch"
        self.rel = "src/hey_swift/integration/e2e_loop.py"
        self.old = self.root / self.rel
        self.old.parent.mkdir(parents=True)
        self.old.write_text("original\n")
        self.payload = self.package / "payload" / self.rel
        self.payload.parent.mkdir(parents=True)
        self.payload.write_text("updated\n")
        (self.root / "models").mkdir()
        (self.root / "models/protected.onnx").write_bytes(b"synthetic protected asset")
        (self.root / "audio").mkdir()
        (self.root / "audio/wake_chime.wav").write_bytes(b"local amplified cue")
        (self.root / "audio/wake_chime_original.wav").write_bytes(b"local original cue")
        self.manifest = {
            "files": [{"path": self.rel, "sha256": updater.sha(self.payload),
                       "before_sha256": updater.sha(self.old)}],
            "protected": {"models/protected.onnx": updater.sha(self.root / "models/protected.onnx")},
        }
        self.write_manifest()
        stopped = patch.object(updater, "ensure_stopped")
        stopped.start()
        self.addCleanup(stopped.stop)

    def write_manifest(self):
        (self.package / "PATCH_MANIFEST.json").write_text(json.dumps(self.manifest))

    def test_refuses_local_edits_without_changes(self):
        self.old.write_text("Pi local fix\n")
        with self.assertRaisesRegex(RuntimeError, "local edit conflict"):
            updater.validate(self.root, self.package)
        self.assertEqual(self.old.read_text(), "Pi local fix\n")

    def test_refuses_corrupt_payload(self):
        self.payload.write_text("corrupt\n")
        with self.assertRaisesRegex(ValueError, "Corrupt payload"):
            updater.validate(self.root, self.package)

    def test_refuses_protected_asset_change(self):
        (self.root / "models/protected.onnx").write_bytes(b"different model")
        with self.assertRaisesRegex(RuntimeError, "Protected model/config"):
            updater.validate(self.root, self.package)

    def test_refuses_existing_receipt_before_editing(self):
        (self.root / "V6_PATCH_SHA256SUMS").write_text("prior receipt\n")
        with self.assertRaises(FileExistsError):
            updater.apply(self.root, self.package)
        self.assertEqual(self.old.read_text(), "original\n")

    def test_apply_preserves_chimes_and_protected_asset_then_rolls_back(self):
        with patch.object(updater, "post_validate"):
            updater.apply(self.root, self.package)
        self.assertEqual(self.old.read_text(), "updated\n")
        self.assertEqual((self.root / "audio/wake_chime.wav").read_bytes(), b"local amplified cue")
        self.assertEqual((self.root / "audio/wake_chime_original.wav").read_bytes(), b"local original cue")
        backup = next((self.root / "deploy/rpi/backups").iterdir())
        updater.rollback(self.root, backup)
        self.assertEqual(self.old.read_text(), "original\n")
        self.assertEqual(updater.sha(self.root / "models/protected.onnx"),
                         self.manifest["protected"]["models/protected.onnx"])

    def test_failed_validation_restores_original(self):
        with patch.object(updater, "post_validate", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                updater.apply(self.root, self.package)
        self.assertEqual(self.old.read_text(), "original\n")
        self.assertFalse((self.root / "V6_PATCH_SHA256SUMS").exists())

    def test_rollback_refuses_later_local_edits(self):
        with patch.object(updater, "post_validate"):
            updater.apply(self.root, self.package)
        backup = next((self.root / "deploy/rpi/backups").iterdir())
        self.old.write_text("later Pi fix\n")
        with self.assertRaisesRegex(RuntimeError, "subsequent local edits"):
            updater.rollback(self.root, backup)
        self.assertEqual(self.old.read_text(), "later Pi fix\n")

    def test_refuses_path_escape(self):
        self.manifest["files"][0]["path"] = "src/hey_swift/../../../outside.py"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe patch path"):
            updater.validate(self.root, self.package)


if __name__ == "__main__":
    unittest.main()
