"""Keep sandbox staging separate from final Pi deliverables."""
import importlib.util
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "rpi_packager", Path(__file__).resolve().parents[1] / "scripts/package_rpi_patch_v6.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class PackagingLayoutTests(unittest.TestCase):
    def test_archives_in_work_unpacked_release_in_deploy(self):
        root = Path("/tmp/example-repo")
        stage, archive, checksum, final = packager.output_paths(root, "release.v6")
        self.assertEqual(stage, root / "work/rpi_patch_staging/release.v6")
        self.assertEqual(archive, root / "work/release.v6.tar.gz")
        self.assertEqual(checksum, root / "work/release.v6.tar.gz.sha256")
        self.assertEqual(final, root / "deploy/rpi/release.v6")

    def test_refuses_existing_stage_or_final_deliverables(self):
        for position in range(4):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = packager.output_paths(root, "release")
                target = paths[position]
                target.parent.mkdir(parents=True)
                if position in {0, 3}:
                    target.mkdir()
                else:
                    target.write_text("existing artifact")
                with self.assertRaises(FileExistsError):
                    packager.output_paths(root, "release")
                self.assertTrue(target.exists())

    def test_refuses_invalid_names(self):
        for name in ("", ".", "..", "../escape", "/absolute"):
            with self.assertRaises(ValueError):
                packager.output_paths(Path("/tmp/example-repo"), name)

    def test_build_keeps_tar_in_work_and_publishes_complete_unpacked_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("src/hey_swift/example.py", "docs/rpi-runtime-v6.md",
                             "scripts/apply_rpi_runtime_patch.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic packaging fixture\n")
            protected = root / "fixture-config.yaml"
            protected.write_text("unchanged: true\n")
            base = root / "base.tar.gz"
            with tarfile.open(base, "w:gz") as tar:
                tar.add(protected, arcname="baseline/configs/protected.yaml")
            with patch.object(packager, "ROOT", root), \
                 patch.object(packager, "BASE", base), \
                 patch.object(packager, "BASE_SHA", packager.digest(base)), \
                 patch.object(packager, "FILES", ["src/hey_swift/example.py"]), \
                 patch.object(packager, "PROTECTED", ["configs/protected.yaml"]), \
                 patch.object(packager.subprocess, "run"), \
                 patch.object(packager.subprocess, "check_output", return_value="fixture-head\n"), \
                 patch.object(packager.sys, "argv", ["packager", "--name", "release"]):
                packager.main()
            final = root / "deploy/rpi"
            self.assertEqual({p.name for p in final.iterdir()}, {"release"})
            self.assertTrue((final / "release/payload").is_dir())
            self.assertFalse((root / "work/rpi_patch_staging/release").exists())
            with tarfile.open(root / "work/release.tar.gz") as tar:
                self.assertIn("release/PATCH_MANIFEST.json", tar.getnames())
            self.assertEqual((root / "work/release.tar.gz.sha256").read_text(),
                             packager.digest(root / "work/release.tar.gz") + "  release.tar.gz\n")


if __name__ == "__main__":
    unittest.main()
