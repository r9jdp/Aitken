"""Read-only/in-memory safety checks for the AURN archive importer."""
import io
from pathlib import Path
import stat
import sys
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_aurn_bundle import REPO, inside, safe_members


class ImportSafetyTests(unittest.TestCase):
    def archive(self, extra=()):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in ("manifest.json", "aurn_sites.json", "aurn_features.npz",
                         "selected_sites.csv", "coverage.csv", "aurn_daily.csv", "daily_availability.csv"):
                archive.writestr("aurn/" + name, b"example")
            for name in extra:
                archive.writestr(name, b"example")
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_mac_resource_forks_are_ignored(self):
        with self.archive(["__MACOSX/aurn/._manifest.json"]) as archive:
            self.assertEqual(len(safe_members(archive)), 7)

    def test_traversal_and_windows_absolute_paths_are_rejected(self):
        for name in ("aurn/../escape.csv", "aurn/C:/escape.csv", "aurn\\..\\escape.csv", "/aurn/escape.csv"):
            with self.subTest(name=name), self.archive([name]) as archive:
                with self.assertRaises(ValueError):
                    safe_members(archive)

    def test_case_insensitive_duplicate_is_rejected(self):
        with self.archive(["aurn/MANIFEST.json"]) as archive:
            with self.assertRaises(ValueError):
                safe_members(archive)

    def test_executable_payload_is_rejected(self):
        with self.archive(["aurn/run.py"]) as archive:
            with self.assertRaises(ValueError):
                safe_members(archive)

    def test_symlink_is_rejected(self):
        member = zipfile.ZipInfo("aurn/link.csv")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.archive([member]) as archive:
            with self.assertRaises(ValueError):
                safe_members(archive)

    def test_output_must_be_inside_repository(self):
        self.assertEqual(inside(REPO / "data/external_features/aurn"), REPO / "data/external_features/aurn")
        with self.assertRaises(ValueError):
            inside(REPO)
        with self.assertRaises(ValueError):
            inside(REPO.parent / "outside")


if __name__ == "__main__":
    unittest.main()
