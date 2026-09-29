"""In-memory checks for supplement integrity and observation identity gates."""
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_aurn_supplement as audit


class SupplementTests(unittest.TestCase):
    def fixture(self, extra=None, corrupt=False):
        payload = {k: k.encode() for k in audit.EXPECTED}
        hashes = {k: hashlib.sha256(v).hexdigest() for k, v in payload.items()}
        payload["README.md"] = b"Provenance only"
        payload["CHECKSUMS.sha256"] = "\n".join(f"{v}  {k}" for k, v in hashes.items()).encode()
        if corrupt:
            payload[audit.SCREEN] = b"modified"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name, data in payload.items():
                archive.writestr(audit.PREFIX + name, data)
            if extra:
                archive.writestr(extra, b"not trusted")
        stream.seek(0)
        return zipfile.ZipFile(stream), hashes

    def test_exact_payload_passes(self):
        archive, hashes = self.fixture()
        with archive, patch.dict(audit.EXPECTED, hashes, clear=True):
            self.assertEqual(len(audit.supplement_payload(archive)), 6)

    def test_modified_payload_fails(self):
        archive, hashes = self.fixture(corrupt=True)
        with archive, patch.dict(audit.EXPECTED, hashes, clear=True):
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                audit.supplement_payload(archive)

    def test_traversal_and_unexpected_code_fail(self):
        for name in (audit.PREFIX + "../escape.json", audit.PREFIX + "run.py",
                     audit.PREFIX + "C:/escape.json", "/" + audit.PREFIX + "bad.json"):
            archive, hashes = self.fixture(extra=name)
            with self.subTest(name=name), archive, patch.dict(audit.EXPECTED, hashes, clear=True):
                with self.assertRaises(ValueError):
                    audit.supplement_payload(archive)

    def test_comparison_rejects_selection_change(self):
        with self.assertRaises(ValueError):
            audit.check_tree({"eligible": False}, {"eligible": True})
        with self.assertRaises(ValueError):
            audit.check_tree({"rmse": 3.3}, {"rmse": 3.2})


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.station = {"source": np.array([1, 1]), "date_idx": np.array([730, 731]),
                        "site_idx": np.array([0, 0]), "site_codes": np.array(["A"]),
                        "row": np.array([2, 2]), "col": np.array([3, 3]),
                        "value": np.array([10., 20.], np.float32)}
        self.frame = pd.DataFrame({"station_row_index": [0, 1], "date_idx": [730, 731],
                                   "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                                   "site_code": ["A", "A"], "row": [2, 2], "col": [3, 3],
                                   "observed_pm25": [10., 20.], "predicted_pm25": [9., 19.]})

    def test_unchanged_rows_pass(self):
        self.assertTrue(audit.validate_rows(self.frame, self.station, 730, 731)["complete_original_laqn_membership"])

    def test_duplicate_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.validate_rows(pd.concat([self.frame, self.frame.iloc[[0]]]), self.station, 730, 731)

    def test_changed_observation_fails(self):
        self.frame.loc[1, "observed_pm25"] = 15.
        with self.assertRaises(AssertionError):
            audit.validate_rows(self.frame, self.station, 730, 731)

    def test_missing_original_row_fails(self):
        with self.assertRaises(AssertionError):
            audit.validate_rows(self.frame.iloc[:1], self.station, 730, 731)


if __name__ == "__main__":
    unittest.main()
