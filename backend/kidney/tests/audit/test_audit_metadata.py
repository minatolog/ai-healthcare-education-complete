"""Behaviour checks for the read-only MR-01 audit, using synthetic metadata."""
import csv
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_kidney_metadata.py"
SPEC = importlib.util.spec_from_file_location("audit_kidney_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data2d"
        (self.data / "images").mkdir(parents=True)
        (self.data / "masks").mkdir()

    def row(self, case="case_00000", tumour=True, **extra):
        return {"case": case, "z": "1", "image": f"images/{case}_z001.png",
                "mask": f"masks/{case}_z001.png", "has_kidney": "1",
                "has_tumor": str(int(tumour)), "kidney_px": "30",
                "tumor_px": "5" if tumour else "0", "n_slices": "3", **extra}

    def write(self, rows, splits):
        with (self.data / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.data / "splits.json").write_text(json.dumps(splits))

    def png(self, relative):
        # The auditor intentionally inspects only headers, not PNG pixel data.
        (self.data / relative).write_bytes(
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">IIBBBBB", 256, 256, 8, 0, 0, 0, 0) + b"\0" * 4)

    def codes(self, report):
        return {item["code"] for item in report["issues"]}

    def test_patient_overlap_and_source_identity_overlap(self):
        rows = [self.row(source_patient_id="same_source"),
                self.row("case_00001", source_patient_id="same_source")]
        self.write(rows, {"train": ["case_00000"], "val": ["case_00001"],
                          "test": ["case_00000"]})
        report = MODULE.audit(self.root, {"train": 1, "val": 1, "test": 1})
        self.assertIn("patient_split_leakage", self.codes(report))
        self.assertEqual(report["splits"]["source_patient_overlap_groups"]["same_source"],
                         ["test", "train", "val"])

    def test_incomplete_pairing_and_all_positive_viability(self):
        rows = [self.row(), self.row("case_00001")]
        self.write(rows, {"train": ["case_00000"], "test": ["case_00001"]})
        self.png(rows[0]["image"])
        self.png(rows[0]["mask"])
        self.png(rows[1]["image"])
        report = MODULE.audit(self.root, {"train": 1, "test": 1})
        self.assertEqual(report["slice_cache"]["paired_png_stems"], 1)
        self.assertEqual(report["slice_cache"]["image_only_stems"], 1)
        self.assertEqual(report["slice_cache"]["missing_manifest_mask_rows"], 1)
        self.assertEqual(report["slice_cache"]["complete_manifest_patient_count"], 1)
        self.assertIn("no_negative_patient_cohort", self.codes(report))
        self.assertFalse(report["patient_binary_classification"][
            "can_establish_two_patient_classes_from_this_manifest"])

    def test_zero_tumour_slice_never_becomes_confirmed_negative_patient(self):
        self.write([self.row(tumour=False)], {"train": ["case_00000"]})
        report = MODULE.audit(self.root, {"train": 1})
        self.assertEqual(report["manifest"]["patients_without_observed_tumour_slice"], 1)
        self.assertIsNone(report["patient_binary_classification"]["confirmed_negative_patients"])
        self.assertNotIn("no_negative_patient_cohort", self.codes(report))

    def test_duplicate_and_unsafe_paths_are_rejected(self):
        row = self.row(image="../outside.png")
        self.write([row, row], {"train": ["case_00000", "case_00000"]})
        codes = self.codes(MODULE.audit(self.root, {"train": 2}))
        self.assertTrue({"unsafe_manifest_paths", "duplicate_case_slice",
                         "duplicate_patient_in_split"} <= codes)

    def test_counts_unknown_patients_and_stale_stats(self):
        self.write([self.row()], {"train": ["case_99999"]})
        (self.data / "dataset_stats.json").write_text(json.dumps({"n_cases": 99}))
        report = MODULE.audit(self.root)
        self.assertTrue({"split_count_mismatch", "split_patient_missing_from_manifest",
                         "unassigned_manifest_patients", "stale_dataset_stats"}
                        <= self.codes(report))

    def test_new_pool_must_be_disjoint_from_training_pool(self):
        self.write([self.row()], {"train": ["case_00000"], "new": ["case_00000"]})
        report = MODULE.audit(self.root, {"train": 1})
        self.assertEqual(report["splits"]["pairwise_patient_overlaps"]["new/train"],
                         ["case_00000"])
        self.assertIn("patient_split_leakage", self.codes(report))

    def test_read_only_no_absolute_paths_and_ignores_ds_store(self):
        row = self.row()
        self.write([row], {"train": ["case_00000"]})
        self.png(row["image"])
        self.png(row["mask"])
        (self.data / "images" / ".DS_Store").write_text("ignored")
        before = {p.relative_to(self.root).as_posix(): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        report = MODULE.audit(self.root, {"train": 1})
        after = {p.relative_to(self.root).as_posix(): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertEqual(report["slice_cache"]["images"]["file_count"], 1)
        self.assertFalse(report["raw_volume_contents_loaded"])

    def test_invalid_png_header_and_malformed_metadata(self):
        row = self.row()
        self.write([row], {"train": ["case_00000"]})
        (self.data / row["image"]).write_text("not png")
        report = MODULE.audit(self.root, {"train": 1})
        self.assertIn("invalid_png_headers", self.codes(report))
        (self.data / "splits.json").write_text("[1, 2]")
        self.assertIn("invalid_split_schema", self.codes(MODULE.audit(self.root)))

    def test_cli_strict_emits_redacted_json_and_fails_for_incomplete_cache(self):
        self.write([self.row()], {"train": ["case_00000"]})
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root),
             "--expected", "train=1", "--strict"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "action_required")
        self.assertNotIn(str(self.root), result.stdout)

    def test_empty_manifest_and_non_object_stats_are_reported(self):
        row = self.row()
        self.write([row], {"train": ["case_00000"]})
        (self.data / "dataset_stats.json").write_text("[]")
        self.assertIn("unreadable_dataset_stats", self.codes(MODULE.audit(self.root)))
        self.write([row], {"train": []})
        with (self.data / "manifest.csv").open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writeheader()
        self.assertIn("empty_manifest", self.codes(MODULE.audit(self.root)))

    def test_duplicate_case_header_rejected_before_patient_identity_is_overwritten(self):
        self.write([self.row()], {"train": ["case_00000"]})
        (self.data / "manifest.csv").write_text(
            "case,case,z,n_slices,has_kidney,has_tumor,kidney_px,tumor_px,image,mask\n"
            "case_99999,case_00000,1,3,1,1,30,5,images/case_00000_z001.png,masks/case_00000_z001.png\n")
        report = MODULE.audit(self.root, {"train": 1})
        self.assertEqual(report["status"], "action_required")
        self.assertEqual(report["issues"], [{"code": "invalid_manifest_headers", "severity": "error",
                                             "empty_header_count": 0, "duplicate_header_count": 1}])
        self.assertNotIn("manifest", report)

    def test_empty_or_whitespace_header_names_are_rejected(self):
        row = self.row()
        for invalid_header in ("", "   "):
            with self.subTest(header=invalid_header):
                self.write([row], {"train": ["case_00000"]})
                with (self.data / "manifest.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([*row, invalid_header])
                    writer.writerow([*row.values(), "unused"])
                report = MODULE.audit(self.root, {"train": 1})
                self.assertEqual(report["status"], "action_required")
                self.assertIn("invalid_manifest_headers", self.codes(report))
                self.assertEqual(report["issues"][0]["empty_header_count"], 1)

    def test_short_row_missing_png_path_returns_json_diagnostic_in_strict_cli(self):
        self.write([self.row()], {"train": ["case_00000"]})
        (self.data / "manifest.csv").write_text(
            "case,z,n_slices,has_kidney,has_tumor,kidney_px,tumor_px,image,mask\n"
            "case_00000,1,3,1,1,30,5,images/case_00000_z001.png\n")
        before = (self.data / "manifest.csv").read_bytes()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root),
             "--expected", "train=1", "--strict"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "action_required")
        self.assertIn("malformed_manifest_rows", self.codes(report))
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn(str(self.root), result.stdout)
        self.assertEqual((self.data / "manifest.csv").read_bytes(), before)

    def test_empty_image_or_mask_paths_are_diagnostics_not_path_exceptions(self):
        for field in ("image", "mask"):
            for value in ("", "   "):
                with self.subTest(field=field, value=value):
                    self.write([self.row(**{field: value})], {"train": ["case_00000"]})
                    report = MODULE.audit(self.root, {"train": 1})
                    self.assertEqual(report["status"], "action_required")
                    self.assertIn("malformed_manifest_rows", self.codes(report))

    def test_extra_row_fields_and_missing_optional_fields_are_rejected(self):
        row = self.row(source_patient_id="source_00000")
        self.write([row], {"train": ["case_00000"]})
        original = (self.data / "manifest.csv").read_text()
        lines = original.splitlines()
        for broken_row in (lines[1] + ",unexpected", lines[1].rsplit(",", 1)[0]):
            with self.subTest(row=broken_row):
                (self.data / "manifest.csv").write_text(lines[0] + "\n" + broken_row + "\n")
                report = MODULE.audit(self.root, {"train": 1})
                self.assertEqual(report["status"], "action_required")
                self.assertIn("malformed_manifest_rows", self.codes(report))

    def test_unterminated_csv_quote_returns_unreadable_metadata_diagnostic(self):
        self.write([self.row()], {"train": ["case_00000"]})
        with (self.data / "manifest.csv").open("a") as handle:
            handle.write('"unterminated record\n')
        report = MODULE.audit(self.root, {"train": 1})
        self.assertEqual(report["status"], "action_required")
        self.assertIn("unreadable_metadata", self.codes(report))


if __name__ == "__main__":
    unittest.main()
