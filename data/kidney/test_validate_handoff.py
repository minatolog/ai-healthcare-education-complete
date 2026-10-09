"""Small meaningful corruption fixtures for the standalone handoff validator."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import validate_handoff as validator


HEADERS = ["case", "z", "image", "mask", "has_kidney", "has_tumor",
           "kidney_px", "tumor_px", "n_slices", "sp_z", "sp_y", "sp_x"]


def row(patient="case_00000", z=4, **changes):
    result = {"case": patient, "z": str(z), "image": "images/%s_z%03d.png" % (patient, z),
              "mask": "masks/%s_z%03d.png" % (patient, z), "has_kidney": "1",
              "has_tumor": "1", "kidney_px": "1", "tumor_px": "1",
              "n_slices": "8", "sp_z": "3.0", "sp_y": "0.7", "sp_x": "0.7"}
    result.update(changes)
    return result


class HandoffFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def png(self, name="data2d/masks/case_00000_z004.png", values=(1, 2), mode="L", size=(256, 256)):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new(mode, size)
        if mode == "L":
            for index, value in enumerate(values):
                image.putpixel((index, 0), value)
        image.save(path)
        histogram = image.histogram()
        facts = {"byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "decoded_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                 "kind": "mask", "label_counts": {str(i): histogram[i] for i in (0, 1, 2)}}
        return name, facts

    def checksums(self, paths):
        content = "".join(hashlib.sha256((self.root / path).read_bytes()).hexdigest()
                          + "  " + path + "\n" for path in sorted(paths))
        (self.root / "checksums.sha256").write_text(content, encoding="utf-8")

    def test_valid_reference_pair_checks_original_pixels(self):
        name, facts = self.png()
        validator.verify_png(self.root, name, facts, row())

    def test_mask_tampering_fails_original_audit_binding(self):
        name, facts = self.png()
        Image.new("L", (256, 256), 2).save(self.root / name)
        with self.assertRaisesRegex(validator.HandoffError, "differs from the preserved source audit"):
            validator.verify_png(self.root, name, facts, row())

    def test_unrecognised_mask_labels_are_not_accepted(self):
        name, facts = self.png(values=(1, 3))
        with self.assertRaisesRegex(validator.HandoffError, "Unknown mask class"):
            validator.verify_png(self.root, name, facts, row())

    def test_reference_mask_counts_must_match_csv(self):
        name, facts = self.png()
        with self.assertRaisesRegex(validator.HandoffError, "manifest metadata"):
            validator.verify_png(self.root, name, facts, row(kidney_px="2"))

    def test_color_png_is_rejected_despite_matching_file_hash(self):
        name, facts = self.png(mode="RGB")
        with self.assertRaisesRegex(validator.HandoffError, "single grayscale"):
            validator.verify_png(self.root, name, facts, row())

    def test_ct_is_not_interpreted_as_mask_class_ids(self):
        name, facts = self.png("data2d/images/case_00000_z004.png", values=(79, 200))
        facts["kind"] = "image"
        validator.verify_png(self.root, name, facts, row())

    def test_patient_slice_duplicates_are_rejected(self):
        with self.assertRaisesRegex(validator.HandoffError, "Duplicate patient/slice"):
            validator.manifest_index(HEADERS, [row(), row()])

    def test_source_slice_must_belong_to_original_scan(self):
        with self.assertRaisesRegex(validator.HandoffError, "Invalid source slice"):
            validator.manifest_index(HEADERS, [row(z=8)])

    def test_project_projection_preserves_source_and_locked_pool(self):
        first, second = row(), row("case_00001")
        source = validator.manifest_index(HEADERS, [first, second])
        patients = {"case_00000": {"pool": "train"}}
        result = validator.verify_projection(HEADERS, source, HEADERS + ["pool"],
                                             [{**first, "pool": "train"}], patients)
        self.assertEqual(set(result), {("case_00000", 4)})
        with self.assertRaisesRegex(validator.HandoffError, "pool mismatch"):
            validator.verify_projection(HEADERS, source, HEADERS + ["pool"],
                                        [{**first, "pool": "test"}], patients)

    def test_projection_must_not_silently_drop_selected_slice(self):
        first, second = row(z=4), row(z=5)
        source = validator.manifest_index(HEADERS, [first, second])
        with self.assertRaisesRegex(validator.HandoffError, "every selected source slice"):
            validator.verify_projection(HEADERS, source, HEADERS + ["pool"],
                                        [{**first, "pool": "train"}], {"case_00000": {"pool": "train"}})

    def test_projection_cannot_change_original_spacing(self):
        original = row()
        source = validator.manifest_index(HEADERS, [original])
        with self.assertRaisesRegex(validator.HandoffError, "Project/source metadata mismatch"):
            validator.verify_projection(HEADERS, source, HEADERS + ["pool"],
                                        [{**original, "sp_x": "1.0", "pool": "train"}],
                                        {"case_00000": {"pool": "train"}})

    def test_path_traversal_is_rejected_before_reading(self):
        for path in ("../secret.txt", "/tmp/secret.txt", "data2d/../secret.txt", "data2d//file.png"):
            with self.subTest(path=path), self.assertRaises(validator.HandoffError):
                validator.safe_file(self.root, path)

    def test_symlink_asset_is_rejected(self):
        (self.root / "real.txt").write_text("contents", encoding="utf-8")
        (self.root / "linked.txt").symlink_to(self.root / "real.txt")
        with self.assertRaisesRegex(validator.HandoffError, "Symlinks"):
            validator.safe_file(self.root, "linked.txt")

    def test_checksum_coverage_rejects_unindexed_file(self):
        (self.root / "metadata.json").write_text("{}", encoding="utf-8")
        self.checksums(["metadata.json"])
        (self.root / "unexpected.txt").write_text("extra", encoding="utf-8")
        with self.assertRaisesRegex(validator.HandoffError, "Checksum coverage"):
            validator.read_checksums(self.root)

    def test_duplicate_checksum_path_is_rejected(self):
        (self.root / "metadata.json").write_text("{}", encoding="utf-8")
        self.checksums(["metadata.json"])
        path = self.root / "checksums.sha256"
        path.write_text(path.read_text() * 2, encoding="utf-8")
        with self.assertRaisesRegex(validator.HandoffError, "Duplicate"):
            validator.read_checksums(self.root)

    def test_metadata_only_does_not_open_png_bytes(self):
        name, facts = self.png()
        (self.root / "metadata.json").write_text("{}", encoding="utf-8")
        self.checksums([name, "metadata.json"])
        checksums = validator.read_checksums(self.root)
        original_read_bytes = Path.read_bytes
        def guarded_read_bytes(path):
            if path.suffix == ".png":
                raise AssertionError("Metadata-only validation read PNG contents.")
            return original_read_bytes(path)
        with patch.object(Path, "read_bytes", guarded_read_bytes):
            self.assertEqual(validator.verify_checksums(self.root, checksums, metadata_only=True), 1)
        Image.new("L", (256, 256), 2).save(self.root / name)
        with self.assertRaisesRegex(validator.HandoffError, "Checksum mismatch"):
            validator.verify_checksums(self.root, checksums, metadata_only=False)

    def test_duplicate_json_keys_are_rejected(self):
        (self.root / "metadata.json").write_text('{"pool":"train","pool":"test"}', encoding="utf-8")
        with self.assertRaisesRegex(validator.HandoffError, "Duplicate JSON key"):
            validator.read_json(self.root, "metadata.json")

    def test_malformed_csv_columns_are_rejected(self):
        (self.root / "metadata.csv").write_text("case,z\ncase_00000,4,unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(validator.HandoffError, "Malformed"):
            validator.read_csv(self.root, "metadata.csv")


if __name__ == "__main__":
    unittest.main()
