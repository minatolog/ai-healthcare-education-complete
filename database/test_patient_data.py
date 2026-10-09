"""Data-only regression checks; no model or application dependencies."""
import copy
import unittest

from build_patient_seed import ROOT, MANIFEST, build, diabetes_records, read_csv, validate_manifest


class PatientDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_csv(ROOT / MANIFEST)
        cls.patients, cls.images, cls.mapping = build()

    def test_full_manifest_coverage_and_identity(self):
        self.assertEqual(len(self.patients), 40)
        self.assertEqual(len({p['name'] for p in self.patients}), 40)
        self.assertTrue(all(p['gender'] is None for p in self.patients))
        self.assertEqual(len(self.images), 7514)
        case_by_code = {p['patient_code']: p['kidney_case_id'] for p in self.patients}
        actual = {(case_by_code[i['patient_code']], i['slice_index'], i['img_type'], i['file_path'])
                  for i in self.images}
        expected = {(r['case'], int(r['z']), kind, 'data/kidney/data2d/' + r[key])
                    for r in self.rows for key, kind in [('image', 'kidney_ct'), ('mask', 'kidney_reference_mask')]}
        self.assertEqual(actual, expected)

    def test_original_diabetes_values_are_preserved(self):
        originals = {r['patient_id']: r for r in diabetes_records(ROOT)}
        for p in self.patients:
            values = {'patient_id': p['patient_code'], 'age_years': p['age'],
                      **p['clinical_data'], **p['biomarker'],
                      'diabetes_label': int(p['label'] == 'diabetes_positive')}
            self.assertEqual(values, originals[p['patient_code']])

    def test_csv_reordering_does_not_change_case_order(self):
        self.assertEqual(validate_manifest(self.rows, ROOT), validate_manifest(list(reversed(self.rows)), ROOT))

    def test_reject_patient_spanning_pools(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['pool'] = 'test'
        with self.assertRaisesRegex(ValueError, 'multiple pools'):
            validate_manifest(rows, ROOT)

    def test_reject_duplicate_slice(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_manifest(self.rows + [self.rows[0]], ROOT)

    def test_reject_wrong_or_escaping_file_path(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['image'] = '../outside.png'
        with self.assertRaisesRegex(ValueError, 'path'):
            validate_manifest(rows, ROOT)

    def test_reference_targets_not_in_diabetes_features_or_ct_metadata(self):
        for p in self.patients:
            self.assertNotIn('diabetes_label', p['clinical_data'])
            self.assertNotIn('diabetes_label', p['biomarker'])
        for image in self.images:
            if image['img_type'] == 'kidney_ct':
                self.assertNotIn('reference_labels', image['image_metadata'])


if __name__ == '__main__':
    unittest.main()
