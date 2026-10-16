"""Verify real JSON data, modality boundaries and allowlisted CT serving."""
import json
import threading
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from build_patient_data import build
from server import make_server, PatientStore, ROOT


class PatientApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = PatientStore()
        cls.server = make_server(port=0, store=cls.store)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def get(self, path):
        with urlopen(self.base + path) as response:
            return json.load(response)

    def test_store_matches_all_source_fields(self):
        patients, images, _ = build()
        self.assertEqual(self.store.patients, patients)
        self.assertEqual(json.loads((ROOT/'backend/data/patient_images.json').read_text()), images)
        self.assertEqual(json.loads((ROOT/'backend/data/training_runs.json').read_text()), [])

    def test_modality_boundaries_and_values(self):
        diabetes = self.get('/api/diabetes/patients')['patients']
        kidney = self.get('/api/kidney/patients')['patients']
        self.assertEqual(len(diabetes), 40)
        self.assertEqual(len(kidney), 40)
        for p, d, k in zip(self.store.patients, diabetes, kidney):
            self.assertEqual(d['biomarker'], p['biomarker'])
            self.assertEqual(d['clinical_data'], p['clinical_data'])
            self.assertEqual(k['case_id'], p['kidney_case_id'])
            self.assertEqual(k['pool'], p['kidney_pool'])
            self.assertNotIn('case_id', d)
            for field in ['age', 'label', 'biomarker', 'clinical_data', 'gender']:
                self.assertNotIn(field, k)
        self.assertEqual(sum(p['slice_count'] for p in kidney), 3757)

    def test_every_ct_series_has_exact_sorted_source_slices(self):
        _, images, _ = build()
        for p in self.store.patients:
            code = p['patient_code']
            slices = self.get(f'/api/kidney/patients/{code}/ct')['slices']
            expected = sorted(i['slice_index'] for i in images if i['patient_code'] == code and i['img_type'] == 'kidney_ct')
            self.assertEqual([s['z'] for s in slices], expected)
            for s in (slices[0], slices[-1]):
                with urlopen(self.base + s['url']) as response:
                    self.assertEqual(response.headers['Content-Type'], 'image/png')
                    self.assertEqual(response.read(), self.store.files[s['url']].read_bytes())

    def test_unknown_paths_masks_and_traversal_are_not_served(self):
        for path in ['/api/kidney/patients/P999/ct', '/api/ct/P001/99999.png',
                     '/backend/data/patients.json', '/api/ct/../../backend/data/patients.json',
                     '/api/ct/%2e%2e/patients.json', '/data/kidney/data2d/masks/case_00000_z004.png']:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + path)
            self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base + '/api/diabetes/patients', data=b'{}'))
        self.assertEqual(error.exception.code, 405)


if __name__ == '__main__':
    unittest.main()
