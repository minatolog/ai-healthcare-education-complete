"""Verify the model API links the browser workflow to the real diabetes model."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import make_server, ROOT


TRAINING_IDS = [f'P{index:03d}' for index in range(1, 21)]


class ModelApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(port=0, models=None)
        cls.server.quiet = True
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

    def post(self, path, payload):
        request = Request(
            self.base + path,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        with urlopen(request) as response:
            return response.status, json.load(response)

    def post_error(self, path, payload):
        with self.assertRaises(HTTPError) as error:
            self.post(path, payload)
        return error.exception.code, json.load(error.exception)

    def test_status_reports_which_models_can_run(self):
        models = self.get('/api/model/status')['models']
        self.assertTrue(models['diabetes']['available'])
        self.assertEqual(models['diabetes']['supported_operations'], ['train', 'predict'])
        self.assertEqual(models['diabetes']['committed_model']['best_tree_count'], 60)

        kidney = models['kidney']
        checkpoint_exists = (ROOT / 'artifacts/kidney/kidney_unet.pt').is_file()
        self.assertEqual(kidney['available'], checkpoint_exists and kidney['torch_available'])
        if not kidney['available']:
            self.assertTrue(kidney['reasons'])

    def test_training_returns_real_measured_metrics(self):
        status, session = self.post('/api/diabetes/train', {'patient_ids': TRAINING_IDS})
        self.assertEqual(status, 200)
        self.assertEqual(session['status'], 'trained')
        self.assertEqual(session['selection']['training_patient_ids'], TRAINING_IDS)
        self.assertEqual(len(session['split']['training']), 20)
        self.assertEqual(len(session['split']['evaluation']), 10)
        self.assertEqual(len(session['split']['held_out']), 10)
        self.assertFalse(set(session['split']['evaluation']) & set(TRAINING_IDS))

        points = session['training_curve']['points']
        self.assertEqual(len(points), 20)
        self.assertEqual(points[-1]['trees_built'], 100)
        self.assertTrue(all(0.0 <= point['validation_accuracy'] <= 1.0 for point in points))

        model = session['model']
        self.assertIn(model['best_tree_count'], [point['trees_built'] for point in points])
        self.assertEqual(
            model['best_tree_count'],
            max(points, key=lambda point: (point['validation_accuracy'], -point['trees_built']))['trees_built'],
        )

        evaluation = session['evaluation']
        self.assertEqual(evaluation['sample_count'], 10)
        self.assertEqual(len(evaluation['predictions']), 10)
        correct = sum(1 for row in evaluation['predictions'] if row['correct'])
        self.assertAlmostEqual(evaluation['metrics']['accuracy'], correct / 10, places=4)
        self.assertEqual(len(session['held_out_evaluation']['predictions']), 10)
        self.assertEqual(len(session['feature_importance']['features']), 6)
        self.assertTrue(session['educational_only'])

    def test_prediction_uses_the_session_model_and_explains_it(self):
        self.post('/api/diabetes/train', {'patient_ids': TRAINING_IDS})
        status, prediction = self.post('/api/diabetes/predict', {'patient_code': 'P031'})
        self.assertEqual(status, 200)
        self.assertEqual(prediction['patient_id'], 'P031')
        self.assertEqual(prediction['source'], 'session_model')
        self.assertIn(prediction['predicted_label'], [0, 1])
        self.assertIn(prediction['actual_label'], [0, 1])
        self.assertTrue(0.0 <= prediction['estimated_diabetes_probability'] <= 1.0)
        self.assertEqual(len(prediction['feature_effects']), 6)
        names = {effect['feature_name'] for effect in prediction['feature_effects']}
        self.assertEqual(names, {
            'age_years', 'glucose_mmol_l', 'hba1c_percent',
            'bmi', 'systolic_bp_mmhg', 'family_history',
        })

    def test_invalid_requests_are_rejected(self):
        for payload in [
            {'patient_ids': TRAINING_IDS[:10]},
            {'patient_ids': TRAINING_IDS[:19] + [TRAINING_IDS[0]]},
            {'patient_ids': TRAINING_IDS[:19] + ['P999']},
            {'patient_ids': ['P031'] + TRAINING_IDS[1:20]},
            {'patient_ids': 'P001'},
        ]:
            code, body = self.post_error('/api/diabetes/train', payload)
            self.assertEqual(code, 400, body)
            self.assertEqual(body['error'], 'invalid_request')
        code, body = self.post_error('/api/diabetes/predict', {'patient_code': 'P999'})
        self.assertEqual(code, 400)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base + '/api/diabetes/predict',
                            data=b'not json',
                            headers={'Content-Type': 'application/json'}))
        self.assertIn(error.exception.code, (400, 405))

    def test_kidney_model_either_serves_real_output_or_explains_why_not(self):
        kidney = self.get('/api/model/status')['models']['kidney']
        if not kidney['available']:
            for path, payload in [
                ('/api/kidney/train', {'patient_ids': TRAINING_IDS}),
                ('/api/kidney/predict', {'patient_code': 'P031'}),
            ]:
                code, body = self.post_error(path, payload)
                self.assertEqual(code, 503)
                self.assertEqual(body['error'], 'model_unavailable')
                self.assertEqual(body['mode'], 'kidney')
            return

        kidney_ids = [p['patient_code'] for p in self.get('/api/kidney/patients')['patients'] if p['pool'] == 'train']
        status, session = self.post('/api/kidney/train', {'patient_ids': kidney_ids})
        self.assertEqual(status, 200)
        self.assertEqual(session['mode'], 'kidney')
        self.assertEqual(session['training_mode'], 'recorded_checkpoint')
        self.assertEqual(session['selection']['training_size'], 20)
        self.assertEqual(session['selection']['evaluation_size'], 10)
        self.assertGreaterEqual(len(session['training_curve']['points']), 1)
        self.assertIsNotNone(session['evaluation']['metrics']['kidney_dice'])
        self.assertIsNotNone(session['evaluation']['metrics']['tumour_dice'])
        self.assertEqual(len(session['evaluation']['predictions']), 10)

        status, prediction = self.post('/api/kidney/predict', {'patient_code': 'P031'})
        self.assertEqual(status, 200)
        self.assertEqual(prediction['patient_id'], 'P031')
        self.assertEqual(prediction['source'], 'checkpoint')
        self.assertIn(prediction['predicted_label'], [0, 1])
        self.assertGreater(prediction['slice_count'], 0)
        self.assertIn(prediction['actual_label'], [0, 1])

    def test_kidney_does_not_silently_accept_a_different_training_group(self):
        if not self.get('/api/model/status')['models']['kidney']['available']:
            self.skipTest('Kidney runtime unavailable')
        for ids in (TRAINING_IDS, ['P001'] * 20, 'P001', []):
            code, body = self.post_error('/api/kidney/train', {'patient_ids': ids})
            self.assertEqual(code, 400)
            self.assertEqual(body['error'], 'invalid_request')

    def test_unknown_overlay_slice_is_a_client_error(self):
        if not self.get('/api/model/status')['models']['kidney']['available']:
            self.skipTest('Kidney runtime unavailable')
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base + '/api/kidney/patients/P031/overlay/999999.png')
        self.assertEqual(error.exception.code, 400)

    def test_kidney_overlay_returns_a_real_png(self):
        if not self.get('/api/model/status')['models']['kidney']['available']:
            self.skipTest('Kidney runtime unavailable')
        z = self.get('/api/kidney/patients/P006/ct')['slices'][0]['z']
        with urlopen(self.base + f'/api/kidney/patients/P006/overlay/{z}.png') as response:
            self.assertEqual(response.headers['Content-Type'], 'image/png')
            body = response.read()
        self.assertTrue(body.startswith(bytes([137,80,78,71,13,10,26,10])))
        self.assertGreater(len(body), 100)


if __name__ == '__main__':
    unittest.main()
