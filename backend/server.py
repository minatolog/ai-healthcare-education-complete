"""Local patient and model API. Python 3.10+.

``GET`` endpoints serve the stored JSON teaching dataset and CT images.
``POST`` endpoints run the real diabetes model: training a Random Forest for the
student's 20 selected patients and predicting one held-out patient.

Start it from the repository root with ``python backend/server.py``.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from model_service import ModelUnavailable, RequestError, make_model_services

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'backend/data'
MAX_BODY_BYTES = 100_000


class PatientStore:
    def __init__(self, directory=STORE):
        self.patients = json.loads((directory / 'patients.json').read_text(encoding='utf-8'))
        images = json.loads((directory / 'patient_images.json').read_text(encoding='utf-8'))
        self.ct = defaultdict(list)
        self.files = {}
        case_by_code = {p['patient_code']: p['kidney_case_id'] for p in self.patients}
        for image in images:
            if image['img_type'] != 'kidney_ct':
                continue
            code, z = image['patient_code'], image['slice_index']
            case = case_by_code[code]
            expected = ROOT / 'data/kidney/data2d/images' / f'{case}_z{z:03d}.png'
            if ROOT / image['file_path'] != expected or not expected.is_file():
                raise ValueError(f'Invalid CT record: {code}/{z}')
            url = f'/api/ct/{code}/{z}.png'
            if url in self.files:
                raise ValueError(f'Duplicate CT record: {url}')
            self.files[url] = expected
            self.ct[code].append({'z': z, 'url': url})
        for patient in self.patients:
            slices = self.ct[patient['patient_code']]
            slices.sort(key=lambda item: item['z'])
            if len(slices) != patient['provenance']['selected_slice_count']:
                raise ValueError(f'Incomplete CT series: {patient["patient_code"]}')

    def roster(self, mode):
        records = []
        for p in self.patients:
            code = p['patient_code']
            record = {'id': int(code[1:]), 'patient_code': code, 'name': p['name']}
            if mode == 'diabetes':
                record.update(age=p['age'], label=p['label'],
                              clinical_data=p['clinical_data'], biomarker=p['biomarker'],
                              pool='test' if record['id'] > 30 else 'development')
            else:
                record.update(case_id=p['kidney_case_id'], pool=p['kidney_pool'],
                              slice_count=len(self.ct[code]))
            records.append(record)
        return records


def make_server(host='127.0.0.1', port=8000, store=None, models=None):
    store = store or PatientStore()
    models = models if models is not None else make_model_services(store)

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.write_body(body)

        def write_body(self, body):
            """Send bytes, ignoring clients that navigated away mid-response."""

            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def read_json_body(self):
            raw_length = self.headers.get('Content-Length', '0')
            try:
                length = int(raw_length)
            except ValueError as error:
                raise RequestError('Content-Length must be an integer.') from error
            if length <= 0 or length > MAX_BODY_BYTES:
                raise RequestError(
                    f'Request body must be JSON between 1 and {MAX_BODY_BYTES} bytes.'
                )
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RequestError('Request body must be valid UTF-8 JSON.') from error
            if not isinstance(payload, dict):
                raise RequestError('Request body must be a JSON object.')
            return payload

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/health':
                return self.json_response(200, {'status': 'ok', 'storage': 'json', 'patients': len(store.patients)})
            if path == '/api/model/status':
                return self.json_response(200, {
                    'models': {mode: service.status() for mode, service in models.items()},
                })
            match = re.fullmatch(r'/api/(diabetes|kidney)/patients', path)
            if match:
                return self.json_response(200, {'patients': store.roster(match[1])})
            match = re.fullmatch(r'/api/kidney/patients/(P\d{3})/ct', path)
            if match and match[1] in store.ct:
                return self.json_response(200, {'patient_code': match[1], 'slices': store.ct[match[1]]})
            match = re.fullmatch(r'/api/kidney/patients/(P\d{3})/overlay/(\d+)\.png', path)
            if match:
                try:
                    body = models['kidney'].overlay(match[1], int(match[2]))
                except ModelUnavailable as error:
                    return self.json_response(503, error.as_dict())
                except RequestError as error:
                    return self.json_response(400, error.as_dict())
                except Exception as error:  # noqa: BLE001 - report model failures to the UI
                    return self.json_response(500, {'error': 'model_error', 'message': str(error)})
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'private, max-age=3600')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.write_body(body)
                return
            if path in store.files:
                try:
                    body = store.files[path].read_bytes()
                except OSError:
                    return self.json_response(404, {'error': 'CT image unavailable'})
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'private, max-age=3600')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.write_body(body)
                return
            self.json_response(404, {'error': 'Not found'})

        def do_POST(self):
            path = urlsplit(self.path).path
            match = re.fullmatch(r'/api/(diabetes|kidney)/(train|predict)', path)
            if not match:
                return self.json_response(405, {'error': 'Only model train and predict endpoints accept POST'})
            mode, operation = match[1], match[2]
            try:
                payload = self.read_json_body()
            except RequestError as error:
                return self.json_response(400, error.as_dict())
            service = models[mode]
            try:
                if operation == 'train':
                    result = service.train(payload.get('patient_ids'))
                else:
                    result = service.predict(payload.get('patient_code'))
            except ModelUnavailable as error:
                return self.json_response(503, error.as_dict())
            except RequestError as error:
                return self.json_response(400, error.as_dict())
            except Exception as error:  # noqa: BLE001 - report model failures to the UI
                return self.json_response(500, {'error': 'model_error', 'mode': mode,
                                                'operation': operation, 'message': str(error)})
            self.json_response(200, result)

        def log_message(self, fmt, *args):
            if getattr(self.server, 'quiet', False):
                return
            super().log_message(fmt, *args)

    class Server(ThreadingHTTPServer):
        daemon_threads = True

    return Server((host, port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--quiet', action='store_true', help='Do not log every request.')
    args = parser.parse_args()
    server = make_server(port=args.port)
    server.quiet = args.quiet
    print(f'Patient JSON API: http://127.0.0.1:{server.server_port}', flush=True)
    print('Model API: GET /api/model/status, POST /api/<mode>/train, POST /api/<mode>/predict', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
