"""Local, read-only patient API backed by JSON. Python 3.10+, no dependencies."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'backend/data'


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


def make_server(host='127.0.0.1', port=8000, store=None):
    store = store or PatientStore()

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/api/health':
                return self.json_response(200, {'status': 'ok', 'storage': 'json', 'patients': len(store.patients)})
            match = re.fullmatch(r'/api/(diabetes|kidney)/patients', path)
            if match:
                return self.json_response(200, {'patients': store.roster(match[1])})
            match = re.fullmatch(r'/api/kidney/patients/(P\d{3})/ct', path)
            if match and match[1] in store.ct:
                return self.json_response(200, {'patient_code': match[1], 'slices': store.ct[match[1]]})
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
                self.wfile.write(body)
                return
            self.json_response(404, {'error': 'Not found'})

        def do_POST(self):
            self.json_response(405, {'error': 'Patient data is read-only'})

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = make_server(port=args.port)
    print(f'Patient JSON API: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
