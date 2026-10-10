"""Verify the seed in a temporary local PostgreSQL cluster, then stop it.

Uses installed PostgreSQL executables and Python's standard library only.
Does not connect to or modify an existing PostgreSQL instance.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile

from build_patient_seed import ROOT, PATIENT_FIELDS, IMAGE_FIELDS, build, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pg-bin', type=Path, default=Path(shutil.which('psql') or 'psql').parent)
    args = parser.parse_args()
    suffix = '.exe' if os.name == 'nt' else ''
    binaries = {name: str((args.pg_bin / (name + suffix)).resolve())
                for name in ('initdb', 'pg_ctl', 'psql')}
    for binary in binaries.values():
        if not Path(binary).is_file():
            raise SystemExit(f'PostgreSQL executable not found: {binary}')
    base = (ROOT / '.local').resolve()
    base.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='postgres-check-', dir=base)).resolve()
    cluster = temporary / 'cluster'
    password_file = temporary / 'password.txt'
    password = secrets.token_urlsafe(32)
    password_file.write_text(password, encoding='utf-8')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(PGHOST='127.0.0.1', PGPORT=str(port), PGUSER='seedcheck',
               PGDATABASE='postgres', PGPASSWORD=password, PGCLIENTENCODING='UTF8', PGCONNECT_TIMEOUT='5')
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

    def run(name, *arguments, expected=0):
        # Windows background servers can inherit pipes and keep communicate() open.
        # Files allow subprocess.run() to wait for pg_ctl itself to exit.
        stdout_path, stderr_path = temporary / 'command-out.txt', temporary / 'command-err.txt'
        with stdout_path.open('w', encoding='utf-8') as out, stderr_path.open('w', encoding='utf-8') as err:
            result = subprocess.run([binaries[name], *map(str, arguments)], env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                    creationflags=creationflags, timeout=90)
        stdout = stdout_path.read_text(encoding='utf-8', errors='replace')
        stderr = stderr_path.read_text(encoding='utf-8', errors='replace')
        if result.returncode != expected:
            raise RuntimeError(f'{name} failed ({result.returncode}): {stdout}\n{stderr}')
        return stdout

    def sql(statement, expected=0):
        return run('psql', '-X', '-w', '-A', '-t', '-v', 'ON_ERROR_STOP=1', '-c', statement, expected=expected)

    def load_seed(expected=0):
        return run('psql', '-X', '-w', '-v', 'ON_ERROR_STOP=1', '-f', ROOT / 'database/seed_patients.sql', expected=expected)

    started = False
    try:
        run('initdb', '-D', cluster, '-U', 'seedcheck', '-A', 'scram-sha-256',
            '--pwfile', password_file, '--encoding=UTF8', '--locale=C')
        password_file.unlink()
        run('pg_ctl', '-D', cluster, '-l', temporary / 'postgres.log',
            '-o', f'-h 127.0.0.1 -p {port}', '-w', 'start')
        started = True
        run('psql', '-X', '-w', '-v', 'ON_ERROR_STOP=1', '-f', ROOT / 'database/schema.sql')
        load_seed()
        patients, images, _ = build()
        pfields = ','.join(PATIENT_FIELDS)
        actual = json.loads(sql(f'SELECT json_agg(r ORDER BY patient_code) FROM (SELECT {pfields} FROM patient) r'))
        if actual != patients:
            raise AssertionError('Patient field-by-field readback failed')
        actual_images = json.loads(sql('SELECT json_agg(r ORDER BY patient_code,slice_index,img_type) FROM '
                                      '(SELECT p.patient_code,i.img_type,i.slice_index,i.file_path,i.description,i.image_metadata '
                                      'FROM patient_image i JOIN patient p ON p.id=i.patient_id) r'))
        if actual_images != sorted(images, key=lambda x: (x['patient_code'], x['slice_index'], x['img_type'])):
            raise AssertionError('Image field-by-field readback failed')
        before = sql('SELECT count(*),max(id) FROM patient_image')
        load_seed()
        if sql('SELECT count(*),max(id) FROM patient_image') != before:
            raise AssertionError('Repeat import changed image count or IDs')
        sql("UPDATE patient SET name='Edited Name' WHERE patient_code='P001'")
        load_seed(expected=3)
        if sql("SELECT name FROM patient WHERE patient_code='P001'").strip() != 'Edited Name':
            raise AssertionError('Conflict import overwrote existing data')
        sql(f"UPDATE patient SET name='{patients[0]['name']}' WHERE patient_code='P001'")
        sql("INSERT INTO patient_image(patient_id,img_type,slice_index,file_path,description,image_metadata) "
            "VALUES (-1,'kidney_ct',0,'data/kidney/data2d/nonexistent.png','Invalid test','{}')", expected=1)
        if sql('SELECT count(*) FROM training_run').strip() != '0':
            raise AssertionError('Unexpected training results')
        report = {'status': 'passed', 'postgres_version': sql('SHOW server_version').strip(),
                  'database_scope': 'Temporary isolated verification cluster; stopped after validation',
                  'patients_verified': len(patients), 'image_records_verified': len(images),
                  'checks': ['Every patient field equals generated source data',
                             'Every CT and mask record equals the manifest-derived seed',
                             'Repeat import preserves counts and existing IDs',
                             'Conflicting import fails and preserves edited records',
                             'Foreign key rejects an unknown patient', 'No fabricated training runs'],
                  'files_sha256': {p: digest(ROOT / p) for p in ('database/schema.sql', 'database/seed_patients.sql')}}
    finally:
        if started or (cluster / 'postmaster.pid').exists():
            run('pg_ctl', '-D', cluster, '-m', 'fast', '-w', 'stop')
        # Delete only the unique cluster this invocation created inside this workspace.
        if temporary.parent != base or not temporary.name.startswith('postgres-check-'):
            raise RuntimeError('Refusing cleanup outside the validation workspace')
        shutil.rmtree(temporary)
    (ROOT / 'data/patients/validation_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
