#!/usr/bin/env python3
"""Step 5: check selected PNG data and create or verify a patient cohort lock.

No model, optimizer, augmentation, validation/test score or image preview runs.
The original manifest, split and CT/mask files are read only.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

OUTPUTS = ('png_data_audit.json', 'project40_splits.json', 'step5_summary.json', 'latest_audit_attempt.json')


def load_helpers():
    source = Path(__file__).resolve().parents[1]
    for name in ('png_audit.py', 'cohort.py'):
        if not (source / 'kidney_segmentation' / name).is_file():
            raise ValueError('Missing Step 5 helper: %s' % name)
    sys.path.insert(0, str(source))
    try:
        auditor = importlib.import_module('kidney_segmentation.png_audit')
        cohort = importlib.import_module('kidney_segmentation.cohort')
    except ImportError as error:
        raise ValueError('Use the starter environment with NumPy and Pillow installed: %s' % error) from None
    return auditor, cohort


def checked_output_dir(requested, data_root):
    output = Path(os.path.abspath(str(requested.expanduser())))
    if any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError('Output directories must not contain symlinks.')
    output = output.resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError('Save Step 5 files outside kits_project; original data stays unchanged.')
    if output.exists() and not output.is_dir():
        raise ValueError('The output directory is an existing file.')
    for name in OUTPUTS:
        target = output / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError('Step 5 output must be a regular file: %s' % name)
    return output


def registered_exposures(starter, additional):
    # Earlier steps explicitly inspected and updated case_00000.
    patients = {'case_00000', *additional}
    reports = starter / 'learning_outputs'
    if reports.is_dir():
        for path in sorted(reports.rglob('*.json')):
            if path.name not in ('inspection.json', 'loader_report.json', 'training_step.json'):
                continue
            if path.is_symlink() or reports.resolve() not in path.resolve().parents:
                raise ValueError('A development report resolves outside learning_outputs.')
            record = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                raise ValueError('A saved development report must contain a JSON object: %s' % path.name)
            for block, field in (('development_inspection', 'case_inspected_for_development'),
                                 ('development_exposure', 'case_inspected_for_development')):
                entry = record.get(block, {})
                if not isinstance(entry, dict):
                    raise ValueError('Invalid %s object in development report %s.' % (block, path.name))
                value = entry.get(field)
                if value is not None:
                    patients.add(value)
            checkpoint = record.get('checkpoint_metadata', {})
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('development_exposure', {}), dict):
                raise ValueError('Invalid checkpoint exposure object in development report %s.' % path.name)
            value = checkpoint.get('development_exposure', {}).get('patient_id')
            if value is not None:
                patients.add(value)
    if any(not isinstance(patient, str) or not re.fullmatch(r'case_[0-9]{5}', patient) for patient in patients):
        raise ValueError('Registered development exposure must use case_XXXXX IDs.')
    return sorted(patients)


def atomic_json(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.json.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write('\n')
        if path.is_symlink():
            raise ValueError('An output file became a symlink.')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_lock(path, content):
    # Never replace a cohort lock, even if a second process creates it first.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.cohort.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise ValueError('The cohort lock already exists. It was not replaced; rerun to verify it.') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def progress(*values):
    print('[DATA CHECK] ' + ' '.join(str(value) for value in values), flush=True)


def run(args):
    if not 0 <= args.seed < 2**63:
        raise ValueError('--seed must be a non-negative integer below 2**63.')
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError('The --root kits_project directory does not exist.')
    output = checked_output_dir(args.output_dir, root)
    starter = Path(__file__).resolve().parents[1]
    exposures = registered_exposures(starter, args.exposed_case)
    auditor, cohort = load_helpers()
    lock_path = output / 'project40_splits.json'
    # Read an existing lock before running the longer pixel check.
    existing_bytes = lock_path.read_bytes() if lock_path.exists() else None
    existing = json.loads(existing_bytes.decode('utf-8')) if existing_bytes is not None else None
    print('Read-only PNG audit. No model or held-out performance evaluation will run.', flush=True)
    print('Registered development patients: %s' % ', '.join(exposures), flush=True)
    audit = auditor.audit_png_dataset(root, progress=progress)
    output = checked_output_dir(args.output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    if audit['status'] != 'passed':
        # Keep the old evidence paired with an existing lock unchanged.
        atomic_json(output / ('latest_audit_attempt.json' if existing is not None else 'png_data_audit.json'), audit)
        atomic_json(output / 'step5_summary.json', {
            'step': '05_audit_and_lock_cohort', 'status': 'data_check_failed',
            'cohort_lock_created': False, 'existing_lock_preserved': existing is not None,
            'audit_status': audit['status'], 'model_run': False,
            'message': 'Resolve the actual audit errors before creating or using a cohort lock.',
        })
        report_name = 'latest_audit_attempt.json' if existing is not None else 'png_data_audit.json'
        raise ValueError('PNG data checks failed. Read %s; no new cohort lock was created.' % report_name)
    if existing is not None:
        try:
            cohort.validate_cohort(existing, audit)
            if existing['seed'] != args.seed or existing['required_patient_counts']['prediction'] != 5:
                raise ValueError('Requested seed/prediction count differs from the existing lock.')
            if set(existing['exposed_patient_ids']) != set(exposures):
                raise ValueError('Development exposure changed since the cohort was locked.')
            if lock_path.read_bytes() != existing_bytes:
                raise ValueError('The cohort lock changed while the audit was running. Rerun to verify its current content.')
        except ValueError as error:
            atomic_json(output / 'latest_audit_attempt.json', audit)
            atomic_json(output / 'step5_summary.json', {
                'step': '05_audit_and_lock_cohort', 'status': 'existing_lock_rejected',
                'existing_lock_preserved': True, 'bound_audit_preserved': True,
                'latest_audit_fingerprint_sha256': audit['audit_fingerprint_sha256'],
                'reason': str(error), 'model_run': False,
            })
            raise ValueError('Existing cohort lock was preserved but cannot be used: %s' % error) from None
        payload, action = existing, 'verified_existing_lock'
    else:
        payload = cohort.make_cohort(audit, seed=args.seed, exposed_patient_ids=exposures, prediction_count=5)
        cohort.validate_cohort(payload, audit)
        create_lock(lock_path, payload)
        action = 'created_new_lock'
    atomic_json(output / 'png_data_audit.json', audit)
    counts = {role: len(records) for role, records in payload['pools'].items()}
    slice_counts = {role: sum(record['selected_slice_count'] for record in records)
                    for role, records in payload['pools'].items()}
    summary = {
        'step': '05_audit_and_lock_cohort', 'status': 'completed', 'lock_action': action,
        'patient_counts': counts, 'selected_slice_counts': slice_counts,
        'seed': args.seed, 'registered_exposed_patient_ids': exposures,
        'audit_fingerprint_sha256': audit['audit_fingerprint_sha256'],
        'cohort_fingerprint_sha256': payload['cohort_fingerprint_sha256'],
        'model_run': False, 'formal_training_performed': False,
        'held_out_performance_evaluated': False,
        'scope': 'selected_manifest_2d_png_slices',
        'manual_anatomical_alignment_verified': False,
        'external_patient_identity_linkage_verified': False,
        'development_registration_note': 'Also register any cases inspected outside saved learning reports with --exposed-case.',
        'next_step': 'Review training-only alignment/source provenance, then implement the locked-cohort PNG training entry.',
    }
    atomic_json(output / 'step5_summary.json', summary)
    print('\n[PATIENT COHORT LOCK]')
    for role in ('train', 'validation', 'test', 'prediction'):
        print('%s: %d patients; %d selected slices' % (role, counts[role], slice_counts[role]))
    print('Lock action: %s; seed=%d' % (action, args.seed))
    print('All slices of one patient follow the same pool; registered development cases are training only.')
    print('No patient/source identity or meaningful duplicate-content component crosses pools.')
    print('Cohort lock: %s' % lock_path)
    print('Audit: %s' % (output / 'png_data_audit.json'))
    print('Original splits.json, manifest and CT/mask files were not modified.')
    print('Machine file/pixel checks do not establish anatomical correctness or full-scan coverage.')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 5: audit selected PNG pairs and create or verify a patient-level 20/10/10 cohort plus five independent prediction cases')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'), help='Original kits_project directory')
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step5'), help='Audit and immutable cohort lock directory outside kits_project')
    parser.add_argument('--seed', type=int, default=42, help='Cohort selection seed; cannot replace an existing lock')
    parser.add_argument('--exposed-case', action='append', default=[], help='Additional patient already inspected or trained during development; repeat as needed')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
