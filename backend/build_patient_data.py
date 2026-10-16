"""Build a reproducible backend JSON store from the explicit project40 manifest.

Only Python's standard library is required. No source datasets are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'kidney-diabetes-teaching40-v1'
MANIFEST = Path('data/kidney/project40_manifest.csv')
DIABETES = Path('data/synthetic/diabetes_patients.csv')
PATIENT_FIELDS = ('patient_code', 'name', 'age', 'gender', 'label', 'clinical_data',
                  'biomarker', 'kidney_case_id', 'kidney_pool', 'dataset_version', 'provenance')
IMAGE_FIELDS = ('patient_code', 'img_type', 'slice_index', 'file_path', 'description', 'image_metadata')
FIRST_NAMES = ('Alex Avery Bailey Blake Cameron Casey Charlie Dakota Devon Drew Eden '
               'Elliot Emery Finley Frankie Harper Hayden Jamie Jesse Jordan Jules Kai '
               'Kendall Lane Logan Morgan Parker Peyton Quinn Reese Remy Riley River '
               'Robin Rowan Sage Sam Skyler Taylor Tyler').split()
LAST_NAMES = ('Adams Allen Baker Bennett Brooks Brown Campbell Carter Clark Collins '
              'Cooper Davis Edwards Evans Fisher Foster Green Hall Harris Hayes Hill '
              'Howard Jackson James Kelly King Lee Lewis Martin Miller Mitchell Moore '
              'Morgan Parker Reed Scott Smith Taylor Turner Walker').split()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diabetes_records(root):
    # Load only the existing validation module, without importing ML dependencies.
    spec = importlib.util.spec_from_file_location(
        '_patient_seed_diabetes_schema', ROOT / 'services/diabetes_model/schema.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sorted((asdict(r) for r in module.load_reference_cohort(root / DIABETES)),
                  key=lambda row: row['patient_id'])


def validate_manifest(rows, root):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        case = row['case']
        if not re.fullmatch(r'case_\d{5}', case):
            raise ValueError(f'Invalid case ID: {case}')
        z = int(row['z'])
        if z < 0 or (case, z) in seen:
            raise ValueError(f'Duplicate or invalid slice: {case}/{z}')
        seen.add((case, z))
        if row['pool'] not in {'train', 'validation', 'test'}:
            raise ValueError(f'Invalid pool: {case}')
        for key, directory in [('image', 'images'), ('mask', 'masks')]:
            relative = PurePosixPath(row[key])
            expected = PurePosixPath(directory) / f'{case}_z{z:03d}.png'
            if relative != expected:
                raise ValueError(f'Invalid {key} path: {relative}')
            if not (root / 'data/kidney/data2d' / relative).is_file():
                raise ValueError(f'Missing file: {relative}')
        for flag, pixels in [('has_kidney', 'kidney_px'), ('has_tumor', 'tumor_px')]:
            count = int(row[pixels])
            if count < 0 or int(row[flag]) not in (0, 1) or int(row[flag]) != int(count > 0):
                raise ValueError(f'Invalid reference counts: {case}/{z}')
        if int(row['n_slices']) <= z:
            raise ValueError(f'Slice outside source volume: {case}/{z}')
        groups[case].append(row)
    for case, slices in groups.items():
        if len({r['pool'] for r in slices}) != 1:
            raise ValueError(f'Patient spans multiple pools: {case}')
    counts = Counter(slices[0]['pool'] for slices in groups.values())
    if len(groups) != 40 or len(rows) != 3757 or counts != {'train': 20, 'validation': 10, 'test': 10}:
        raise ValueError(f'Unexpected project40 counts: {len(groups)} patients, {len(rows)} pairs, {counts}')
    return {case: sorted(slices, key=lambda r: int(r['z'])) for case, slices in sorted(groups.items())}


def build(root=ROOT):
    groups = validate_manifest(read_csv(root / MANIFEST), root)
    diabetes = diabetes_records(root)
    rng = random.Random(9900)
    first, last = FIRST_NAMES.copy(), LAST_NAMES.copy()
    rng.shuffle(first)
    rng.shuffle(last)
    patients, images, mapping = [], [], []
    # Sorted case order, NOT CSV row order: reordering slices cannot change identity.
    for index, ((case, slices), clinical) in enumerate(zip(groups.items(), diabetes)):
        code = clinical['patient_id']
        patient = dict(
            patient_code=code, name=f'{first[index]} {last[index]}',
            age=clinical['age_years'], gender=None,
            label='diabetes_positive' if clinical['diabetes_label'] else 'diabetes_negative',
            clinical_data={k: clinical[k] for k in ('bmi', 'systolic_bp_mmhg', 'family_history')},
            biomarker={k: clinical[k] for k in ('glucose_mmol_l', 'hba1c_percent')},
            kidney_case_id=case, kidney_pool=slices[0]['pool'], dataset_version=VERSION,
            provenance={
                'identity_is_fictional': True, 'pairing_is_synthetic': True,
                'pairing_method': 'Sorted kidney case IDs paired with sorted diabetes patient IDs',
                'pairing_note': 'Independent sources; not a matched real patient or a multimodal research cohort.',
                'name_seed': 9900, 'diabetes_is_synthetic': True, 'kidney_images_are_synthetic': False,
                'age_source': 'synthetic_diabetes', 'diabetes_source_patient_id': code,
                'kidney_source_patient_id': f'kits19:{case}',
                'diabetes_source': DIABETES.as_posix(), 'kidney_source': MANIFEST.as_posix(),
                'selected_slice_count': len(slices),
            })
        patients.append(patient)
        mapping.append({k: patient[k] for k in ('patient_code', 'name', 'age', 'gender', 'kidney_case_id', 'kidney_pool')} |
                       {'diabetes_source_patient_id': code, 'diabetes_label': clinical['diabetes_label'],
                        'ct_slice_count': len(slices), 'reference_mask_count': len(slices),
                        'pairing_is_synthetic': 'true'})
        for row in slices:
            for key, img_type in [('image', 'kidney_ct'), ('mask', 'kidney_reference_mask')]:
                is_mask = key == 'mask'
                metadata = {'source_case_id': case, 'source_pool': row['pool'],
                            'source_slice_index': int(row['z']), 'is_synthetic': False,
                            'role': 'reference_target' if is_mask else 'model_input',
                            'source_n_slices': int(row['n_slices']),
                            'source_voxel_spacing_zyx_mm': [float(row[k]) for k in ('sp_z', 'sp_y', 'sp_x')],
                            'spacing_note': 'Original volume spacing; not resized PNG spacing.',
                            'width': 256, 'height': 256, 'mode': 'L'}
                if is_mask:
                    metadata['reference_labels'] = {k: int(row[k]) for k in
                                                   ('has_kidney', 'has_tumor', 'kidney_px', 'tumor_px')}
                    metadata['label_values'] = {'0': 'background', '1': 'kidney', '2': 'tumour'}
                images.append(dict(patient_code=code, img_type=img_type, slice_index=int(row['z']),
                                   file_path=(PurePosixPath('data/kidney/data2d') / row[key]).as_posix(),
                                   description=('Reference segmentation mask; not a model input or patient diagnosis.'
                                                if is_mask else 'Processed KiTS19 axial CT slice.'),
                                   image_metadata=metadata))
    return patients, images, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate source files without writing outputs.')
    args = parser.parse_args()
    patients, images, mapping = build()
    summary = {'dataset_version': VERSION, 'patients': len(patients),
               'ct_images': sum(r['img_type'] == 'kidney_ct' for r in images),
               'reference_masks': sum(r['img_type'] == 'kidney_reference_mask' for r in images),
               'patient_pools': dict(Counter(p['kidney_pool'] for p in patients)),
               'slice_pools': dict(Counter(i['image_metadata']['source_pool'] for i in images if i['img_type'] == 'kidney_ct')),
               'gender': 'Unspecified; JSON null for all patients',
               'pairing': 'Synthetic educational pairing of independent sources',
               'sources_sha256': {p.as_posix(): digest(ROOT / p) for p in (MANIFEST, DIABETES)}}
    if not args.check:
        output = ROOT / 'backend/data'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'patients.json').write_text(json.dumps(patients, indent=2) + '\n', encoding='utf-8')
        # One image object per line keeps a full-image catalogue reviewable in Git.
        (output / 'patient_images.json').write_text(
            '[\n' + ',\n'.join(json.dumps(i, separators=(',', ':')) for i in images) + '\n]\n',
            encoding='utf-8')
        (output / 'import_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        with (output / 'patient_mapping.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
            writer.writeheader()
            writer.writerows(mapping)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
