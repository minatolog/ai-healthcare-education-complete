#!/usr/bin/env python3
"""Step 6A: render two reference pairs per locked training patient, no model.

Reference labels guide this training-only visual QA selection. They must never
be used to choose evaluation or inference slices. Original files stay read only.
"""
import argparse
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

POLICY = 'max_tumour_opposite_foreground_quartile_v1'


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def load_helpers():
    starter = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(starter))
    cohort = importlib.import_module('kidney_segmentation.cohort')
    spec = importlib.util.spec_from_file_location('step6a_reference_helpers', starter / 'learning/02_inspect_2d_pair.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return cohort, reference


def quantile(values, fraction):
    """Linear quantile over actual axial indices, including irregular gaps."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def select_rows(rows):
    """Return two distinct training QA rows; never define evaluation sampling."""
    if len(rows) < 2 or len({int(row['z']) for row in rows}) != len(rows):
        raise ValueError('Training review requires at least two distinct exported slices per patient.')
    if any(int(row['tumor_px']) > 0 for row in rows):
        first = min(rows, key=lambda row: (-int(row['tumor_px']), int(row['z'])))
    else:
        first = min(rows, key=lambda row: (-int(row['kidney_px']), int(row['z'])))
    first_reason = ('largest_reference_tumour_area' if int(first['tumor_px']) > 0
                    else 'largest_reference_kidney_area_no_tumour_rows')
    if int(first['kidney_px']) + int(first['tumor_px']) == 0:
        first_reason = 'no_reference_foreground_available_background_fallback'
    foreground = [row for row in rows if int(row['kidney_px']) + int(row['tumor_px']) > 0]
    axial = [int(row['z']) for row in (foreground or rows)]
    fraction = 0.75 if int(first['z']) <= quantile(axial, 0.5) else 0.25
    target = quantile(axial, fraction)
    candidates = [row for row in foreground if int(row['z']) != int(first['z'])]
    second_reason = 'opposite_foreground_axial_quartile'
    if not candidates:
        candidates = [row for row in rows if int(row['z']) != int(first['z'])]
        second_reason = 'distinct_exported_background_context_fallback'
    second = min(candidates, key=lambda row: (abs(int(row['z']) - target), int(row['z'])))
    return [(first, {'reason': first_reason}),
            (second, {'reason': second_reason, 'requested_axial_quantile': fraction,
                      'requested_original_slice_index': target})]


def checked_output(requested, root):
    output = Path(os.path.abspath(str(requested.expanduser())))
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Review output directories must not contain symlinks.')
    if output == root or root in output.parents:
        raise ValueError('Save reference previews outside kits_project.')
    if output.exists() and not output.is_dir():
        raise ValueError('The review output path is an existing file.')
    return output


def checked_destination(output, relative):
    path = output / relative
    if any(part.is_symlink() for part in (path, *path.parents)) or output not in path.resolve().parents:
        raise ValueError('Reference output paths must remain inside the output directory without symlinks.')
    if path.exists() and not path.is_file():
        raise ValueError('An existing reference output is not a regular file.')
    return path


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def training_asset_path(reference, data, row, kind, folder):
    path = reference.checked_asset(data, row[kind], folder, row['case'], int(row['z']))
    lexical = data / row[kind]
    if path != lexical or any(part.is_symlink() for part in (lexical, *lexical.parents)):
        raise ValueError('Training reference assets must not be symlinks or resolve to another patient path.')
    return path


def save_without_replacing(path, content):
    """Create atomically or reuse identical evidence; preserve edited reports."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing review evidence differs; preserve it and use a versioned output directory: %s' % path)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.review.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError('A review output was created concurrently; it was not replaced.') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def patient_preview(reference, patient_id, slices):
    canvas = Image.new('RGB', (832, 900), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), 'TRAINING ONLY | %s | Two reference slices | No model' % patient_id,
              fill='black', font=ImageFont.load_default(size=17))
    for offset, (ct, mask, z, _) in enumerate(slices):
        panels = reference.reference_images(ct, mask)
        canvas.paste(reference.labelled_preview(panels, patient_id, z), (0, 36 + offset * 432))
    return canvas


def overview(reference, decoded):
    canvas = Image.new('RGB', (1536, 840), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), 'TRAINING REFERENCE OVERVIEW | Green: kidney | Red: tumour | No model prediction',
              fill='black', font=ImageFont.load_default(size=21))
    draw.text((16, 42), 'Open every full-size patient preview to inspect alignment. Thumbnails do not establish label correctness.',
              fill='black', font=ImageFont.load_default(size=16))
    for index, (patient_id, slices) in enumerate(decoded):
        x, y = 16 + (index % 5) * 304, 78 + (index // 5) * 182
        draw.text((x, y), patient_id, fill='black', font=ImageFont.load_default(size=17))
        for column, (ct, mask, z, _) in enumerate(slices):
            overlay = reference.reference_images(ct, mask)[2]
            canvas.paste(overlay.resize((136, 136), Image.Resampling.NEAREST), (x + column * 144, y + 24))
            draw.text((x + column * 144, y + 162), 'z=%d' % z, fill='black', font=ImageFont.load_default(size=13))
    draw.text((16, 818), 'Selected training slices for visual QA only; manual review is pending. Original CT/mask files remain unchanged.',
              fill='black', font=ImageFont.load_default(size=15))
    return canvas


def review(args):
    root = args.root.expanduser().resolve()
    data = (root / 'data2d').resolve()
    if not root.is_dir() or root not in data.parents:
        raise ValueError('Use the original kits_project root with a contained data2d directory.')
    output = checked_output(args.output_dir, root)
    cohort_module, reference = load_helpers()
    cohort_bytes, audit_bytes = args.cohort.read_bytes(), args.audit.read_bytes()
    lock, audit = json.loads(cohort_bytes), json.loads(audit_bytes)
    cohort_module.validate_cohort(lock, audit)
    manifest = (data / 'manifest.csv').resolve()
    split_path = (data / 'splits.json').resolve()
    if data not in manifest.parents or data not in split_path.parents:
        raise ValueError('Manifest and original split must remain inside data2d.')
    manifest_bytes, split_bytes = manifest.read_bytes(), split_path.read_bytes()
    if sha256(manifest_bytes) != audit['manifest_sha256'] or sha256(split_bytes) != audit['original_split_sha256']:
        raise ValueError('Original manifest/split changed after Step 5; existing review bindings cannot be used.')

    # Build the complete training-only plan before opening any PNG.
    training = lock['pools']['train']
    allowlist = {record['patient_id'] for record in training}
    rows_by_patient = {patient_id: [] for patient_id in allowlist}
    reader = csv.DictReader(io.StringIO(manifest_bytes.decode('utf-8-sig'), newline=''))
    for row in reader:
        if row['case'] in allowlist:
            rows_by_patient[row['case']].append(row)
    plan = []
    for record in training:
        rows = rows_by_patient[record['patient_id']]
        if len(rows) != record['selected_slice_count']:
            raise ValueError('Training patient row count differs from the lock.')
        plan.append((record, select_rows(rows)))
    for _, selections in plan:
        for row, _ in selections:
            for kind, folder in (('image', 'images'), ('mask', 'masks')):
                training_asset_path(reference, data, row, kind, folder)
    print('Training-only reference review: 20 patients, two selected pairs per patient.', flush=True)
    print('Reference labels guide visual QA selection only. No model or held-out PNG will be read.', flush=True)

    decoded, patients, files = [], [], {}
    for record, selections in plan:
        patient_id, slices, sample_records = record['patient_id'], [], []
        for row, reason in selections:
            z = int(row['z'])
            if row['case'] not in allowlist or not 0 <= z < int(row['n_slices']):
                raise ValueError('Preview identity must be a bounded slice of a locked training patient.')
            observed = {}
            arrays = {}
            for kind, folder in (('image', 'images'), ('mask', 'masks')):
                path = training_asset_path(reference, data, row, kind, folder)
                pixels, byte_hash = reference.read_greyscale_png(path)
                facts = audit['assets'][row[kind]]
                pixel_hash = sha256(pixels.tobytes(order='C'))
                if (facts['kind'] != kind or facts['patient_id'] != patient_id or facts['slice_index'] != z
                        or byte_hash != facts['byte_sha256'] or pixel_hash != facts['decoded_pixel_sha256']):
                    raise ValueError('Selected %s content/identity changed after Step 5: %s' % (kind, row[kind]))
                observed[kind] = {'path': row[kind], 'byte_sha256': byte_hash, 'decoded_pixel_sha256': pixel_hash,
                                  'matches_saved_audit': True}
                arrays[kind] = pixels
            counts, _, _ = reference.verify_mask(arrays['mask'], row)
            slices.append((arrays['image'], arrays['mask'], z, reason))
            sample_records.append({'patient_id': patient_id, 'original_slice_index': z,
                                   'selection': reason, 'source_assets': observed, 'reference_label_counts': counts,
                                   'coordinate_space': 'processed_png_256x256'})
        filename = 'patient_previews/%s_reference_preview.png' % patient_id
        content = png_bytes(patient_preview(reference, patient_id, slices))
        files[filename] = content
        patients.append({**record, 'samples': sample_records, 'preview_path': filename,
                         'preview_sha256': sha256(content), 'manual_review_status': 'pending'})
        decoded.append((patient_id, slices))
    files['training_review_overview.png'] = png_bytes(overview(reference, decoded))
    report = {
        'schema_version': 'training_reference_review_v1', 'sampling_policy_version': POLICY,
        'sampling_scope': 'training visual alignment review only; not training subsampling, evaluation or inference sampling',
        'selection_uses_reference_labels': True, 'loaded_patient_pool': 'train',
        'patient_count': len(patients), 'preview_pair_count': sum(len(item['samples']) for item in patients),
        'patient_ids': [item['patient_id'] for item in patients],
        'cohort_fingerprint_sha256': lock['cohort_fingerprint_sha256'],
        'audit_fingerprint_sha256': audit['audit_fingerprint_sha256'],
        'manifest_sha256': sha256(manifest_bytes), 'original_split_sha256': sha256(split_bytes),
        'cohort_file_sha256': sha256(cohort_bytes), 'audit_file_sha256': sha256(audit_bytes),
        'contains_model_prediction': False, 'model_run': False, 'training_started': False,
        'manual_review_status': 'pending', 'anatomical_alignment_verified': False,
        'source_patient_aliases_verified': False, 'full_scan_coverage_verified': False,
        'label_map': {'0': 'background', '1': 'kidney', '2': 'tumour'},
        'overview_path': 'training_review_overview.png',
        'overview_sha256': sha256(files['training_review_overview.png']), 'patients': patients,
        'limitations': ['Two reference slices cannot certify every slice of a training patient.',
                        'PNG appearance cannot establish physical spacing, original orientation or clinical annotation correctness.',
                        'Automatic decoding/rendering does not record a passed human review.'],
    }
    files['training_reference_review.json'] = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if (args.cohort.read_bytes() != cohort_bytes or args.audit.read_bytes() != audit_bytes
            or manifest.read_bytes() != manifest_bytes or split_path.read_bytes() != split_bytes):
        raise ValueError('A bound input changed during review preparation; no previews were saved.')
    destinations = {name: checked_destination(output, name) for name in files}
    # Preflight all existing files before writing any; human-edited evidence stays intact.
    for name, path in destinations.items():
        if path.exists() and path.read_bytes() != files[name]:
            raise ValueError('Existing review evidence differs; preserve it and use a versioned output directory: %s' % path)
    for name, path in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_without_replacing(path, files[name])
    print('\n[TRAINING REFERENCE PREVIEWS]')
    print('Patients=20; decoded CT/mask pairs=%d; validation/test/prediction PNGs read=0' % report['preview_pair_count'])
    print('Patient previews: %s' % (output / 'patient_previews'))
    print('Overview: %s' % (output / 'training_review_overview.png'))
    print('Report: %s' % (output / 'training_reference_review.json'))
    print('manual_review_status=pending. Inspect the full-size patient previews for visible pairing/alignment issues.')
    print('No model ran. Original CT/mask, manifest, split and cohort files were not modified.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 6A: create training-only reference CT/mask previews bound to the Step 5 cohort')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'), help='Original kits_project root')
    parser.add_argument('--cohort', type=Path, default=Path('learning_outputs/step5/project40_splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('learning_outputs/step5/png_data_audit.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step6a'))
    args = parser.parse_args(argv)
    try:
        review(args)
    except (OSError, ValueError, KeyError, TypeError, ImportError, csv.Error) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
