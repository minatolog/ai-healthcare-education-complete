#!/usr/bin/env python3
"""Step 6C: reference-guided train/validation prediction previews, not testing."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

POLICY = 'max_reference_tumour_distinct_kidney_axial_median_v1'


def digest(content):
    return hashlib.sha256(content).hexdigest()


def load_runtime():
    root = Path(__file__).resolve().parents[1]
    source = root / 'kidney-segmentation' if (root / 'kidney-segmentation').is_dir() else root
    sys.path.insert(0, str(source))
    return importlib.import_module('kidney_segmentation.png_training')


def load_checked_checkpoint(runtime, path, check_only=False):
    if check_only:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('schema_version') != runtime.SCHEMA:
            raise ValueError('Use a completed PNG segmentation checkpoint, not Step 4 or volume weights.')
        metadata = payload['metadata']
        config = runtime._validate_metadata(metadata)
        if runtime._state_hash(payload['state_dict']) != metadata['state_fingerprint_sha256']:
            raise ValueError('Checkpoint tensor fingerprint differs from its metadata.')
        return None, config, metadata
    return runtime.load_png_checkpoint(path, device='cpu')


def validate_binding(context, config, metadata, summary, checkpoint_hash):
    if (summary.get('schema_version') != 'png_training_run_v1'
            or summary.get('status') != 'completed'
            or summary.get('formal_training_completed') is not True):
        raise ValueError('Step 6C requires a completed real Step 6B run.')
    if (summary.get('artifact_sha256', {}).get('best_model.pt') != checkpoint_hash
            or summary.get('checkpoint', {}).get('sha256') != checkpoint_hash):
        raise ValueError('Checkpoint bytes differ from the completed run evidence.')
    if metadata.get('data_mode') != 'real':
        raise ValueError('This student entry rejects synthetic fixture checkpoints.')
    if (metadata.get('bound_file_sha256') != context['file_sha256']
            or config != context['config']
            or metadata['cohort']['cohort_fingerprint_sha256'] != context['cohort']['cohort_fingerprint_sha256']
            or metadata['audit']['audit_fingerprint_sha256'] != context['audit']['audit_fingerprint_sha256']
            or summary.get('cohort_fingerprint_sha256') != context['cohort']['cohort_fingerprint_sha256']
            or summary.get('audit_fingerprint_sha256') != context['audit']['audit_fingerprint_sha256']):
        raise ValueError('Checkpoint/run evidence differs from the current locked inputs.')


def select_indices(dataset):
    """Two fixed reference-guided diagnostic slices per patient, never scoring a pool."""
    selected = []
    for patient in dataset.patient_ids:
        rows = [(index, row) for index, row in enumerate(dataset.rows) if row['case'] == patient]
        if len(rows) < 2:
            raise ValueError('Each preview patient needs at least two distinct exported slices.')
        tumour = any(int(row['tumor_px']) > 0 for _, row in rows)
        area = 'tumor_px' if tumour else 'kidney_px'
        first = min(rows, key=lambda item: (-int(item[1][area]), int(item[1]['z'])))
        selected.append((first[0], {'reason': 'largest_reference_tumour_area' if tumour
                                  else 'largest_reference_kidney_area_no_tumour'}))
        kidney_rows = [(index, row) for index, row in rows if int(row['kidney_px']) > 0]
        target = float(np.median([int(row['z']) for _, row in (kidney_rows or rows)]))
        candidates = [item for item in kidney_rows if item[0] != first[0]]
        reason = 'distinct_reference_kidney_axial_median'
        if not candidates:
            candidates = [item for item in rows if item[0] != first[0]]
            reason = 'distinct_exported_axial_median_fallback'
        second = min(candidates, key=lambda item: (abs(int(item[1]['z']) - target), int(item[1]['z'])))
        selected.append((second[0], {'reason': reason, 'requested_original_slice_index': target}))
    return selected


def checked_output(requested, root, run_dir):
    output = Path(os.path.abspath(str(requested.expanduser())))
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Preview output paths must not contain symlinks.')
    if any(output == source or source in output.parents for source in (root, run_dir)):
        raise ValueError('Save previews outside the original data and Step 6B run directories.')
    if output.exists():
        raise ValueError('Preview output already exists; preserve it and use a new review directory.')
    return output


def render_preview_overlay(ct, mask):
    """Render processed PNG pixels directly, without importing volume/HU tools."""
    if ct.shape != mask.shape or ct.shape != (256, 256) or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError('Preview overlays require aligned 256x256 three-class PNG labels.')
    rgb = np.repeat(np.asarray(ct, dtype=np.uint8)[..., None], 3, axis=-1).astype(np.float32)
    colors = np.array([[0, 0, 0], [50, 200, 100], [255, 80, 80]], dtype=np.float32)
    active = mask > 0
    rgb[active] = 0.6 * rgb[active] + 0.4 * colors[mask[active].astype(np.int64)]
    return Image.fromarray(rgb.round().astype(np.uint8))


def patient_preview(patient, role, samples):
    canvas = Image.new('RGB', (832, 704), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((12, 8), '%s | %s | REFERENCE-GUIDED DIAGNOSTIC PREVIEW' % (role.upper(), patient), fill='black', font=font)
    draw.text((12, 32), 'Green: kidney | Red: tumour | Model overlay is segmentation, not Grad-CAM', fill='black', font=font)
    for index, sample in enumerate(samples):
        y = 66 + index * 304
        draw.text((12, y), 'Original slice z=%d | Reference tumour=%d px | Predicted tumour=%d px' %
                  (sample['z'], int((sample['reference'] == 2).sum()), int((sample['prediction'] == 2).sum())), fill='black', font=font)
        panels = [Image.fromarray(sample['ct']), render_preview_overlay(sample['ct'], sample['reference']),
                  render_preview_overlay(sample['ct'], sample['prediction'])]
        for column, (title, panel) in enumerate(zip(('CT', 'REFERENCE OVERLAY', 'MODEL PREDICTION'), panels)):
            x = 12 + column * 276
            draw.text((x, y + 22), title, fill='black', font=font)
            canvas.paste(panel, (x, y + 44))
    draw.text((12, 678), 'Two selected slices per patient. Use Step 6B reports for full validation metrics.', fill='black', font=font)
    return canvas


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(args):
    if not 1 <= args.cpu_threads <= 16:
        raise ValueError('Use 1-16 CPU threads.')
    runtime = load_runtime()
    root = args.root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    output = checked_output(args.output_dir, root, run_dir)
    context = runtime.load_training_context(root, args.cohort, args.audit)
    summary_bytes = (run_dir / 'run_summary.json').read_bytes()
    summary = json.loads(summary_bytes)
    if not isinstance(summary, dict):
        raise ValueError('The Step 6B summary must be a JSON object.')
    checkpoint = run_dir / 'best_model.pt'
    checkpoint_hash = digest(checkpoint.read_bytes())
    # Reject incomplete runs or changed weights before deserializing a model.
    if (summary.get('status') != 'completed' or summary.get('formal_training_completed') is not True
            or summary.get('artifact_sha256', {}).get('best_model.pt') != checkpoint_hash
            or summary.get('checkpoint', {}).get('sha256') != checkpoint_hash):
        raise ValueError('A completed run with matching checkpoint hashes is required.')
    model, config, metadata = load_checked_checkpoint(runtime, checkpoint, args.check_only)
    validate_binding(context, config, metadata, summary, checkpoint_hash)
    report = {'schema_version': 'png_prediction_review_v1',
              'status': 'preflight_completed' if args.check_only else 'running',
              'scope': 'reference_guided_train_validation_diagnostic_previews_only',
              'selection_policy': POLICY, 'selection_uses_reference_labels': True,
              'checkpoint_sha256': checkpoint_hash, 'checkpoint_path': str(checkpoint),
              'run_summary_sha256': digest(summary_bytes),
              'cohort_fingerprint_sha256': context['cohort']['cohort_fingerprint_sha256'],
              'audit_fingerprint_sha256': context['audit']['audit_fingerprint_sha256'],
              'decoded_pairs': {'train': 0, 'validation': 0, 'test': 0, 'prediction': 0},
              'model_run': False, 'training_performed': False, 'optimizer_steps': 0,
              'final_test_performed': False, 'independent_prediction_performed': False,
              'grad_cam_performed': False, 'clinical_alignment_verified': False,
              'full_scan_coverage_verified': False, 'manual_review_status': 'pending',
              'aggregate_performance_evaluation': False, 'patients': [],
              'legend': {'kidney': 'green', 'tumour': 'red'},
              'limitations': ['Reference-guided examples cannot establish blind or aggregate model performance.',
                              'Use the complete Step 6B validation report for validation metrics.']}
    if args.check_only:
        runtime._check_bindings(context)
        if digest(checkpoint.read_bytes()) != checkpoint_hash or (run_dir / 'run_summary.json').read_bytes() != summary_bytes:
            raise ValueError('Bound training evidence changed during preflight.')
        print('[CHECK ONLY] Current inputs and completed PNG checkpoint passed. No model initialization, PNG decoding or output writing.')
        return report
    datasets = {role: runtime.AuditedPNGDataset(context, role) for role in ('train', 'validation')}
    plans = {role: select_indices(dataset) for role, dataset in datasets.items()}
    before_state = runtime._state_hash(model.state_dict())
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / 'predictions_review.json'
    write_report(report_path, report)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.cpu_threads)
        with torch.inference_mode():
            for role, dataset in datasets.items():
                for patient in dataset.patient_ids:
                    samples, records = [], []
                    for index, selection in plans[role]:
                        row = dataset.rows[index]
                        if row['case'] != patient:
                            continue
                        item = dataset[index]
                        # The model receives CT only. Targets are used for display after inference.
                        logits = model(item['input'].unsqueeze(0))
                        if tuple(logits.shape) != (1, 3, 256, 256) or not torch.isfinite(logits).all():
                            raise ValueError('Expected finite three-class 256x256 prediction logits.')
                        predicted = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                        reference = item['target'].cpu().numpy().astype(np.uint8)
                        ct = np.rint((item['input'][0].cpu().numpy() * config.std + config.mean) * 255).astype(np.uint8)
                        z = int(item['slice_index'])
                        asset_dir = output / role / patient
                        asset_dir.mkdir(parents=True, exist_ok=True)
                        name = 'z%03d_predicted_mask.png' % z
                        Image.fromarray(predicted).save(asset_dir / name)
                        records.append({'original_slice_index': z, 'selection': selection,
                                        'reference_label_counts': item['mask_label_counts'],
                                        'prediction_label_counts': {str(k): int((predicted == k).sum()) for k in (0, 1, 2)},
                                        'source_sha256': item['source_sha256'],
                                        'predicted_mask_path': (asset_dir / name).relative_to(output).as_posix(),
                                        'predicted_mask_sha256': digest((asset_dir / name).read_bytes())})
                        samples.append({'ct': ct, 'reference': reference, 'prediction': predicted, 'z': z})
                        report['decoded_pairs'][role] += 1
                    preview_path = output / role / patient / 'prediction_comparison.png'
                    patient_preview(patient, role, samples).save(preview_path)
                    report['patients'].append({'patient_id': patient, 'pool': role, 'samples': records,
                                               'preview_path': preview_path.relative_to(output).as_posix(),
                                               'preview_sha256': digest(preview_path.read_bytes())})
                    report['model_run'] = True
                    write_report(report_path, report)
                    print('[PREVIEW] %s %s; slices=%d' % (role, patient, len(samples)), flush=True)
        runtime._check_bindings(context)
        if (digest(checkpoint.read_bytes()) != checkpoint_hash
                or (run_dir / 'run_summary.json').read_bytes() != summary_bytes
                or runtime._state_hash(model.state_dict()) != before_state):
            raise ValueError('Bound inputs, checkpoint or model state changed during preview inference.')
        expected = {role: 2 * len(dataset.patient_ids) for role, dataset in datasets.items()}
        if any(report['decoded_pairs'][role] != count for role, count in expected.items()):
            raise ValueError('Every selected diagnostic pair must be processed exactly once.')
        report.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      model_state_unchanged=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error_type=type(error).__name__, error_message=str(error))
        raise
    finally:
        torch.set_num_threads(previous_threads)
        write_report(report_path, report)
    print('\n[PREDICTION REVIEW COMPLETED]')
    print('Decoded pairs: train=%d; validation=%d; test=0; prediction=0' %
          (report['decoded_pairs']['train'], report['decoded_pairs']['validation']))
    print('Previews: %s' % output)
    print('Report: %s' % report_path)
    print('Reference-guided diagnostic examples. No training, final test or aggregate performance evaluation.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 6C: inspect reference-guided train/validation segmentation predictions')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'))
    parser.add_argument('--cohort', type=Path, default=Path('learning_outputs/step5/project40_splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('learning_outputs/step5/png_data_audit.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('learning_outputs/step6b/run01'))
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step6c/run01'))
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
