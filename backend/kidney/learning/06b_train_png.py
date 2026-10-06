#!/usr/bin/env python3
"""Step 6B: one-channel PNG training/validation using the immutable patient lock.

The student starts the real run explicitly. --check-only validates metadata and
the offline encoder cache without decoding CT/mask PNGs or updating a model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_runtime():
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    return importlib.import_module('kidney_segmentation.png_training')


def verify_offline_cache():
    from kidney_segmentation.pretrained import _verified_cache, REVISION, WEIGHTS_SHA256
    snapshot, _, _ = _verified_cache()
    return {'source_revision': REVISION, 'weights_sha256': WEIGHTS_SHA256,
            'snapshot': str(snapshot), 'download_attempted': False,
            'model_initialized': False, 'encoder_tensor_equality_checked_in_this_preflight': False}


def atomic_json(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.json.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
        if path.is_symlink():
            raise ValueError('The run evidence destination became a symlink.')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def reserve_output(requested, root):
    output = Path(os.path.abspath(str(requested.expanduser())))
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Run directories must not contain symlinks.')
    if output == root or root in output.parents:
        raise ValueError('Save training outputs outside kits_project.')
    if output.exists():
        raise ValueError('The run directory already exists and will not be replaced. Use an explicit new directory such as run02.')
    output.mkdir(parents=True, exist_ok=False)
    return output


def reference_review_record(path, context):
    """Record preview provenance without inventing a passed human review."""
    if not path.is_file():
        return {'report_available': False, 'manual_review_status': 'not_recorded'}
    content = path.read_bytes()
    report = json.loads(content)
    if (not isinstance(report, dict) or report.get('schema_version') != 'training_reference_review_v1'
            or report.get('cohort_fingerprint_sha256') != context['cohort']['cohort_fingerprint_sha256']
            or report.get('audit_fingerprint_sha256') != context['audit']['audit_fingerprint_sha256']
            or report.get('contains_model_prediction') is not False):
        raise ValueError('The reference-review report differs from the locked training inputs.')
    return {'report_available': True, 'report_sha256': hashlib.sha256(content).hexdigest(),
            'manual_review_status': report.get('manual_review_status', 'not_recorded'),
            'anatomical_alignment_verified': report.get('anatomical_alignment_verified', False)}


def settings(args):
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Epochs, batch size and finite learning rate must be positive.')
    if not 0 <= args.seed < 2**63 or not 1 <= args.cpu_threads <= 16 or args.device != 'cpu':
        raise ValueError('Use CPU, 1-16 threads and a non-negative seed below 2**63 for this first PNG entry.')
    return {'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
            'seed': args.seed, 'device': args.device, 'cpu_threads': args.cpu_threads,
            'training_sampling': 'all exported training rows, shuffled once per epoch; no label-based sampling',
            'resume': False}


def run(args):
    config = settings(args)
    runtime = load_runtime()
    root = args.root.expanduser().resolve()
    context = runtime.load_training_context(root, args.cohort, args.audit)
    pools = context['cohort']['pools']
    counts = {role: len(records) for role, records in pools.items()}
    slices = {role: sum(record['selected_slice_count'] for record in records) for role, records in pools.items()}
    cache = verify_offline_cache()
    review = reference_review_record(args.reference_review, context)
    base = {
        'schema_version': 'png_training_run_v1', 'scope': 'selected_exported_2d_png_segmentation',
        'patient_counts': counts, 'selected_slice_counts': slices, 'configuration': config,
        'cohort_fingerprint_sha256': context['cohort']['cohort_fingerprint_sha256'],
        'audit_fingerprint_sha256': context['audit']['audit_fingerprint_sha256'],
        'reference_review': review, 'visual_review_note': args.visual_review_note,
        'clinical_alignment_verified': False, 'full_scan_coverage_verified': False,
        'final_test_performed': False, 'independent_prediction_performed': False,
    }
    print('Locked PNG inputs: train=%d patients/%d slices; validation=%d patients/%d slices.' %
          (counts['train'], slices['train'], counts['validation'], slices['validation']), flush=True)
    print('Final test and independent prediction remain metadata only for this command.', flush=True)
    if args.check_only:
        result = {**base, 'status': 'preflight_completed', 'offline_cache': cache,
                  'model_initialized': False, 'ct_mask_pngs_decoded': 0,
                  'optimizer_steps': 0, 'training_performed': False,
                  'pixel_integrity_note': 'Metadata/cache checks only; selected source hashes are checked by the training dataset during a real run.'}
        print('[CHECK ONLY] Cohort/manifest/split bindings and pinned offline encoder files passed. No training or CT/mask decoding.', flush=True)
        return result

    output = reserve_output(args.output_dir, root)
    started = utc_now()
    current = {**base, 'status': 'running', 'started_at_utc': started,
               'formal_training_completed': False, 'last_reported_progress': None}
    atomic_json(output / 'run_summary.json', current)
    progress_path = output / 'progress.jsonl'

    def on_progress(event):
        record = {'recorded_at_utc': utc_now(), **event}
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with progress_path.open('a', encoding='utf-8') as handle:
            handle.write(encoded + '\n')
        current['last_reported_progress'] = record
        atomic_json(output / 'run_summary.json', current)
        print('[PROGRESS] ' + encoded, flush=True)

    try:
        fitted = runtime.train_png(root, args.cohort, args.audit, epochs=args.epochs,
                                   batch_size=args.batch_size, lr=args.learning_rate, seed=args.seed,
                                   device=args.device, num_threads=args.cpu_threads, progress=on_progress,
                                   visual_review_note=args.visual_review_note)
        checkpoint = runtime.save_png_checkpoint(output / 'best_model.pt', fitted['model'], fitted['metadata'])
        atomic_json(output / 'training_history.json', {
            'schema_version': 'png_training_history_v1', 'scope': base['scope'],
            'configuration': config, 'history': fitted['history']})
        atomic_json(output / 'validation_report.json', fitted['best_validation_report'])
        artifacts = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                     for name in ('best_model.pt', 'training_history.json', 'validation_report.json')}
        current.update(status='completed', finished_at_utc=utc_now(), formal_training_completed=True,
                       epochs_completed=len(fitted['history']), checkpoint=checkpoint, artifact_sha256=artifacts,
                       best_epoch=fitted['metadata'].get('best_epoch'),
                       validation_patient_mean=fitted['best_validation_report']['patient_mean'],
                       pretrained_encoder_proof=fitted['proof'])
        atomic_json(output / 'run_summary.json', current)
    except KeyboardInterrupt:
        current.update(status='interrupted', finished_at_utc=utc_now(), formal_training_completed=False,
                       interruption_note='The run did not complete. This version does not resume an interrupted run; use a new output directory.')
        atomic_json(output / 'run_summary.json', current)
        print('\nRun interrupted. Last reported progress was preserved in: %s' % output, flush=True)
        raise
    except Exception as error:
        current.update(status='failed', finished_at_utc=utc_now(), formal_training_completed=False,
                       error_type=type(error).__name__, error_message=str(error))
        atomic_json(output / 'run_summary.json', current)
        raise
    print('\n[LOCKED PNG TRAINING COMPLETED]')
    print('Epochs=%d; best validation checkpoint: %s' % (len(fitted['history']), output / 'best_model.pt'))
    print('Validation patient means: %s' % json.dumps(current['validation_patient_mean'], allow_nan=False))
    print('Run evidence: %s' % output)
    print('Metrics describe selected exported slices. No final test, independent prediction or patient diagnosis was performed.')
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 6B: train a one-channel U-Net on the locked PNG training pool and validate on the ten validation patients')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'))
    parser.add_argument('--cohort', type=Path, default=Path('learning_outputs/step5/project40_splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('learning_outputs/step5/png_data_audit.json'))
    parser.add_argument('--reference-review', type=Path, default=Path('learning_outputs/step6a/training_reference_review.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step6b/run01'))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--visual-review-note', default=None, help='Optional description of your limited training-preview inspection; does not certify clinical annotation quality')
    parser.add_argument('--check-only', action='store_true', help='Metadata and offline-cache checks only; no CT/mask decode or model initialization')
    args = parser.parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
