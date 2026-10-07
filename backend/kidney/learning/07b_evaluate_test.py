#!/usr/bin/env python3
"""Step 7B: one recorded final-test execution with a frozen PNG checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from png_final_cli_common import (atomic_json, checked_output, digest_file,
                                  load_runtime, utc_now, write_new_json)


def run(args):
    if args.batch_size not in (1, 2) or not 1 <= args.cpu_threads <= 16:
        raise ValueError('Use batch size 1 or 2 and 1-16 CPU threads.')
    runtime = load_runtime()
    inspected = runtime.inspect_frozen_model(args.root, args.cohort, args.audit, args.run_dir, args.freeze)
    freeze_path = Path(inspected['freeze_path'])
    output = checked_output(args.output_dir,
                            (inspected['context']['root'], inspected['run_dir'], freeze_path.parent))
    access_path = freeze_path.parent / 'test_access.json'
    if access_path.exists() or access_path.is_symlink():
        raise ValueError('Final test was already opened for this freeze. Preserve test_access.json and its recorded output; do not rerun or choose another model using test results.')
    record = inspected['freeze_record']
    print('Frozen checkpoint: %s; selected epoch: %s' %
          (record['checkpoint_sha256'], record['selected_epoch']), flush=True)
    print('Final-test pool: %d patients / %d selected slices; no training or model selection.' %
          (record['patient_counts']['test'], record['selected_slice_counts']['test']), flush=True)
    if args.check_only:
        print('[CHECK ONLY] Frozen model and current inputs passed. No model initialization, PNG decoding, test opening or output writing.')
        return {'status': 'preflight_completed', 'model_initialized': False, 'pngs_decoded': 0,
                'test_opened': False, 'training_performed': False}
    state = {'schema_version': 'png_final_test_run_v1', 'status': 'reserved',
             'started_at_utc': utc_now(), 'checkpoint_sha256': record['checkpoint_sha256'],
             'freeze_sha256': inspected['freeze_sha256'],
             'freeze_fingerprint_sha256': record['freeze_fingerprint_sha256'],
             'output_dir': str(output), 'batch_size': args.batch_size, 'cpu_threads': args.cpu_threads,
             'final_test_opened': True, 'final_test_completed': False,
             'training_performed': False, 'model_selection_performed': False,
             'independent_prediction_performed': False, 'last_reported_progress': None}
    # Exclusive creation prevents two processes or another output name from opening this freeze twice.
    write_new_json(access_path, state)
    summary_path = output / 'evaluation_summary.json'
    owns_output = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        state['status'] = 'running'
        atomic_json(summary_path, state)
        atomic_json(access_path, state)

        def on_progress(event):
            value = {'recorded_at_utc': utc_now(), **event}
            with (output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(value, allow_nan=False) + '\n')
            state['last_reported_progress'] = value
            atomic_json(summary_path, state)
            print('[FINAL TEST PROGRESS] %s' % json.dumps(value, allow_nan=False), flush=True)

        report = runtime.evaluate_frozen_test(inspected, batch_size=args.batch_size,
                                              num_threads=args.cpu_threads, progress=on_progress)
        report_path = output / 'test_report.json'
        write_new_json(report_path, report)
        state.update(status='completed', finished_at_utc=utc_now(), final_test_completed=True,
                     test_report_sha256=digest_file(report_path),
                     artifact_sha256={'test_report.json': digest_file(report_path)},
                     patient_mean=report['patient_mean'])
        atomic_json(summary_path, state)
        atomic_json(access_path, state)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     finished_at_utc=utc_now(), error_type=type(error).__name__, error_message=str(error))
        # Preserve the original failure even when one evidence location is
        # unwritable. Never touch a directory created by another process.
        destinations = [access_path]
        if owns_output:
            destinations.append(summary_path)
        for destination in destinations:
            try:
                atomic_json(destination, state)
            except Exception as evidence_error:
                print('Could not record failure at %s: %s' %
                      (destination, evidence_error), file=sys.stderr, flush=True)
        raise
    print('\n[FROZEN PNG FINAL TEST COMPLETED]')
    print('Patients=%d; selected slices=%d' % (record['patient_counts']['test'], record['selected_slice_counts']['test']))
    print('Test patient means: %s' % json.dumps(report['patient_mean'], allow_nan=False))
    print('Test report: %s' % report_path)
    print('Selected exported-slice segmentation only. No training, model selection, independent prediction or patient diagnosis.')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 7B: evaluate the frozen PNG model on reserved final-test patients')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'))
    parser.add_argument('--cohort', type=Path, default=Path('learning_outputs/step5/project40_splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('learning_outputs/step5/png_data_audit.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('learning_outputs/step6b/run05'))
    parser.add_argument('--freeze', type=Path, default=Path('learning_outputs/step7a/run05/frozen_model.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step7b/run05_test'))
    parser.add_argument('--batch-size', type=int, choices=(1, 2), default=2)
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
