#!/usr/bin/env python3
"""Step 8: frozen-model CT-only inference on the separate prediction pool."""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from png_final_cli_common import atomic_json, checked_output, digest_file, utc_now


def load_runtime():
    root = Path(__file__).resolve().parents[1]
    source = root/'kidney-segmentation' if (root/'kidney-segmentation').is_dir() else root
    sys.path.insert(0, str(source))
    return importlib.import_module('kidney_segmentation.png_prediction')


def inspect_test_receipt(inspected, test_dir):
    """Require the completed test evidence; its scores never select predictions."""
    directory = Path(test_dir).expanduser().absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError('Final-test evidence paths must not contain symlinks.')
    paths = {'summary': directory/'evaluation_summary.json',
             'report': directory/'test_report.json',
             'access': Path(inspected['freeze_path']).parent/'test_access.json'}
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError('Complete Step 7B before this learning step; preserve its recorded evidence.')
    content = {name: path.read_bytes() for name, path in paths.items()}
    records = {name: json.loads(value) for name, value in content.items()}
    if any(not isinstance(record, dict) for record in records.values()):
        raise ValueError('Final-test evidence must contain JSON objects.')
    expected_checkpoint = inspected['checkpoint_sha256']
    expected_freeze = inspected['freeze_sha256']
    report_hash = digest_file(paths['report'])
    for name in ('summary', 'access'):
        record = records[name]
        if (record.get('schema_version') != 'png_final_test_run_v1'
                or record.get('status') != 'completed'
                or record.get('final_test_completed') is not True
                or record.get('checkpoint_sha256') != expected_checkpoint
                or record.get('freeze_sha256') != expected_freeze
                or record.get('test_report_sha256') != report_hash
                or record.get('output_dir') != str(directory.resolve())):
            raise ValueError('Completed final-test receipt differs from this frozen model/report.')
    report = records['report']
    if (report.get('schema_version') != 'png_frozen_test_v1'
            or report.get('status') != 'completed'
            or report.get('checkpoint_sha256') != expected_checkpoint
            or report.get('freeze_sha256') != expected_freeze
            or report.get('model_state_unchanged') is not True
            or report.get('training_performed') is not False
            or report.get('model_selection_performed') is not False
            or report.get('optimizer_steps') != 0):
        raise ValueError('Final-test report does not describe unchanged frozen-model evaluation.')
    return {'directory': directory.resolve(), 'paths': paths,
            'sha256': {name: digest_file(path) for name, path in paths.items()}}


def check_test_receipt_unchanged(receipt):
    for name, path in receipt['paths'].items():
        if digest_file(path) != receipt['sha256'][name]:
            raise ValueError('Completed final-test evidence changed during prediction.')


def run(args):
    if args.batch_size not in (1, 2) or not 1 <= args.cpu_threads <= 16:
        raise ValueError('Use batch size 1 or 2 and 1-16 CPU threads.')
    runtime = load_runtime()
    inspected = runtime.inspect_prediction_inputs(args.root, args.cohort, args.audit,
                                                   args.run_dir, args.freeze)
    receipt = inspect_test_receipt(inspected, args.test_dir)
    output = checked_output(args.output_dir, (inspected['context']['root'], inspected['run_dir'],
                            Path(inspected['freeze_path']).parent, receipt['directory']))
    inputs = inspected['prediction_inputs']
    print('Independent prediction: %d patients / %d exported CT slices.' %
          (inputs['patient_count'], inputs['slice_count']), flush=True)
    print('CT-only model input. Reference-mask PNGs will not be read; no performance scores or diagnosis.', flush=True)
    if args.check_only:
        print('[CHECK ONLY] Frozen model, completed test receipt and prediction identities passed. No model initialization, PNG decoding or output writing.')
        return {'status': 'preflight_completed', 'model_initialized': False, 'ct_pngs_decoded': 0,
                'reference_mask_pngs_read': 0, 'training_performed': False}
    state = {'schema_version': 'png_prediction_run_v1', 'status': 'running',
             'started_at_utc': utc_now(), 'checkpoint_sha256': inspected['checkpoint_sha256'],
             'freeze_sha256': inspected['freeze_sha256'], 'output_dir': str(output),
             'batch_size': args.batch_size, 'cpu_threads': args.cpu_threads,
             'completed_test_receipt_sha256': receipt['sha256'],
             'training_performed': False, 'model_selection_performed': False,
             'reference_mask_pngs_read': 0, 'aggregate_performance_evaluation': False,
             'last_reported_progress': None}
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output/'prediction_summary.json'
    try:
        # Core requires an empty new directory. Its first event starts the
        # caller-owned status/progress files after the directory check.
        def on_progress(event):
            value = {'recorded_at_utc': utc_now(), **event}
            with (output/'progress.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(value, allow_nan=False)+'\n')
            state['last_reported_progress'] = value
            atomic_json(summary_path, state)
            print('[PREDICTION PROGRESS] %s' % json.dumps(value, allow_nan=False), flush=True)

        report = runtime.predict_frozen_pool(inspected, output, batch_size=args.batch_size,
                                              num_threads=args.cpu_threads, progress=on_progress)
        report_path = output/'prediction_report.json'
        if json.loads(report_path.read_text()) != report:
            raise ValueError('Saved prediction report differs from the completed return value.')
        check_test_receipt_unchanged(receipt)
        state.update(status='completed', finished_at_utc=utc_now(),
                     artifact_sha256={'prediction_report.json': digest_file(report_path)},
                     original_test_evidence_unchanged=True)
        atomic_json(summary_path, state)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     finished_at_utc=utc_now(), error_type=type(error).__name__, error_message=str(error))
        try:
            atomic_json(summary_path, state)
        except Exception as evidence_error:
            print('Could not record prediction failure: %s' % evidence_error, file=sys.stderr, flush=True)
        raise
    print('\n[FROZEN CT-ONLY PREDICTION COMPLETED]')
    print('Patients=%d; CT slices=%d; reference-mask PNGs read=0' %
          (inputs['patient_count'], inputs['slice_count']))
    print('Prediction assets: %s' % output)
    print('Report: %s' % report_path)
    print('Model segmentation outputs on selected exported slices. No training, test rerun, reference metrics or patient diagnosis.')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 8: frozen-model inference on the separate prediction CT pool')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'))
    parser.add_argument('--cohort', type=Path, default=Path('learning_outputs/step5/project40_splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('learning_outputs/step5/png_data_audit.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('learning_outputs/step6b/run05'))
    parser.add_argument('--freeze', type=Path, default=Path('learning_outputs/step7a/run05/frozen_model.json'))
    parser.add_argument('--test-dir', type=Path, default=Path('learning_outputs/step7b/run05_test'))
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step8/run05_prediction'))
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
