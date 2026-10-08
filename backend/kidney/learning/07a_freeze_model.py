#!/usr/bin/env python3
"""Step 7A: record the selected checkpoint without reading test PNGs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from png_final_cli_common import checked_output, digest_file, load_runtime, write_new_json


def run(args):
    runtime = load_runtime()
    inspected = runtime.inspect_completed_run(args.root, args.cohort, args.audit, args.run_dir)
    output = checked_output(args.output_dir, (inspected['context']['root'], inspected['run_dir']))
    record = runtime.make_freeze_record(inspected)
    runtime.validate_freeze_record(record, inspected)
    print('Selected epoch: %s; checkpoint SHA256: %s' %
          (record['selected_epoch'], record['checkpoint_sha256']), flush=True)
    print('Reserved final test: %d patients / %d selected slices.' %
          (record['patient_counts']['test'], record['selected_slice_counts']['test']), flush=True)
    if args.check_only:
        print('[CHECK ONLY] Completed run and current data bindings passed. No model initialization, PNG decoding or output writing.')
        return {'status': 'preflight_completed', 'model_initialized': False, 'pngs_decoded': 0,
                'training_performed': False, 'final_test_performed': False, 'freeze_record': record}
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'frozen_model.json'
    write_new_json(path, record)
    # Verify the written snapshot against the current source before reporting it frozen.
    runtime.inspect_frozen_model(args.root, args.cohort, args.audit, args.run_dir, path)
    print('\n[MODEL FROZEN]')
    print('Freeze record: %s' % path)
    print('Checkpoint bytes remain at their original run path. No training, final test or PNG decoding.')
    return {'status': 'frozen', 'freeze_path': str(path), 'freeze_sha256': digest_file(path),
            'model_initialized': False, 'pngs_decoded': 0, 'training_performed': False,
            'final_test_performed': False, 'freeze_record': record}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Step 7A: freeze the selected real PNG checkpoint and evidence')
    parser.add_argument('--root', type=Path, default=Path('../kits_project'))
    parser.add_argument('--cohort', type=Path, default=Path('learning_outputs/step5/project40_splits.json'))
    parser.add_argument('--audit', type=Path, default=Path('learning_outputs/step5/png_data_audit.json'))
    parser.add_argument('--run-dir', type=Path, default=Path('learning_outputs/step6b/run05'))
    parser.add_argument('--output-dir', type=Path, default=Path('learning_outputs/step7a/run05'))
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
