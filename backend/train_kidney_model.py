"""Train the website's teaching kidney U-Net and store its checkpoint.

Run from the repository root:

    python backend/train_kidney_model.py

The checkpoint is written to ``artifacts/kidney/kidney_unet.pt`` and stays out of
Git, matching the team's rule that model weights live outside the repository.
The HTTP API picks it up automatically on the next start.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'backend') not in sys.path:
    sys.path.insert(0, str(ROOT / 'backend'))

import kidney_web_model as kidney  # noqa: E402

DEFAULT_OUTPUT = ROOT / 'artifacts/kidney/kidney_unet.pt'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=kidney.DEFAULT_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=kidney.DEFAULT_BATCH_SIZE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.time()

    def report(event: dict) -> None:
        print(
            f"epoch {event['epoch']:>2}/{event['total_epochs']}  "
            f"loss {event['train_loss']:.4f}  "
            f"kidney dice {event['validation_kidney_dice']}  "
            f"tumour dice {event['validation_tumour_dice']}",
            flush=True,
        )

    result = kidney.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        progress=report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kidney.save_checkpoint(args.output, result)

    summary = {key: value for key, value in result.items() if key != 'state_dict'}
    summary['elapsed_seconds'] = round(time.time() - started, 1)
    summary['checkpoint'] = str(args.output.relative_to(ROOT))
    metrics_path = args.output.with_suffix('.metrics.json')
    metrics_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    print('')
    print(f"selected epoch      : {summary['selected_epoch']}")
    print(f"test kidney dice    : {summary['test']['patient_mean_kidney_dice']}")
    print(f"test tumour dice    : {summary['test']['patient_mean_tumour_dice']}")
    print(f"elapsed             : {summary['elapsed_seconds']}s")
    print(f"checkpoint          : {summary['checkpoint']}")
    print(f"metrics             : {metrics_path.relative_to(ROOT)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
