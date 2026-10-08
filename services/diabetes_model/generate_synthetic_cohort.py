"""Generate the reproducible 40-patient synthetic diabetes cohort."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

from .schema import CSV_COLUMNS, PatientRecord, validate_reference_cohort


DEFAULT_SEED = 9900
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "data" / "synthetic" / "diabetes_patients.csv"
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _generate_profile(rng: random.Random, label: int) -> PatientRecord:
    """Generate one fictional profile with overlapping class distributions."""

    if label == 1:
        age = round(_clamp(rng.gauss(52, 16), 21, 82))
        glucose = _clamp(rng.gauss(7.8, 2.2), 4.5, 13.5)
        hba1c = _clamp(rng.gauss(6.7, 1.3), 4.7, 10.0)
        bmi = _clamp(rng.gauss(30.5, 6.0), 18.5, 44.0)
        systolic_bp = round(_clamp(rng.gauss(133, 19), 94, 178))
        family_history = int(rng.random() < 0.58)
    else:
        age = round(_clamp(rng.gauss(46, 17), 18, 82))
        glucose = _clamp(rng.gauss(5.9, 1.3), 3.8, 9.0)
        hba1c = _clamp(rng.gauss(5.8, 0.9), 4.2, 7.6)
        bmi = _clamp(rng.gauss(28.0, 5.8), 17.5, 42.0)
        systolic_bp = round(_clamp(rng.gauss(127, 18), 90, 172))
        family_history = int(rng.random() < 0.42)

    return PatientRecord(
        patient_id="pending",
        age_years=age,
        glucose_mmol_l=round(glucose, 1),
        hba1c_percent=round(hba1c, 1),
        bmi=round(bmi, 1),
        systolic_bp_mmhg=systolic_bp,
        family_history=family_history,
        diabetes_label=label,
    )


def generate_synthetic_cohort(seed: int = DEFAULT_SEED) -> list[PatientRecord]:
    """Return the same balanced cohort for a given random seed."""

    rng = random.Random(seed)
    cohort = [
        _generate_profile(rng, label)
        for label in ([0] * 20 + [1] * 20)
    ]
    rng.shuffle(cohort)
    cohort = [
        replace(record, patient_id=f"P{index:03d}")
        for index, record in enumerate(cohort, start=1)
    ]
    validate_reference_cohort(cohort)
    return cohort


def write_cohort(records: list[PatientRecord], output_path: str | Path) -> Path:
    """Write a cohort to CSV using the stable schema column order."""

    validate_reference_cohort(records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(record.to_csv_row() for record in records)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic diabetes education cohort."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = write_cohort(generate_synthetic_cohort(args.seed), args.output)
    print(f"Wrote 40 synthetic patient records to {output_path}")


if __name__ == "__main__":
    main()
