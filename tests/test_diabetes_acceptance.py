"""Tests for the final diabetes ML delivery acceptance command."""

from __future__ import annotations

import json
import tempfile
import unittest

from services.diabetes_model.acceptance import (
    DELIVERY_VERSION,
    run_delivery_acceptance,
)


class DiabetesDeliveryAcceptanceTests(unittest.TestCase):
    def test_committed_delivery_passes_every_check(self) -> None:
        report = run_delivery_acceptance()

        self.assertTrue(report.passed)
        self.assertEqual(report.delivery_version, DELIVERY_VERSION)
        self.assertEqual(len(report.checks), 8)
        self.assertTrue(all(check.passed for check in report.checks))

    def test_acceptance_report_is_json_ready(self) -> None:
        payload = run_delivery_acceptance().as_dict()

        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["educational_only"])
        self.assertEqual(payload["check_count"], 8)
        json.dumps(payload)

    def test_missing_delivery_returns_failed_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            report = run_delivery_acceptance(temp_directory)

        self.assertFalse(report.passed)
        self.assertTrue(any(not check.passed for check in report.checks))


if __name__ == "__main__":
    unittest.main()
