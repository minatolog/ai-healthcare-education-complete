"""Synthetic audit records test patient split integrity; no patient files are read."""
import copy
import hashlib
import pytest

from kidney_segmentation.cohort import canonical_fingerprint, make_cohort, validate_cohort


def audited_pool(count=70, components=None):
    audit = {"schema_version": "png_dataset_audit_v1", "status": "passed",
             "manifest_sha256": hashlib.sha256(b"fixture manifest").hexdigest(), "issues": [],
             "duplicate_patient_components": components or [], "patients": []}
    for index in range(count):
        patient = "case_%05d" % index
        audit["patients"].append({"patient_id": patient, "source_patient_id": "kits19:" + patient,
                                  "eligible": True, "selected_slice_count": 10 + index,
                                  "content_fingerprint_sha256": hashlib.sha256(patient.encode()).hexdigest()})
    audit["audit_fingerprint_sha256"] = canonical_fingerprint(audit, "audit_fingerprint_sha256")
    return audit


def reseal(payload, field="cohort_fingerprint_sha256"):
    payload[field] = canonical_fingerprint(payload, field)
    return payload


def ids(payload, role):
    return [record["patient_id"] for record in payload["pools"][role]]


def test_determinism_fixed_counts_exposure_and_explicit_exclusions():
    audit = audited_pool()
    original = copy.deepcopy(audit)
    cohort = make_cohort(audit)
    assert audit == original
    assert cohort == make_cohort(audit)
    assert {role: len(records) for role, records in cohort["pools"].items()} == {"train": 20, "validation": 10, "test": 10, "prediction": 5}
    assert "case_00000" in ids(cohort, "train")
    assigned = [patient for role in cohort["pools"] for patient in ids(cohort, role)]
    assert len(set(assigned)) == 45
    assert not set(assigned).intersection(cohort["excluded_patient_ids"])
    assert len(cohort["excluded_patient_ids"]) == 25
    assert cohort["model_trained"] is False and cohort["manual_clinical_alignment_verified"] is False
    assert cohort["preprocessing"]["config"]["input_channels"] == 1
    validate_cohort(cohort, audit)


def test_seed_changes_membership_without_changing_exposure_rule():
    audit = audited_pool()
    first, second = make_cohort(audit, seed=42), make_cohort(audit, seed=43)
    assert first["pools"] != second["pools"]
    assert "case_00000" in ids(first, "train") and "case_00000" in ids(second, "train")
    validate_cohort(second, audit)


def test_transitive_duplicate_component_keeps_exposed_representative_and_records_exclusions():
    audit = audited_pool(components=[["case_00000", "case_00001"], ["case_00001", "case_00002"]])
    cohort = make_cohort(audit)
    assert "case_00000" in ids(cohort, "train")
    assigned = {patient for role in cohort["pools"] for patient in ids(cohort, role)}
    assert not assigned.intersection({"case_00001", "case_00002"})
    excluded = {record["patient_id"]: record for record in cohort["excluded_patients"]}
    for patient in ("case_00001", "case_00002"):
        assert excluded[patient]["reason"] == "duplicate_component_nonrepresentative"
        assert excluded[patient]["representative_patient_id"] == "case_00000"
    validate_cohort(cohort, audit)
    with pytest.raises(ValueError, match="Multiple exposed"):
        make_cohort(audit, exposed_patient_ids=("case_00000", "case_00001"))


def test_shared_source_or_complete_content_is_also_deduplicated():
    audit = audited_pool()
    audit["patients"][1]["source_patient_id"] = audit["patients"][0]["source_patient_id"]
    audit["patients"][2]["content_fingerprint_sha256"] = audit["patients"][0]["content_fingerprint_sha256"]
    reseal(audit, "audit_fingerprint_sha256")
    cohort = make_cohort(audit)
    assigned = {patient for role in cohort["pools"] for patient in ids(cohort, role)}
    assert assigned.intersection({"case_00000", "case_00001", "case_00002"}) == {"case_00000"}
    validate_cohort(cohort, audit)


def test_unknown_or_ineligible_exposure_and_too_small_unique_pool_fail():
    with pytest.raises(ValueError, match="Insufficient unique"):
        make_cohort(audited_pool(count=44))
    with pytest.raises(ValueError, match="Insufficient unique"):
        make_cohort(audited_pool(count=45, components=[["case_00000", "case_00001"]]))
    audit = audited_pool()
    with pytest.raises(ValueError, match="audit-eligible"):
        make_cohort(audit, exposed_patient_ids=("case_99999",))
    audit["patients"][0]["eligible"] = False
    reseal(audit, "audit_fingerprint_sha256")
    with pytest.raises(ValueError, match="audit-eligible"):
        make_cohort(audit)
    with pytest.raises(ValueError, match="1 to 20"):
        make_cohort(audited_pool(), exposed_patient_ids=tuple("case_%05d" % i for i in range(21)))


def test_machine_errors_and_stale_audit_hash_fail():
    invalid = audited_pool()
    invalid["status"] = "failed"
    reseal(invalid, "audit_fingerprint_sha256")
    with pytest.raises(ValueError, match="passed version-1"):
        make_cohort(invalid)
    audit = audited_pool()
    audit["patients"][2]["selected_slice_count"] += 1
    with pytest.raises(ValueError, match="stale"):
        make_cohort(audit)
    audit = audited_pool()
    audit["issues"] = [{"code": "MASK_PAIR_MISSING"}]
    reseal(audit, "audit_fingerprint_sha256")
    with pytest.raises(ValueError, match="machine errors"):
        make_cohort(audit)


def test_existing_lock_rejects_new_audit_even_when_patient_ids_are_unchanged():
    audit = audited_pool()
    cohort = make_cohort(audit)
    changed = copy.deepcopy(audit)
    changed["manifest_sha256"] = hashlib.sha256(b"changed manifest").hexdigest()
    reseal(changed, "audit_fingerprint_sha256")
    with pytest.raises(ValueError, match="changed manifest/audit"):
        validate_cohort(cohort, changed)


def test_patient_overlap_stale_lock_and_source_content_forgery_are_rejected():
    audit = audited_pool()
    cohort = make_cohort(audit)
    changed = copy.deepcopy(cohort)
    changed["pools"]["test"][0] = copy.deepcopy(changed["pools"]["train"][0])
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_cohort(changed, audit)
    reseal(changed)
    with pytest.raises(ValueError, match="overlap"):
        validate_cohort(changed, audit)
    changed = copy.deepcopy(cohort)
    changed["pools"]["test"][0]["source_patient_id"] = cohort["pools"]["train"][0]["source_patient_id"]
    reseal(changed)
    with pytest.raises(ValueError, match="metadata differs"):
        validate_cohort(changed, audit)


def test_exposed_patient_cannot_be_moved_to_test_even_after_rehashing():
    audit = audited_pool()
    cohort = make_cohort(audit)
    position = ids(cohort, "train").index("case_00000")
    cohort["pools"]["train"][position], cohort["pools"]["test"][0] = cohort["pools"]["test"][0], cohort["pools"]["train"][position]
    reseal(cohort)
    with pytest.raises(ValueError, match="Development-exposed"):
        validate_cohort(cohort, audit)


def test_duplicate_group_is_rejected_if_an_excluded_alias_is_inserted_into_test():
    audit = audited_pool(components=[["case_00000", "case_00001"]])
    cohort = make_cohort(audit)
    alias = next(patient for patient in audit["patients"] if patient["patient_id"] == "case_00001")
    cohort["pools"]["test"][0] = {key: alias[key] for key in ("patient_id", "source_patient_id", "selected_slice_count", "content_fingerprint_sha256")}
    reseal(cohort)
    with pytest.raises(ValueError, match="duplicate patient component"):
        validate_cohort(cohort, audit)
