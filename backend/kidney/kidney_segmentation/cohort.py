"""Pure functions for an audit-bound patient-level PNG cohort lock.

These functions do not read patient files, write a lock, use old splits, select
labels, train a model, or establish clinical image/mask alignment. The caller
must preserve an existing lock rather than overwrite it with another cohort.
"""
import hashlib
import json
import re

from .png_dataset import PNGConfig

AUDIT_SCHEMA = "png_dataset_audit_v1"
COHORT_SCHEMA = "png_cohort_v1"
POLICY_VERSION = "patient_sha256_rank_v1"
ROLES = ("train", "validation", "test", "prediction")


def canonical_fingerprint(payload, fingerprint_field):
    """Hash every field except the specified self-referential fingerprint."""
    content = {key: value for key, value in payload.items() if key != fingerprint_field}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _audit_index(audit):
    if not isinstance(audit, dict) or audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != "passed":
        raise ValueError("A passed version-1 PNG dataset machine audit is required.")
    if audit.get("issues", []) or audit.get("global_issues", []):
        raise ValueError("Resolve audit machine errors before creating a cohort.")
    if not _is_hash(audit.get("manifest_sha256")) or not _is_hash(audit.get("audit_fingerprint_sha256")):
        raise ValueError("The audit requires valid manifest and audit fingerprints.")
    if canonical_fingerprint(audit, "audit_fingerprint_sha256") != audit["audit_fingerprint_sha256"]:
        raise ValueError("The audit fingerprint is stale or the audit content has changed.")
    patients = audit.get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError("The audit must contain patient records.")
    index = {}
    for patient in patients:
        if not isinstance(patient, dict):
            raise ValueError("Invalid audit patient record.")
        patient_id = patient.get("patient_id")
        source_id = patient.get("source_patient_id")
        count = patient.get("selected_slice_count")
        if not isinstance(patient_id, str) or not re.fullmatch(r"case_[0-9]{5}", patient_id) or patient_id in index:
            raise ValueError("Audit patient IDs must be valid and unique.")
        if not isinstance(source_id, str) or not source_id.strip() or type(patient.get("eligible")) is not bool:
            raise ValueError("Audit source identity and eligibility are required.")
        if type(count) is not int or count < 0:
            raise ValueError("Audit selected_slice_count must be a non-negative integer.")
        fingerprint = patient.get("content_fingerprint_sha256")
        if patient["eligible"] and (count == 0 or not _is_hash(fingerprint)):
            raise ValueError("Eligible patients require selected slices and a content fingerprint.")
        if fingerprint is not None and not _is_hash(fingerprint):
            raise ValueError("Invalid patient content fingerprint.")
        index[patient_id] = patient
    components = audit.get("duplicate_patient_components", [])
    if not isinstance(components, list):
        raise ValueError("Duplicate patient components must be a list.")
    for component in components:
        if (not isinstance(component, list) or len(component) < 2 or
                any(not isinstance(patient_id, str) for patient_id in component) or len(set(component)) != len(component)):
            raise ValueError("A duplicate component requires at least two unique patient IDs.")
        if any(patient_id not in index for patient_id in component):
            raise ValueError("A duplicate component references an unknown patient.")
    return index


def _groups(audit, index):
    # Connected components also cover transitive duplicates. Shared source or
    # complete patient fingerprints receive the same protection as audit edges.
    parent = {patient_id: patient_id for patient_id in index}

    def find(patient_id):
        while parent[patient_id] != patient_id:
            parent[patient_id] = parent[parent[patient_id]]
            patient_id = parent[patient_id]
        return patient_id

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)

    for component in audit.get("duplicate_patient_components", []):
        for patient_id in component[1:]:
            union(component[0], patient_id)
    for field in ("source_patient_id", "content_fingerprint_sha256"):
        seen = {}
        for patient_id, patient in index.items():
            value = patient.get(field)
            if value is None:
                continue
            if value in seen:
                union(seen[value], patient_id)
            else:
                seen[value] = patient_id
    groups = {}
    for patient_id in index:
        groups.setdefault(find(patient_id), []).append(patient_id)
    return [sorted(group) for _, group in sorted(groups.items())]


def _rank(seed, role, patient_id):
    encoded = json.dumps([POLICY_VERSION, seed, role, patient_id], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), patient_id


def _settings(seed, exposed_patient_ids, prediction_count, index):
    if type(seed) is not int or seed < 0 or type(prediction_count) is not int or prediction_count < 1:
        raise ValueError("Use a non-negative integer seed and a positive integer prediction_count.")
    if isinstance(exposed_patient_ids, (str, bytes)):
        raise ValueError("Supply exposed patient IDs as a sequence.")
    try:
        exposed = tuple(exposed_patient_ids)
    except TypeError:
        raise ValueError("Supply exposed patient IDs as a non-empty sequence.") from None
    if not exposed or len(exposed) > 20 or any(not isinstance(patient_id, str) for patient_id in exposed) or len(set(exposed)) != len(exposed):
        raise ValueError("Supply 1 to 20 distinct development-exposed patient IDs.")
    if any(patient_id not in index or not index[patient_id]["eligible"] for patient_id in exposed):
        raise ValueError("Every development-exposed patient must exist and be audit-eligible.")
    return sorted(exposed)


def _record(patient):
    return {field: patient.get(field) for field in
            ("patient_id", "source_patient_id", "selected_slice_count", "content_fingerprint_sha256")}


def _build(audit, index, seed, exposed, prediction_count):
    exposed_set = set(exposed)
    representatives, duplicate_exclusions = [], {}
    groups = _groups(audit, index)
    for group in groups:
        eligible = [patient_id for patient_id in group if index[patient_id]["eligible"]]
        if not eligible:
            continue
        forced = exposed_set.intersection(eligible)
        if len(forced) > 1:
            raise ValueError("Multiple exposed IDs share a duplicate component; distinct forced training patients cannot be selected.")
        representative = next(iter(forced)) if forced else min(eligible, key=lambda patient_id: _rank(seed, "duplicate_representative", patient_id))
        representatives.append(representative)
        for patient_id in eligible:
            if patient_id != representative:
                duplicate_exclusions[patient_id] = representative
    counts = {"train": 20, "validation": 10, "test": 10, "prediction": prediction_count}
    if len(representatives) < sum(counts.values()):
        raise ValueError("Insufficient unique eligible patient groups: need %d, found %d." % (sum(counts.values()), len(representatives)))
    remaining = set(representatives) - exposed_set
    pools = {}
    train = sorted(exposed, key=lambda patient_id: _rank(seed, "train", patient_id))
    train += sorted(remaining, key=lambda patient_id: _rank(seed, "train", patient_id))[:20 - len(train)]
    pools["train"] = [_record(index[patient_id]) for patient_id in train]
    remaining.difference_update(train)
    for role in ROLES[1:]:
        chosen = sorted(remaining, key=lambda patient_id: _rank(seed, role, patient_id))[:counts[role]]
        pools[role] = [_record(index[patient_id]) for patient_id in chosen]
        remaining.difference_update(chosen)
    assigned = {record["patient_id"] for records in pools.values() for record in records}
    excluded = []
    for patient_id in sorted(set(index) - assigned):
        record = _record(index[patient_id])
        if not index[patient_id]["eligible"]:
            record["reason"] = "audit_ineligible"
        elif patient_id in duplicate_exclusions:
            record.update(reason="duplicate_component_nonrepresentative", representative_patient_id=duplicate_exclusions[patient_id])
        else:
            record["reason"] = "not_selected_by_fixed_counts"
        excluded.append(record)
    config = PNGConfig().to_dict()
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": COHORT_SCHEMA, "sampling_policy_version": POLICY_VERSION,
        "ranking_definition": "SHA256 of compact JSON [policy_version, seed, role, patient_id]; hash order then patient ID",
        "seed": seed, "required_patient_counts": counts, "exposed_patient_ids": exposed,
        "manifest_sha256": audit["manifest_sha256"], "audit_fingerprint_sha256": audit["audit_fingerprint_sha256"],
        "preprocessing": {"input_mode": "single_2d_slice", "config": config, "config_sha256": config_sha256},
        "pools": pools, "excluded_patients": excluded,
        "excluded_patient_ids": [patient["patient_id"] for patient in excluded],
        "duplicate_component_policy": "one eligible representative; development-exposed representative forced into train",
        "selection_scope": "reference-selected exported 2D PNG slices; no full-volume evaluation",
        "machine_audit_status": "passed", "manual_clinical_alignment_verified": False, "model_trained": False,
        "limitations": ["Audit success covers machine checks, not clinical alignment or original source alias verification.",
                        "Source patient IDs are audit-provided case namespaces; real-person aliases are not independently established.",
                        "No healthy-patient label or patient-level tumour/no-tumour diagnosis is inferred.",
                        "Selection uses patient identity, audit eligibility and duplicate protection, not model performance or class labels.",
                        "Preserve an existing lock; changed data, audit or selection settings require an explicit new cohort decision."]
    }
    payload["cohort_fingerprint_sha256"] = canonical_fingerprint(payload, "cohort_fingerprint_sha256")
    return payload


def make_cohort(audit, seed=42, exposed_patient_ids=("case_00000",), prediction_count=5):
    """Return a reproducible 20/10/10 + prediction cohort without writing files."""
    index = _audit_index(audit)
    exposed = _settings(seed, exposed_patient_ids, prediction_count, index)
    return _build(audit, index, seed, exposed, prediction_count)


def validate_cohort(payload, audit):
    """Validate a saved cohort against this exact audit; raise on any mismatch."""
    index = _audit_index(audit)
    if not isinstance(payload, dict) or payload.get("schema_version") != COHORT_SCHEMA:
        raise ValueError("Unsupported PNG cohort lock schema.")
    if payload.get("sampling_policy_version") != POLICY_VERSION:
        raise ValueError("Unknown cohort sampling policy.")
    if payload.get("manifest_sha256") != audit["manifest_sha256"] or payload.get("audit_fingerprint_sha256") != audit["audit_fingerprint_sha256"]:
        raise ValueError("The cohort is bound to a different or changed manifest/audit.")
    if not _is_hash(payload.get("cohort_fingerprint_sha256")) or canonical_fingerprint(payload, "cohort_fingerprint_sha256") != payload["cohort_fingerprint_sha256"]:
        raise ValueError("The cohort fingerprint is stale or the lock has changed.")
    counts = payload.get("required_patient_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("Explicit cohort patient counts are required.")
    prediction_count = counts.get("prediction")
    exposed = _settings(payload.get("seed"), payload.get("exposed_patient_ids"), prediction_count, index)
    if counts != {"train": 20, "validation": 10, "test": 10, "prediction": prediction_count}:
        raise ValueError("Canonical cohort counts must be 20/10/10 plus the prediction pool.")
    pools = payload.get("pools", {})
    if not isinstance(pools, dict) or set(pools) != set(ROLES):
        raise ValueError("All four patient pools are required.")
    assigned, sources, fingerprints = [], [], []
    for role in ROLES:
        records = pools[role]
        if not isinstance(records, list) or len(records) != counts[role]:
            raise ValueError("Incorrect patient count in the %s pool." % role)
        for record in records:
            patient_id = record.get("patient_id") if isinstance(record, dict) else None
            if not isinstance(patient_id, str) or patient_id not in index or not index[patient_id]["eligible"]:
                raise ValueError("A selected patient is unknown or audit-ineligible.")
            if record != _record(index[patient_id]):
                raise ValueError("Selected patient source/content metadata differs from the audit.")
            assigned.append(patient_id)
            sources.append(record["source_patient_id"])
            fingerprints.append(record["content_fingerprint_sha256"])
    if any(len(set(values)) != len(values) for values in (assigned, sources, fingerprints)):
        raise ValueError("Patient, source or content overlap across assigned pools.")
    assigned_set = set(assigned)
    if any(len(assigned_set.intersection(group)) > 1 for group in _groups(audit, index)):
        raise ValueError("A duplicate patient component has more than one selected representative.")
    train_ids = {record["patient_id"] for record in pools["train"]}
    if not set(exposed).issubset(train_ids):
        raise ValueError("Development-exposed patients must be in training, not holdout or prediction pools.")
    excluded = payload.get("excluded_patients")
    if not isinstance(excluded, list) or any(not isinstance(record, dict) for record in excluded):
        raise ValueError("Explicit exclusion records are required.")
    excluded_ids = [record.get("patient_id") for record in excluded]
    if (any(not isinstance(patient_id, str) for patient_id in excluded_ids) or
            len(set(excluded_ids)) != len(excluded_ids) or assigned_set.intersection(excluded_ids) or assigned_set.union(excluded_ids) != set(index)):
        raise ValueError("Excluded and selected IDs must cover the audit exactly without overlap.")
    if payload.get("excluded_patient_ids") != excluded_ids:
        raise ValueError("Excluded patient ID index does not match its records.")
    expected = _build(audit, index, payload["seed"], exposed, prediction_count)
    if payload != expected:
        raise ValueError("Cohort content differs from its deterministic selection/configuration policy.")
