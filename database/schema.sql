BEGIN;

-- Run once in a new healthcare_education database.
CREATE TABLE patient (
    id SERIAL PRIMARY KEY,
    patient_code VARCHAR(32) NOT NULL UNIQUE,
    name VARCHAR(120) NOT NULL CHECK (btrim(name) <> ''),
    age INTEGER NOT NULL CHECK (age BETWEEN 18 AND 85),
    gender VARCHAR(32),
    label VARCHAR(32) NOT NULL CHECK (label IN ('diabetes_negative', 'diabetes_positive')),
    clinical_data JSONB NOT NULL CHECK (jsonb_typeof(clinical_data) = 'object'),
    biomarker JSONB NOT NULL CHECK (jsonb_typeof(biomarker) = 'object'),
    kidney_case_id VARCHAR(32) NOT NULL UNIQUE,
    kidney_pool VARCHAR(10) NOT NULL CHECK (kidney_pool IN ('train', 'validation', 'test')),
    dataset_version VARCHAR(64) NOT NULL,
    provenance JSONB NOT NULL CHECK (jsonb_typeof(provenance) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE patient_image (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patient(id) ON DELETE RESTRICT,
    img_type VARCHAR(32) NOT NULL CHECK (img_type IN ('kidney_ct', 'kidney_reference_mask')),
    slice_index INTEGER NOT NULL CHECK (slice_index >= 0),
    file_path TEXT NOT NULL UNIQUE CHECK (file_path LIKE 'data/kidney/data2d/%'),
    description TEXT NOT NULL,
    image_metadata JSONB NOT NULL CHECK (jsonb_typeof(image_metadata) = 'object'),
    UNIQUE (patient_id, slice_index, img_type)
);

-- Reserved for actual future runs; the patient import creates no training results.
CREATE TABLE training_run (
    id SERIAL PRIMARY KEY,
    work_flow VARCHAR(32) NOT NULL CHECK (work_flow IN ('diabetes', 'kidney')),
    model_type VARCHAR(80) NOT NULL,
    patient_ids JSONB NOT NULL DEFAULT '[]'::JSONB CHECK (jsonb_typeof(patient_ids) = 'array'),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    accuracy DOUBLE PRECISION CHECK (accuracy BETWEEN 0 AND 1),
    metrics JSONB CHECK (jsonb_typeof(metrics) = 'object'),
    feature_importance JSONB,
    model_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX patient_image_patient_id_idx ON patient_image(patient_id);
CREATE INDEX patient_kidney_pool_idx ON patient(kidney_pool);

COMMENT ON TABLE patient IS 'Fictional teaching identities combining independent diabetes and KiTS19 sources, not real matched patients.';
COMMENT ON COLUMN patient.gender IS 'Unspecified: seed values are SQL NULL; names do not define gender.';
COMMENT ON COLUMN patient.age IS 'Synthetic diabetes age; not the age of the original CT subject.';
COMMENT ON COLUMN patient.label IS 'Synthetic diabetes learning target only; never a kidney diagnosis.';
COMMENT ON COLUMN patient.kidney_pool IS 'Fixed pool from project40_manifest.csv, independent of diabetes activity splits.';
COMMENT ON COLUMN patient_image.file_path IS 'Relative to the repository root; actual PNG bytes remain on disk.';
COMMENT ON COLUMN patient_image.image_metadata IS 'Source and reference information, not model input features.';

CREATE VIEW patient_library AS
SELECT p.id, p.patient_code, p.name, p.age, p.gender, p.label,
       p.kidney_case_id, p.kidney_pool,
       count(i.id) FILTER (WHERE i.img_type = 'kidney_ct') AS ct_slice_count,
       count(i.id) FILTER (WHERE i.img_type = 'kidney_reference_mask') AS reference_mask_count
FROM patient p LEFT JOIN patient_image i ON i.patient_id = p.id
GROUP BY p.id;

-- Preserve the six-feature contract used by services/diabetes_model.
-- The target is a separate column and must not enter the feature matrix.
CREATE VIEW diabetes_model_input AS
SELECT patient_code AS patient_id, age AS age_years,
       (biomarker->>'glucose_mmol_l')::DOUBLE PRECISION AS glucose_mmol_l,
       (biomarker->>'hba1c_percent')::DOUBLE PRECISION AS hba1c_percent,
       (clinical_data->>'bmi')::DOUBLE PRECISION AS bmi,
       (clinical_data->>'systolic_bp_mmhg')::INTEGER AS systolic_bp_mmhg,
       (clinical_data->>'family_history')::INTEGER AS family_history,
       CASE label WHEN 'diabetes_positive' THEN 1 ELSE 0 END AS diabetes_label
FROM patient;

COMMIT;
