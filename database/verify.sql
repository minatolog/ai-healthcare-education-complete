-- Expected: patients=40, unspecified_gender=40, ct_images=3757, reference_masks=3757.
SELECT count(*) AS patients, count(*) FILTER (WHERE gender IS NULL) AS unspecified_gender FROM patient;
SELECT kidney_pool, count(*) AS patients, sum(ct_slice_count) AS ct_slices
FROM patient_library GROUP BY kidney_pool ORDER BY kidney_pool;
SELECT img_type, count(*) FROM patient_image GROUP BY img_type ORDER BY img_type;
SELECT * FROM patient_library ORDER BY patient_code;
SELECT * FROM diabetes_model_input ORDER BY patient_id LIMIT 5;
-- Expected: zero; importing patients does not invent training runs.
SELECT count(*) AS training_runs FROM training_run;
-- Expected: zero rows; each slice must have exactly one CT and one reference mask.
SELECT patient_id, slice_index FROM patient_image
GROUP BY patient_id, slice_index HAVING count(*) <> 2;
