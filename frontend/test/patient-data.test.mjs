import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { getBloodTests } from '../src/patients.ts'
import { pathForRoute, routeFromPath } from '../src/routes.ts'

const patients = JSON.parse(readFileSync(new URL('../../backend/data/patients.json', import.meta.url)))

test('all 40 diabetes tables retain stored measurements and units', () => {
  assert.equal(patients.length, 40)
  for (const patient of patients) {
    const values = getBloodTests(patient)
    assert.equal(values.Age, `${patient.age} years`)
    assert.equal(values['Blood Glucose'], `${patient.biomarker.glucose_mmol_l} mmol/L`)
    assert.equal(values.HbA1c, `${patient.biomarker.hba1c_percent}%`)
    assert.equal(values.BMI, `${patient.clinical_data.bmi} kg/m²`)
    assert.equal(values['Systolic Blood Pressure'], `${patient.clinical_data.systolic_bp_mmhg} mmHg`)
    assert.equal(values['Family History'], patient.clinical_data.family_history ? 'Yes' : 'No')
    assert.equal(values['Reference Label'], patient.label === 'diabetes_positive' ? 'Diabetes' : 'No diabetes')
    assert.deepEqual(Object.keys(values), ['Age', 'Blood Glucose', 'HbA1c', 'BMI', 'Systolic Blood Pressure', 'Family History', 'Reference Label'])
  }
})

test('CT-only records cannot produce a fabricated diabetes table', () => {
  assert.deepEqual(getBloodTests({ id: 1, patient_code: 'P001', name: 'Drew Scott', case_id: 'case_00000', pool: 'train', slice_count: 322 }), {})
})

test('prediction routes cover all stored IDs for either modality', () => {
  for (const mode of ['diabetes', 'kidney']) {
    for (let testId = 1; testId <= 40; testId++) {
      const route = { screen: 7, mode, testId }
      assert.deepEqual(routeFromPath(pathForRoute(route)), route)
    }
    assert.deepEqual(routeFromPath(`/${mode}/patients`), { screen: 2, mode, testId: null })
  }
})

test('unknown and malformed patient routes are rejected', () => {
  for (const id of ['0', '41', '-1', '1.5', '001', 'NaN']) {
    assert.equal(routeFromPath(`/kidney/prediction/${id}`), null)
  }
  for (const id of [0, 41, 1.5, NaN, null]) {
    assert.throws(() => pathForRoute({ screen: 7, mode: 'kidney', testId: id }))
  }
})
