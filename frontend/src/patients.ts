import { useEffect, useState } from 'react'
import type { Mode } from './routes'

export interface Patient {
  id: number
  patient_code: string
  name: string
  pool: 'development' | 'train' | 'validation' | 'test'
  age?: number
  label?: 'diabetes_positive' | 'diabetes_negative'
  clinical_data?: { bmi: number; systolic_bp_mmhg: number; family_history: number }
  biomarker?: { glucose_mmol_l: number; hba1c_percent: number }
  case_id?: string
  slice_count?: number
}

export function usePatients(mode: Mode | null) {
  const [result, setResult] = useState<{ mode: Mode; patients: Patient[] } | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    setResult(null)
    setError('')
    if (!mode) return
    const controller = new AbortController()
    fetch(`/api/${mode}/patients`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error('Patient data could not be loaded.')
        const payload = await response.json()
        if (!Array.isArray(payload.patients) || payload.patients.length !== 40) {
          throw new Error('The patient dataset is incomplete.')
        }
        if (!controller.signal.aborted) setResult({ mode, patients: payload.patients })
      })
      .catch(err => {
        if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Patient data could not be loaded.')
      })
    return () => controller.abort()
  }, [mode, attempt])
  return { patients: result?.mode === mode ? result.patients : [], error,
    retry: () => setAttempt(value => value + 1) }
}

export function getBloodTests(patient: Patient): Record<string, string> {
  if (!patient.biomarker || !patient.clinical_data) return {}
  return {
    Age: `${patient.age} years`,
    'Blood Glucose': `${patient.biomarker.glucose_mmol_l} mmol/L`,
    HbA1c: `${patient.biomarker.hba1c_percent}%`,
    BMI: `${patient.clinical_data.bmi} kg/m²`,
    'Systolic Blood Pressure': `${patient.clinical_data.systolic_bp_mmhg} mmHg`,
    'Family History': patient.clinical_data.family_history === 1 ? 'Yes' : 'No',
    'Reference Label': patient.label === 'diabetes_positive' ? 'Diabetes' : 'No diabetes',
  }
}
