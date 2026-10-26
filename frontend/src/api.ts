/** Typed client for the local Python model API (proxied through Vite at /api). */

import type { Mode } from './routes'

// ── Diabetes (Random Forest) ──────────────────────────────────
export interface TrainingPoint {
  event: 'training_progress'
  phase: string
  step: number
  total_steps: number
  progress_percent: number
  trees_built: number
  train_accuracy: number
  validation_accuracy: number
  loss: null
}

export interface EvaluationPrediction {
  patient_id: string
  actual_label: number
  predicted_label: number
  estimated_diabetes_probability: number
  correct: boolean
}

export interface EvaluationResult {
  evaluation_scope: string
  sample_count: number
  metrics: {
    accuracy: number
    balanced_accuracy: number
    precision: number
    recall: number
    f1: number
    confusion_matrix: number[][]
  }
  predictions: EvaluationPrediction[]
  probability_note: string
}

export interface FeatureImportance {
  rank: number
  feature_name: string
  display_name: string
  importance: number
}

export interface TrainingSession {
  mode: 'diabetes'
  status: 'trained'
  educational_only: boolean
  selection: {
    training_patient_ids: string[]
    training_size: number
    evaluation_size: number
    held_out_size: number
    training_class_counts: Record<string, number>
    evaluation_class_counts: Record<string, number>
  }
  split: { training: string[]; evaluation: string[]; held_out: string[] }
  model: {
    algorithm: string
    feature_names: string[]
    best_tree_count: number
    best_evaluation_accuracy: number
    trained_in_session: boolean
  }
  training_curve: { x_axis: string; series: string[]; points: TrainingPoint[] }
  feature_importance: {
    importance_type: string
    features: FeatureImportance[]
    note: string
  }
  evaluation: EvaluationResult
  held_out_evaluation: EvaluationResult
  probability_note: string
}

export interface FeatureEffect {
  rank: number
  feature_name: string
  display_name: string
  patient_value: number
  reference_value: number
  probability_change: number
  direction: 'increased_score' | 'decreased_score' | 'no_change'
}

export interface PredictionResult {
  mode: 'diabetes'
  patient_id: string
  predicted_label: number
  estimated_diabetes_probability: number
  prediction_threshold?: number
  actual_label: number
  source: 'session_model' | 'committed_demo_model'
  feature_effects: FeatureEffect[]
  best_tree_count: number
  note?: string
  probability_note?: string
}

// ── Kidney (U-Net segmentation) ───────────────────────────────
export interface KidneyTrainingPoint {
  epoch: number
  train_loss: number
  validation_kidney_dice: number | null
  validation_tumour_dice: number | null
}

export interface KidneyEvaluationRow {
  case_id: string
  patient_id: string
  actual_label: number
  predicted_label: number
  kidney_dice: number | null
  tumour_dice: number | null
  correct: boolean
}

export interface KidneySession {
  mode: 'kidney'
  status: 'trained'
  training_mode: 'recorded_checkpoint'
  educational_only: boolean
  selection: {
    training_patient_ids: string[]
    training_size: number
    evaluation_size: number
    held_out_size: number
    training_slice_count?: number
    evaluation_slice_count?: number
  }
  split: { training: string[]; validation: string[]; held_out: string[] }
  model: {
    algorithm: string
    class_names: string[]
    selected_epoch: number
    best_validation_tumour_dice: number | null
    device: string
    input_size: number
    epochs: number
    trained_in_session: boolean
  }
  training_curve: { x_axis: string; series: string[]; points: KidneyTrainingPoint[] }
  evaluation: {
    evaluation_scope: string
    sample_count: number
    metrics: {
      kidney_dice: number | null
      tumour_dice: number | null
      pooled_kidney_dice: number | null
      pooled_tumour_dice: number | null
      accuracy: number
      correct: number
      total: number
    }
    predictions: KidneyEvaluationRow[]
    probability_note: string
  }
  probability_note: string
}

export interface KidneySliceScore {
  z: number
  tumour_pixel_fraction: number
  max_tumour_probability: number
}

export interface KidneyPrediction {
  mode: 'kidney'
  patient_id: string
  case_id: string
  slice_count: number
  tumour_slice_count: number
  tumour_slice_ratio: number
  estimated_tumour_probability: number
  predicted_label: number
  reference_tumour_slice_count?: number
  actual_label?: number
  tumour_slices: KidneySliceScore[]
  selected_epoch: number
  source: 'checkpoint'
  probability_note: string
}

export type ModelSession = TrainingSession | KidneySession
export type AnyPrediction = PredictionResult | KidneyPrediction

export function isKidneySession(session: ModelSession | null): session is KidneySession {
  return session !== null && session.mode === 'kidney'
}

export function isKidneyPrediction(prediction: AnyPrediction | null): prediction is KidneyPrediction {
  return prediction !== null && prediction.mode === 'kidney'
}

export function evaluationPatientCodes(session: ModelSession): string[] {
  return isKidneySession(session)
    ? session.evaluation.predictions.map(row => row.patient_id)
    : session.split.evaluation
}

export function overlayUrl(patientCode: string, z: number): string {
  return `/api/kidney/patients/${patientCode}/overlay/${z}.png`
}

// ── HTTP ──────────────────────────────────────────────────────
export interface ModelStatusEntry {
  mode: Mode
  available: boolean
  algorithm: string
  supported_operations: string[]
  reasons?: string[]
  reason?: string
  documentation?: string
}

export class ModelApiError extends Error {
  unavailable: boolean
  status: number

  constructor(message: string, status: number, unavailable = false) {
    super(message)
    this.name = 'ModelApiError'
    this.status = status
    this.unavailable = unavailable
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new ModelApiError('The model service could not be reached. Start the backend and try again.', 0)
  }
  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }
  if (!response.ok) {
    const body = (payload ?? {}) as { message?: string; error?: string }
    const message = body.message || body.error || `The model service returned ${response.status}.`
    throw new ModelApiError(message, response.status, response.status === 503)
  }
  return payload as T
}

export function fetchModelStatus(): Promise<{ models: Record<Mode, ModelStatusEntry> }> {
  return request('/api/model/status')
}

export function trainModel(mode: Mode, patientCodes: string[]): Promise<ModelSession> {
  return request(`/api/${mode}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_ids: patientCodes }),
  })
}

export function predictPatient(mode: Mode, patientCode: string): Promise<AnyPrediction> {
  return request(`/api/${mode}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_code: patientCode }),
  })
}
