"""Model layer that connects the patient API to the project's real models.

Diabetes uses the pure Python/scikit-learn package in ``services/diabetes_model``.
Training runs on the server for the student's 20 selected patients, and the
fitted estimator stays in memory so later predictions use the session model.

Kidney segmentation uses the included teaching U-Net checkpoint. Training and
evaluation pages read its recorded history; patient inference runs the model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COHORT_PATH = ROOT / 'data/synthetic/diabetes_patients.csv'
DEMO_ARTIFACT_PATH = ROOT / 'artifacts/diabetes_demo/diabetes_random_forest.joblib'

TRAINING_SIZE = 20
EVALUATION_SIZE = 10
HELD_OUT_SIZE = 10
DEVELOPMENT_LIMIT = 30  # P001-P030 form the development pool; P031-P040 stay held out.


class RequestError(ValueError):
    """Raised when a request body does not match the model API contract."""

    def __init__(self, message: str, code: str = 'invalid_request') -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {'error': self.code, 'message': self.message}


class ModelUnavailable(RuntimeError):
    """Raised when a model cannot run in this environment."""

    def __init__(self, mode: str, message: str, code: str = 'model_unavailable') -> None:
        super().__init__(message)
        self.mode = mode
        self.message = message
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {'error': self.code, 'mode': self.mode, 'message': self.message}


def _relative(path: Path) -> str:
    """Return a repository-relative path for API responses."""

    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def patient_number(patient_code: str) -> int:
    """Return the numeric part of a ``P###`` teaching identity."""

    if not isinstance(patient_code, str) or len(patient_code) != 4:
        raise RequestError(f'Unknown patient code: {patient_code!r}.')
    if patient_code[0] != 'P' or not patient_code[1:].isdigit():
        raise RequestError(f'Unknown patient code: {patient_code!r}.')
    return int(patient_code[1:])


class _Session:
    """Fitted model and the datasets needed to explain later predictions."""

    def __init__(self, package, result, training, evaluation, patient_codes) -> None:
        self.package = package
        self.result = result
        self.training = training
        self.evaluation = evaluation
        self.patient_codes = patient_codes


class DiabetesModelService:
    """Train and predict with the diabetes Random Forest activity model."""

    mode = 'diabetes'

    def __init__(self, cohort_path: Path = COHORT_PATH,
                 artifact_path: Path = DEMO_ARTIFACT_PATH) -> None:
        self.cohort_path = Path(cohort_path)
        self.artifact_path = Path(artifact_path)
        self._lock = threading.Lock()
        self._records = None
        self._bundle = None
        self._bundle_error: str | None = None
        self._session: _Session | None = None

    # -- lazy loading ------------------------------------------------------
    def _package(self):
        try:
            from services import diabetes_model
        except Exception as error:  # pragma: no cover - environment dependent
            raise ModelUnavailable(
                self.mode,
                'The diabetes model package could not be imported: '
                f'{error}. Install services/diabetes_model/requirements.txt.',
            ) from error
        return diabetes_model

    def _cohort(self):
        if self._records is None:
            package = self._package()
            self._records = package.load_reference_cohort(self.cohort_path)
        return self._records

    def _demo_bundle(self):
        if self._bundle is None and self._bundle_error is None:
            try:
                self._bundle = self._package().load_model_bundle(self.artifact_path)
            except Exception as error:
                self._bundle_error = str(error)
        if self._bundle is None:
            raise ModelUnavailable(
                self.mode,
                f'The committed demonstration model could not be loaded: {self._bundle_error}',
            )
        return self._bundle

    # -- API ---------------------------------------------------------------
    def status(self) -> dict[str, object]:
        status: dict[str, object] = {
            'mode': self.mode,
            'available': True,
            'algorithm': 'RandomForestClassifier',
            'training_size': TRAINING_SIZE,
            'evaluation_size': EVALUATION_SIZE,
            'held_out_size': HELD_OUT_SIZE,
            'session_trained': self._session is not None,
            'supported_operations': ['train', 'predict'],
            'educational_only': True,
        }
        try:
            self._package()
            self._cohort()
        except Exception as error:
            status['available'] = False
            status['reason'] = str(error)
            return status
        metadata_path = self.artifact_path.with_suffix('.metadata.json')
        if self.artifact_path.is_file():
            status['committed_model'] = {'artifact': _relative(self.artifact_path)}
            try:
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                status['committed_model'].update({
                    'best_tree_count': metadata.get('best_tree_count'),
                    'sklearn_version': metadata.get('sklearn_version'),
                })
            except (OSError, json.JSONDecodeError):
                pass
        else:
            status['committed_model'] = {
                'error': f'Missing artifact: {_relative(self.artifact_path)}'
            }
        if self._session is not None:
            status['session'] = {
                'patient_ids': list(self._session.patient_codes),
                'best_tree_count': self._session.result.best_tree_count,
            }
        return status

    def train(self, patient_codes: object) -> dict[str, object]:
        codes = self._validate_training_codes(patient_codes)
        package = self._package()
        records = self._cohort()
        by_code = {record.patient_id: record for record in records}

        selected = set(codes)
        training_records = [by_code[code] for code in codes]
        evaluation_records = [
            record for record in records
            if patient_number(record.patient_id) <= DEVELOPMENT_LIMIT
            and record.patient_id not in selected
        ]
        held_out_records = [
            record for record in records
            if patient_number(record.patient_id) > DEVELOPMENT_LIMIT
        ]
        if len(evaluation_records) != EVALUATION_SIZE:
            raise RequestError(
                f'Expected {EVALUATION_SIZE} evaluation patients, '
                f'found {len(evaluation_records)}.'
            )
        if len(held_out_records) != HELD_OUT_SIZE:
            raise RequestError(
                f'Expected {HELD_OUT_SIZE} held-out patients, '
                f'found {len(held_out_records)}.'
            )

        training = package.prepare_dataset(training_records)
        evaluation = package.prepare_dataset(evaluation_records)
        held_out = package.prepare_dataset(held_out_records)

        result = package.train_random_forest(training, evaluation)
        importance = package.extract_feature_importance(result)
        held_out_evaluation = package.evaluate_test_set(result, held_out)

        session = _Session(package, result, training, evaluation, tuple(codes))
        with self._lock:
            self._session = session

        return {
            'mode': self.mode,
            'status': 'trained',
            'educational_only': True,
            'selection': {
                'training_patient_ids': list(codes),
                'training_size': len(training.patient_ids),
                'evaluation_size': len(evaluation.patient_ids),
                'held_out_size': len(held_out.patient_ids),
                'training_class_counts': _class_counts(training.labels),
                'evaluation_class_counts': _class_counts(evaluation.labels),
            },
            'split': {
                'training': list(training.patient_ids),
                'evaluation': list(evaluation.patient_ids),
                'held_out': list(held_out.patient_ids),
            },
            'model': {
                'algorithm': 'RandomForestClassifier',
                'feature_names': list(package.MODEL_FEATURES),
                'best_tree_count': result.best_tree_count,
                'best_evaluation_accuracy': round(result.best_validation_accuracy, 4),
                'trained_in_session': True,
            },
            'training_curve': {
                'x_axis': 'trees_built',
                'series': ['training_accuracy', 'evaluation_accuracy'],
                'points': result.history_as_dicts(),
            },
            'feature_importance': {
                'importance_type': 'global_split_importance',
                'features': [entry.as_dict() for entry in importance],
                'note': package.FEATURE_IMPORTANCE_NOTE,
            },
            'evaluation': _evaluate(result, evaluation),
            'held_out_evaluation': held_out_evaluation.as_dict(),
            'probability_note': package.PROBABILITY_NOTE,
        }

    def predict(self, patient_code: object) -> dict[str, object]:
        if not isinstance(patient_code, str):
            raise RequestError('patient_code must be a string such as "P031".')
        code = patient_code.strip()
        by_code = {record.patient_id: record for record in self._cohort()}
        record = by_code.get(code)
        if record is None:
            raise RequestError(f'Unknown patient code: {code!r}.')

        with self._lock:
            session = self._session

        if session is None:
            return self._predict_with_committed_model(record)

        package = session.package
        patient_dataset = package.prepare_dataset([record])
        explanation = package.explain_dataset_predictions(
            session.result, session.training, patient_dataset,
        )[0]
        payload = explanation.as_dict()
        payload.update({
            'mode': self.mode,
            'source': 'session_model',
            'educational_only': True,
            'actual_label': record.diabetes_label,
            'best_tree_count': session.result.best_tree_count,
        })
        return payload

    def _predict_with_committed_model(self, record) -> dict[str, object]:
        bundle = self._demo_bundle()
        request = {
            'patient_id': record.patient_id,
            'age_years': record.age_years,
            'glucose_mmol_l': record.glucose_mmol_l,
            'hba1c_percent': record.hba1c_percent,
            'bmi': record.bmi,
            'systolic_bp_mmhg': record.systolic_bp_mmhg,
            'family_history': record.family_history,
        }
        prediction = self._package().predict_patient(bundle, request).as_dict()
        prediction.update({
            'mode': self.mode,
            'source': 'committed_demo_model',
            'actual_label': record.diabetes_label,
            'feature_effects': [],
            'best_tree_count': bundle.best_tree_count,
            'note': (
                'No training session is loaded, so the committed demonstration '
                'model answered and no local explanation is available.'
            ),
        })
        return prediction

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _validate_training_codes(patient_codes: object) -> list[str]:
        if not isinstance(patient_codes, (list, tuple)):
            raise RequestError('patient_ids must be a JSON array of patient codes.')
        codes: list[str] = []
        for value in patient_codes:
            if not isinstance(value, str) or not value.strip():
                raise RequestError('Every patient id must be a non-empty string.')
            codes.append(value.strip())
        if len(codes) != TRAINING_SIZE:
            raise RequestError(f'Select exactly {TRAINING_SIZE} patients for training.')
        if len(set(codes)) != TRAINING_SIZE:
            raise RequestError('Training patients must be unique.')
        reserved = sorted(
            code for code in codes
            if patient_number(code) > DEVELOPMENT_LIMIT
        )
        if reserved:
            raise RequestError(
                'The following patients are held out for prediction and cannot '
                'be trained on: ' + ', '.join(reserved) + '.'
            )
        return codes


class KidneyModelService:
    """Real U-Net segmentation for the kidney activity.

    The checkpoint in ``artifacts/kidney`` is produced by
    ``backend/train_kidney_model.py`` and is intentionally not committed, matching
    the team rule that model weights stay outside Git. When it is missing the
    service explains how to create it instead of inventing predictions.
    """

    mode = 'kidney'
    CHECKPOINT_PATH = ROOT / 'artifacts/kidney/kidney_unet.pt'
    METRICS_PATH = ROOT / 'artifacts/kidney/kidney_unet.metrics.json'
    TRAIN_COMMAND = 'python backend/train_kidney_model.py'

    def __init__(self, case_by_code: dict[str, str] | None = None,
                 checkpoint_path: Path | None = None) -> None:
        self.case_by_code = dict(case_by_code or {})
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.CHECKPOINT_PATH
        self._lock = threading.Lock()
        self._predictor = None
        self._config: dict[str, object] | None = None
        self._load_error: str | None = None

    # -- lazy loading ------------------------------------------------------
    def _module(self):
        try:
            import kidney_web_model as module
        except Exception as error:  # pragma: no cover - environment dependent
            raise ModelUnavailable(
                self.mode,
                'The kidney model could not be imported: '
                f'{error}. Install torch, pillow and numpy in the backend environment.',
            ) from error
        return module

    def _load(self):
        with self._lock:
            if self._predictor is not None:
                return self._predictor, self._config or {}
            if self._load_error is not None:
                raise ModelUnavailable(self.mode, self._load_error)
            if not self.checkpoint_path.is_file():
                raise ModelUnavailable(self.mode, self._missing_message())
            try:
                module = self._module()
                device = module._select_device()
                model, config = module.load_checkpoint(self.checkpoint_path, device)
                self._predictor = module.Predictor(model, config, device)
                self._config = config
            except ModelUnavailable:
                raise
            except Exception as error:
                self._load_error = f'The kidney checkpoint could not be loaded: {error}'
                raise ModelUnavailable(self.mode, self._load_error) from error
            return self._predictor, self._config or {}

    def _missing_message(self) -> str:
        return (
            'No kidney checkpoint is available yet. Train the teaching U-Net once with '
            f'`{self.TRAIN_COMMAND}`; the API then serves real segmentation results.'
        )

    # -- API ---------------------------------------------------------------
    def status(self) -> dict[str, object]:
        torch_available = importlib.util.find_spec('torch') is not None
        checkpoint = self.checkpoint_path.is_file()
        reasons: list[str] = []
        if not torch_available:
            reasons.append('PyTorch is not installed in the backend Python environment.')
        if not checkpoint:
            reasons.append(self._missing_message())

        status: dict[str, object] = {
            'mode': self.mode,
            'available': not reasons,
            'algorithm': 'U-Net segmentation (3 classes: background, kidney, tumour)',
            'training_size': 20,
            'evaluation_size': 10,
            'supported_operations': [] if reasons else ['train', 'predict'],
            'torch_available': torch_available,
            'checkpoint': _relative(self.checkpoint_path) if checkpoint else None,
            'reasons': reasons,
            'documentation': 'artifacts/kidney/README.md',
            'educational_only': True,
        }
        if checkpoint and self.METRICS_PATH.is_file():
            try:
                metrics = json.loads(self.METRICS_PATH.read_text(encoding='utf-8'))
                status['checkpoint_metrics'] = {
                    'selected_epoch': metrics.get('selected_epoch'),
                    'test_kidney_dice': (metrics.get('test') or {}).get('patient_mean_kidney_dice'),
                    'test_tumour_dice': (metrics.get('test') or {}).get('patient_mean_tumour_dice'),
                    'elapsed_seconds': metrics.get('elapsed_seconds'),
                }
            except (OSError, json.JSONDecodeError):
                pass
        return status

    def train(self, patient_codes: object = None) -> dict[str, object]:
        """Return the recorded training session of the shipped checkpoint."""

        _, config = self._load()
        split = config.get('split') or {}
        if patient_codes is not None:
            expected = {self._code_for_case(case) for case in split.get('train', [])}
            if (not isinstance(patient_codes, list)
                    or not all(isinstance(code, str) for code in patient_codes)
                    or len(patient_codes) != len(expected)
                    or set(patient_codes) != expected):
                raise RequestError('Select exactly the 20 locked kidney training cases. This checkpoint is not retrained by the website.')
        history = config.get('history') or []
        test = config.get('test') or {}
        cases = test.get('cases') or []

        predictions = []
        correct = 0
        for row in cases:
            actual_label = 1 if int(row.get('tumour_pixels_in_reference') or 0) > 0 else 0
            predicted_label = 1 if int(row.get('tumour_pixels_predicted') or 0) > 0 else 0
            matched = actual_label == predicted_label
            correct += int(matched)
            predictions.append({
                'case_id': row.get('case_id'),
                'patient_id': self._code_for_case(row.get('case_id')),
                'actual_label': actual_label,
                'predicted_label': predicted_label,
                'kidney_dice': row.get('kidney_dice'),
                'tumour_dice': row.get('tumour_dice'),
                'correct': matched,
            })

        total = len(predictions) or 1
        training_cases = list(split.get('train') or [])
        return {
            'mode': self.mode,
            'status': 'trained',
            'training_mode': 'recorded_checkpoint',
            'educational_only': True,
            'selection': {
                'training_patient_ids': [
                    self._code_for_case(case_id) or case_id for case_id in training_cases
                ],
                'training_size': len(training_cases),
                'evaluation_size': len(split.get('test') or []),
                'held_out_size': len(split.get('test') or []),
                'training_slice_count': split.get('train_slice_count'),
                'evaluation_slice_count': split.get('test_slice_count'),
            },
            'split': {
                'training': training_cases,
                'validation': list(split.get('validation') or []),
                'held_out': list(split.get('test') or []),
            },
            'model': {
                'algorithm': 'U-Net segmentation (3 classes)',
                'class_names': list(config.get('class_names') or []),
                'selected_epoch': config.get('selected_epoch'),
                'best_validation_tumour_dice': config.get('selected_validation_tumour_dice'),
                'device': config.get('device'),
                'input_size': config.get('input_size'),
                'epochs': config.get('epochs'),
                'trained_in_session': False,
            },
            'training_curve': {
                'x_axis': 'epoch',
                'series': ['train_loss', 'validation_kidney_dice', 'validation_tumour_dice'],
                'points': history,
            },
            'evaluation': {
                'evaluation_scope': 'held_out_test',
                'sample_count': len(predictions),
                'metrics': {
                    'kidney_dice': test.get('patient_mean_kidney_dice'),
                    'tumour_dice': test.get('patient_mean_tumour_dice'),
                    'pooled_kidney_dice': test.get('pooled_kidney_dice'),
                    'pooled_tumour_dice': test.get('pooled_tumour_dice'),
                    'accuracy': round(correct / total, 4),
                    'correct': correct,
                    'total': len(predictions),
                },
                'predictions': predictions,
                'probability_note': (
                    'Segmentation overlap and tumour-detection agreement on 10 unseen '
                    'teaching cases; not a clinical result.'
                ),
            },
            'probability_note': (
                'Model-estimated educational score; not a calibrated clinical risk.'
            ),
        }

    def predict(self, patient_code: object) -> dict[str, object]:
        if not isinstance(patient_code, str):
            raise RequestError('patient_code must be a string such as "P031".')
        code = patient_code.strip()
        case_id = self.case_by_code.get(code)
        if case_id is None:
            raise RequestError(f'Unknown patient code: {code!r}.')

        predictor, config = self._load()
        prediction = predictor.predict_case(case_id)
        prediction.update({
            'mode': self.mode,
            'patient_id': code,
            'source': 'checkpoint',
            'selected_epoch': config.get('selected_epoch'),
            'probability_note': (
                'Largest softmax tumour probability across this case and the share of '
                'slices with predicted tumour. Educational only.'
            ),
        })
        return prediction

    def overlay(self, patient_code: str, z: int) -> bytes:
        case_id = self.case_by_code.get(patient_code)
        if case_id is None:
            raise RequestError(f'Unknown patient code: {patient_code!r}.')
        if z not in self._module().case_slices().get(case_id, []):
            raise RequestError('Choose a slice from this patient’s CT manifest.')
        predictor, _ = self._load()
        return predictor.overlay_png(case_id, int(z))

    def _code_for_case(self, case_id: object) -> str | None:
        for code, case in self.case_by_code.items():
            if case == case_id:
                return code
        return None


def make_model_services(store=None):
    """Return the model registry used by the HTTP API."""

    case_by_code = {}
    if store is not None:
        case_by_code = {
            patient['patient_code']: patient['kidney_case_id']
            for patient in store.patients
        }
    return {
        'diabetes': DiabetesModelService(),
        'kidney': KidneyModelService(case_by_code=case_by_code),
    }


def _class_counts(labels: Iterable[int]) -> dict[str, int]:
    counts = {0: 0, 1: 0}
    for label in labels:
        counts[int(label)] += 1
    return {'label_0': counts[0], 'label_1': counts[1]}


def _evaluate(result, dataset) -> dict[str, object]:
    """Score a fitted model on one unseen group and return the API contract."""

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    model = result.model
    class_labels = list(model.classes_)
    positive_index = class_labels.index(1)
    predicted = model.predict(dataset.features)
    probabilities = model.predict_proba(dataset.features)[:, positive_index]
    matrix = confusion_matrix(dataset.labels, predicted, labels=[0, 1])
    predictions = [
        {
            'patient_id': patient_id,
            'actual_label': int(actual),
            'predicted_label': int(predicted_label),
            'estimated_diabetes_probability': round(float(probability), 4),
            'correct': int(actual) == int(predicted_label),
        }
        for patient_id, actual, predicted_label, probability in zip(
            dataset.patient_ids, dataset.labels, predicted, probabilities,
        )
    ]
    return {
        'evaluation_scope': 'unseen_evaluation_group',
        'sample_count': len(predictions),
        'metrics': {
            'accuracy': round(float(accuracy_score(dataset.labels, predicted)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(dataset.labels, predicted)), 4),
            'precision': round(float(precision_score(dataset.labels, predicted, zero_division=0)), 4),
            'recall': round(float(recall_score(dataset.labels, predicted, zero_division=0)), 4),
            'f1': round(float(f1_score(dataset.labels, predicted, zero_division=0)), 4),
            'confusion_matrix': [[int(value) for value in row] for row in matrix],
        },
        'predictions': predictions,
        'probability_note': 'Model-estimated educational score; not a calibrated clinical risk.',
    }
