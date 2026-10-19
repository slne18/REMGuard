"""Leave-one-recording-out REM detector.

The reported metrics come from holding out each recording. A pipeline fitted
on every available recording is saved separately for inference and is not the
source of those metrics.

Chin EMG is not in the feature table. Class imbalance is handled with
``class_weight='balanced'`` inside the training fold. The test epochs are not
resampled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from remguard.features import FEATURE_NAMES, TOTAL_BAND, FeatureTable

RANDOM_STATE = 0


@dataclass
class FoldResult:
    held_out_recording: str
    train_recordings: tuple[str, ...]
    n_train: int
    n_test: int
    onsets_sec: np.ndarray
    y_true: np.ndarray
    forest_pred: np.ndarray
    logistic_pred: np.ndarray
    majority_pred: np.ndarray
    forest: dict[str, float]
    logistic: dict[str, float]
    majority: dict[str, float]


@dataclass
class Evaluation:
    folds: list[FoldResult]
    pooled_forest: dict[str, float]
    pooled_logistic: dict[str, float]
    pooled_majority: dict[str, float]


def make_forest() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_leave_one_recording_out(table: FeatureTable) -> Evaluation:
    """Fit a new pipeline on every fold. No recording is in both sides."""
    recordings = list(dict.fromkeys(table.recording_id.tolist()))
    folds: list[FoldResult] = []
    for held_out in recordings:
        test = table.recording_id == held_out
        train = ~test
        train_recordings = tuple(dict.fromkeys(table.recording_id[train].tolist()))
        if held_out in train_recordings or not np.any(train) or not np.any(test):
            raise RuntimeError(f"invalid split while holding out {held_out}")
        y_train = table.rem[train]
        y_test = table.rem[test]
        forest = make_forest()
        logistic = make_logistic()
        forest.fit(table.values[train], y_train)
        logistic.fit(table.values[train], y_train)
        forest_pred = forest.predict(table.values[test]).astype(int)
        logistic_pred = logistic.predict(table.values[test]).astype(int)
        majority_pred = majority_predict(y_train, int(np.sum(test)))
        folds.append(
            FoldResult(
                held_out_recording=held_out,
                train_recordings=train_recordings,
                n_train=int(np.sum(train)),
                n_test=int(np.sum(test)),
                onsets_sec=table.onset_sec[test].copy(),
                y_true=y_test.copy(),
                forest_pred=forest_pred,
                logistic_pred=logistic_pred,
                majority_pred=majority_pred,
                forest=binary_metrics(y_test, forest_pred),
                logistic=binary_metrics(y_test, logistic_pred),
                majority=binary_metrics(y_test, majority_pred),
            )
        )
    return Evaluation(
        folds=folds,
        pooled_forest=_pooled(folds, "forest_pred"),
        pooled_logistic=_pooled(folds, "logistic_pred"),
        pooled_majority=_pooled(folds, "majority_pred"),
    )


def majority_predict(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the training majority. A tie predicts non-REM."""
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
    label = int(np.argmax(counts))
    return np.full(n_test, label, dtype=int)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics = {
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "n": int(y_true.size),
        "n_rem": int(np.sum(y_true == 1)),
    }
    if np.unique(y_true).size < 2:
        metrics["balanced_accuracy"] = float("nan")
    else:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    return metrics


def fit_detector(table: FeatureTable) -> Pipeline:
    pipeline = make_forest()
    pipeline.fit(table.values, table.rem)
    return pipeline


def save_detector(path: str | Path, pipeline: Pipeline, table: FeatureTable) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = detector_config(table)
    joblib.dump({"pipeline": pipeline, "config": config}, path)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def load_detector(path: str | Path) -> tuple[Pipeline, dict]:
    blob = joblib.load(path)
    pipeline = blob["pipeline"]
    config = blob["config"]
    names = tuple(config["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(f"saved features {names} do not match {FEATURE_NAMES}")
    return pipeline, config


def detector_config(table: FeatureTable) -> dict:
    return {
        "model": "random_forest",
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "feature_names": list(FEATURE_NAMES),
        "excluded_inputs": ["emg_chin"],
        "eeg_total_band_hz": list(TOTAL_BAND),
        "epoch_sec": 30,
        "training_recordings": list(dict.fromkeys(table.recording_id.tolist())),
        "n_training_epochs": int(table.n_epochs),
        "note": (
            "This file is fitted on every training recording so it can score a new "
            "recording. Cross-validated metrics are computed by holding out each "
            "recording and are stored in the phase report, not in this file."
        ),
    }


def _pooled(folds: list[FoldResult], prediction: str) -> dict[str, float]:
    y_true = np.concatenate([fold.y_true for fold in folds])
    y_pred = np.concatenate([getattr(fold, prediction) for fold in folds])
    return binary_metrics(y_true, y_pred)
