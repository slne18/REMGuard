"""Participant-level XGBoost REM detector.

Each recording id is one participant. A five-fold split holds out about 20
percent of participants each time and trains on the rest. Parameter search
and early stopping use a further 20 percent of the training participants, so
the outer test participants are not used to choose the model.

Missing EOG values stay missing. XGBoost splits on them directly. Epochs are
not resampled. Class imbalance is handled with ``scale_pos_weight`` computed
on the fitting participants only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from remguard.features import FeatureTable


@dataclass
class BoostFold:
    fold: int
    test_participants: tuple[str, ...]
    train_participants: tuple[str, ...]
    n_train_epochs: int
    n_test_epochs: int
    best_params: dict
    best_iteration: int
    metrics: dict[str, float]
    majority_error: float


@dataclass
class BoostEvaluation:
    folds: list[BoostFold]
    mean: dict[str, float]
    variance: dict[str, float]
    std: dict[str, float]


def participant_folds(participants: list[str], n_splits: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Return the test participants for each fold. Every id appears once."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(participants) < n_splits:
        raise ValueError(f"{len(participants)} participants is fewer than {n_splits} folds")
    order = np.array(sorted(set(participants)))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return [np.asarray(chunk) for chunk in np.array_split(order, n_splits)]


def inner_participant_split(participants: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out 20 percent of these participants for early stopping and model choice."""
    order = np.array(sorted(set(participants.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n_val = max(1, int(round(0.2 * len(order))))
    if n_val >= len(order):
        n_val = len(order) - 1
    return order[n_val:], order[:n_val]


def evaluate_xgboost(
    table: FeatureTable,
    grid: list[dict],
    n_splits: int = 5,
    seed: int = 0,
    n_estimators: int = 400,
    early_stopping_rounds: int = 30,
    n_jobs: int = 1,
) -> BoostEvaluation:
    participants = list(dict.fromkeys(table.recording_id.tolist()))
    folds = []
    for fold_index, test_ids in enumerate(participant_folds(participants, n_splits, seed)):
        test_set = set(test_ids.tolist())
        test_mask = np.array([recording in test_set for recording in table.recording_id])
        train_mask = ~test_mask
        train_ids = np.array([recording for recording in participants if recording not in test_set])
        fit_ids, val_ids = inner_participant_split(train_ids, seed + fold_index + 1)
        choice = _select_params(
            table,
            fit_ids,
            val_ids,
            grid,
            n_estimators,
            early_stopping_rounds,
            n_jobs,
        )
        model = _fit_full(
            table,
            train_ids,
            choice["params"],
            choice["best_iteration"],
            n_jobs,
        )
        probability = model.predict_proba(table.values[test_mask])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        y_test = table.rem[test_mask]
        majority = int(np.bincount(table.rem[train_mask].astype(int), minlength=2).argmax())
        folds.append(
            BoostFold(
                fold=fold_index,
                test_participants=tuple(sorted(test_set)),
                train_participants=tuple(sorted(train_ids.tolist())),
                n_train_epochs=int(np.sum(train_mask)),
                n_test_epochs=int(np.sum(test_mask)),
                best_params=choice["params"],
                best_iteration=choice["best_iteration"],
                metrics=_metrics(y_test, prediction, probability),
                majority_error=float(np.mean(y_test != majority)),
            )
        )
    return _summarize(folds)


def _select_params(table, fit_ids, val_ids, grid, n_estimators, early_stopping_rounds, n_jobs) -> dict:
    fit_mask = np.isin(table.recording_id, fit_ids)
    val_mask = np.isin(table.recording_id, val_ids)
    best = None
    for params in grid:
        model = _classifier(params, n_estimators, early_stopping_rounds, _positive_weight(table.rem[fit_mask]), n_jobs)
        model.fit(
            table.values[fit_mask],
            table.rem[fit_mask],
            eval_set=[(table.values[val_mask], table.rem[val_mask])],
            verbose=False,
        )
        probability = model.predict_proba(table.values[val_mask])[:, 1]
        loss = float(log_loss(table.rem[val_mask], probability, labels=[0, 1]))
        iteration = int(getattr(model, "best_iteration", n_estimators - 1)) + 1
        if best is None or loss < best["loss"]:
            best = {"params": dict(params), "loss": loss, "best_iteration": iteration}
    return best


def _fit_full(table, train_ids, params, n_estimators, n_jobs) -> XGBClassifier:
    mask = np.isin(table.recording_id, train_ids)
    model = _classifier(params, n_estimators, None, _positive_weight(table.rem[mask]), n_jobs)
    model.fit(table.values[mask], table.rem[mask], verbose=False)
    return model


def _classifier(params: dict, n_estimators: int, early_stopping_rounds: int | None, positive_weight: float, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=max(1, int(n_estimators)),
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        scale_pos_weight=positive_weight,
        random_state=0,
        n_jobs=n_jobs,
        tree_method="hist",
        verbosity=0,
    )


def _positive_weight(labels: np.ndarray) -> float:
    positive = float(np.sum(labels == 1))
    negative = float(np.sum(labels == 0))
    if positive <= 0:
        return 1.0
    return negative / positive


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "error": float(np.mean(y_true != y_pred)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probability)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if np.unique(y_true).size > 1 else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _summarize(folds: list[BoostFold]) -> BoostEvaluation:
    keys = [key for key in folds[0].metrics if key not in {"tn", "fp", "fn", "tp"}]
    mean = {}
    variance = {}
    std = {}
    for key in keys:
        values = np.array([fold.metrics[key] for fold in folds], dtype=float)
        mean[key] = float(np.mean(values))
        variance[key] = float(np.var(values, ddof=1)) if values.size > 1 else 0.0
        std[key] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    majority = np.array([fold.majority_error for fold in folds], dtype=float)
    mean["majority_error"] = float(np.mean(majority))
    variance["majority_error"] = float(np.var(majority, ddof=1))
    std["majority_error"] = float(np.std(majority, ddof=1))
    return BoostEvaluation(folds=folds, mean=mean, variance=variance, std=std)
