"""Train the REM detector with participant-level cross-validation and a random forest.

The night split matches the XGBoost run. Validation log loss chooses the tree
count and the other forest settings. Test nights are scored after that choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from remguard.boosting import inner_participant_split, participant_folds
from train_xgboost_cv import NIGHTS, STATUS, load_nights

REPORT = ROOT / "reports" / "rem_detector_random_forest.md"
SELECTION_PNG = ROOT / "reports" / "rem_detector_random_forest_learning_curves.png"
REFIT_PNG = ROOT / "reports" / "rem_detector_random_forest_refit_curves.png"

MAX_TREES = 400
TREE_STEP = 10
PATIENCE = 30
GRID = [
    {"max_depth": depth, "min_samples_leaf": leaf}
    for depth in (8, 16)
    for leaf in (1, 5)
]


def main() -> None:
    table, skipped = load_nights(NIGHTS)
    participants = list(dict.fromkeys(table.recording_id.tolist()))
    if len(participants) < 5:
        raise RuntimeError(f"only {len(participants)} participants have features; extraction is not finished")
    print(f"{table.n_epochs} epochs from {len(participants)} participants", flush=True)
    folds = []
    for fold_index, test_ids in enumerate(participant_folds(participants, n_splits=5, seed=0)):
        print(f"fold {fold_index + 1}", flush=True)
        folds.append(_fold(table, participants, test_ids, fold_index))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report(table, participants, skipped, folds), encoding="utf-8")
    _plot_selection(folds, SELECTION_PNG)
    _plot_refit(folds, REFIT_PNG)
    errors = np.array([fold["test_metrics"]["error"] for fold in folds], dtype=float)
    print(f"Wrote {REPORT}", flush=True)
    print(
        f"error {errors.mean():.3f} ± {errors.std(ddof=1):.3f} (variance {errors.var(ddof=1):.4f})",
        flush=True,
    )


def _fold(table, participants, test_ids, fold_index) -> dict:
    test_set = set(test_ids.tolist())
    train_ids = np.array([recording for recording in participants if recording not in test_set])
    fit_ids, val_ids = inner_participant_split(train_ids, seed=fold_index + 1)
    fit_mask = np.isin(table.recording_id, fit_ids)
    val_mask = np.isin(table.recording_id, val_ids)
    test_mask = np.array([recording in test_set for recording in table.recording_id])
    train_mask = fit_mask | val_mask
    prepared = _transform(table.values, fit_mask)
    choice = None
    winner = None
    for params in GRID:
        forest = _forest(params, MAX_TREES)
        forest.fit(prepared[fit_mask], table.rem[fit_mask])
        val_trees, val_history = _history(forest, prepared[val_mask], table.rem[val_mask])
        trees, loss = _choose(val_trees, val_history["logloss"])
        print(
            f"  depth {params['max_depth']} leaf {params['min_samples_leaf']}: "
            f"{trees} trees, validation log loss {loss:.3f}",
            flush=True,
        )
        if choice is None or loss < choice["loss"]:
            choice = {"params": dict(params), "trees": trees, "loss": loss}
            winner = forest
    selection = _named_histories(
        winner,
        {
            "train": (prepared[fit_mask], table.rem[fit_mask]),
            "val": (prepared[val_mask], table.rem[val_mask]),
            "test": (prepared[test_mask], table.rem[test_mask]),
        },
    )
    if _choose(selection["trees"], selection["val_logloss"])[0] != choice["trees"]:
        raise RuntimeError(f"fold {fold_index + 1} validation curve disagrees with the selected tree count")
    refit_values = _transform(table.values, train_mask)
    refit = _forest(choice["params"], choice["trees"])
    refit.fit(refit_values[train_mask], table.rem[train_mask])
    refit_curves = _named_histories(
        refit,
        {
            "train": (refit_values[train_mask], table.rem[train_mask]),
            "test": (refit_values[test_mask], table.rem[test_mask]),
        },
    )
    probability = refit.predict_proba(refit_values[test_mask])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    y_test = table.rem[test_mask]
    majority = int(np.bincount(table.rem[train_mask].astype(int), minlength=2).argmax())
    stop = selection["trees"].index(choice["trees"])
    print(
        f"  chosen depth {choice['params']['max_depth']}, leaf {choice['params']['min_samples_leaf']}, "
        f"trees {choice['trees']}",
        flush=True,
    )
    return {
        "fold": fold_index + 1,
        "test_participants": tuple(sorted(test_set)),
        "n_train_epochs": int(np.sum(train_mask)),
        "n_test_epochs": int(np.sum(test_mask)),
        "params": choice["params"],
        "trees": choice["trees"],
        "test_metrics": _metrics(y_test, prediction, probability),
        "majority_error": float(np.mean(y_test != majority)),
        "selection": selection,
        "selection_at_stop": {
            "train_logloss": selection["train_logloss"][stop],
            "val_logloss": selection["val_logloss"][stop],
            "test_logloss": selection["test_logloss"][stop],
            "train_error": selection["train_error"][stop],
            "val_error": selection["val_error"][stop],
            "test_error": selection["test_error"][stop],
        },
        "refit": refit_curves,
    }


def _forest(params: dict, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced",
        random_state=0,
        n_jobs=4,
    )


def _transform(values: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit(values[fit_mask]).transform(values)
    return scaler.fit(imputed[fit_mask]).transform(imputed)


def _history(forest, values: np.ndarray, labels: np.ndarray) -> tuple[list[int], dict[str, list[float]]]:
    running = np.zeros((values.shape[0], 2), dtype=float)
    trees = []
    logloss = []
    error = []
    total = len(forest.estimators_)
    for index, tree in enumerate(forest.estimators_, start=1):
        running += _tree_proba(tree, values)
        if index % TREE_STEP == 0 or index == total:
            probability = np.clip(running[:, 1] / index, 1e-6, 1.0 - 1e-6)
            prediction = (probability >= 0.5).astype(int)
            trees.append(index)
            logloss.append(float(log_loss(labels, probability, labels=[0, 1])))
            error.append(float(np.mean(prediction != labels)))
    return trees, {"logloss": logloss, "error": error}


def _named_histories(forest, groups: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    stored = {name: _history(forest, values, labels) for name, (values, labels) in groups.items()}
    trees = stored[next(iter(stored))][0]
    curves = {"trees": trees}
    for name, (_trees, history) in stored.items():
        curves[f"{name}_logloss"] = history["logloss"]
        curves[f"{name}_error"] = history["error"]
    return curves


def _tree_proba(tree, values: np.ndarray) -> np.ndarray:
    probability = tree.predict_proba(values)
    if probability.shape[1] == 2:
        return probability
    full = np.zeros((values.shape[0], 2), dtype=float)
    full[:, int(tree.classes_[0])] = probability[:, 0]
    return full


def _choose(trees: list[int], losses: list[float]) -> tuple[int, float]:
    best_tree = trees[0]
    best_loss = losses[0]
    for tree, loss in zip(trees, losses):
        if loss < best_loss:
            best_tree = tree
            best_loss = loss
        elif tree - best_tree >= PATIENCE:
            break
    return best_tree, best_loss


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
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


def _summarize(folds: list[dict]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    keys = [key for key in folds[0]["test_metrics"] if key not in {"tn", "fp", "fn", "tp"}]
    mean, variance, std = {}, {}, {}
    for key in keys:
        values = np.array([fold["test_metrics"][key] for fold in folds], dtype=float)
        mean[key] = float(np.mean(values))
        variance[key] = float(np.var(values, ddof=1))
        std[key] = float(np.std(values, ddof=1))
    majority = np.array([fold["majority_error"] for fold in folds], dtype=float)
    mean["majority_error"] = float(np.mean(majority))
    variance["majority_error"] = float(np.var(majority, ddof=1))
    std["majority_error"] = float(np.std(majority, ddof=1))
    return mean, variance, std


def _report(table, participants, skipped, folds) -> str:
    mean, variance, std = _summarize(folds)
    rem_rate = float(np.mean(table.rem == 1))
    lines = [
        "# REM detector, random forest",
        "",
        "The label is expert REM versus expert wake and non-REM. Chin EMG is not an input. Diagnosis is not an input. Each recording id is treated as one participant. The five test groups are the same nights as in the XGBoost run.",
        "",
        "Each fold holds out about 20 percent of the nights. Inside the training nights, a further 20 percent is the validation set. The forest is grown to 400 trees. Validation log loss is read every 10 trees, and growth stops after 30 trees with no improvement. The winning depth, leaf size, and tree count are refit on all training nights. The test nights are not used for either choice.",
        "",
        "Grid: max depth 8 or 16, minimum samples per leaf 1 or 5. Each tree considers sqrt(10) features. Class imbalance uses balanced class weights. A median imputer and a standard scaler are fit on the nights being trained, so missing EOG stays inside the training fold.",
        "",
        f"Participants with labeled epochs: {len(participants)}. Epochs: {table.n_epochs}. Expert REM share: {rem_rate:.3f}.",
        f"Recordings with no usable epochs: {len(skipped)}.",
        "",
        "| Metric | Mean | Standard deviation | Variance |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("error", "Classification error"),
        ("log_loss", "Log loss"),
        ("brier", "Brier score"),
        ("recall", "REM recall"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1", "F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("majority_error", "Majority-class error"),
    ):
        lines.append(f"| {label} | {mean[key]:.3f} | {std[key]:.3f} | {variance[key]:.4f} |")
    lines.extend(
        [
            "",
            f"Learning curves while the tree count is chosen: `{SELECTION_PNG.name}`.",
            f"Learning curves of the refit reported model: `{REFIT_PNG.name}`.",
            "",
            "## Folds",
            "",
        ]
    )
    for fold in folds:
        metrics = fold["test_metrics"]
        at = fold["selection_at_stop"]
        refit = fold["refit"]
        lines.extend(
            [
                f"### Fold {fold['fold']}",
                "",
                f"- Test participants ({len(fold['test_participants'])}): {', '.join(fold['test_participants'])}.",
                f"- Train epochs {fold['n_train_epochs']}, test epochs {fold['n_test_epochs']}.",
                f"- Parameters: depth {fold['params']['max_depth']}, min samples per leaf {fold['params']['min_samples_leaf']}, "
                f"trees {fold['trees']}.",
                f"- At that tree count, before the refit: train log loss {at['train_logloss']:.3f}, "
                f"validation {at['val_logloss']:.3f}, test {at['test_logloss']:.3f}. "
                f"Train error {at['train_error']:.3f}, validation {at['val_error']:.3f}, test {at['test_error']:.3f}.",
                f"- Refit train log loss {refit['train_logloss'][-1]:.3f}, test log loss {refit['test_logloss'][-1]:.3f}. "
                f"Refit train error {refit['train_error'][-1]:.3f}, test error {refit['test_error'][-1]:.3f}.",
                f"- Test error {metrics['error']:.3f}, log loss {metrics['log_loss']:.3f}, Brier {metrics['brier']:.3f}.",
                f"- REM recall {metrics['recall']:.3f}, specificity {metrics['specificity']:.3f}, "
                f"precision {metrics['precision']:.3f}, F1 {metrics['f1']:.3f}, "
                f"balanced accuracy {metrics['balanced_accuracy']:.3f}.",
                f"- Confusion matrix: true non-REM {metrics['tn']}, false REM {metrics['fp']}, "
                f"missed REM {metrics['fn']}, true REM {metrics['tp']}.",
                f"- Majority-class error on this fold: {fold['majority_error']:.3f}.",
                "",
            ]
        )
    if skipped:
        lines.append("## Recordings without a feature table")
        lines.append("")
        for row in skipped:
            lines.append(f"- {row['recording_id']}: {row['reason']}.")
        lines.append("")
    failed = []
    if STATUS.exists():
        import json

        for row in json.loads(STATUS.read_text(encoding="utf-8")):
            if row["reason"] == "failed":
                failed.append(row["recording_id"])
    if failed:
        lines.append("Extraction failed for: " + ", ".join(failed) + ".")
        lines.append("")
    lines.append(
        "These figures describe this CAP collection under this split. They do not estimate performance on other recording systems, and they are not a Parkinson's score."
    )
    lines.append("")
    return "\n".join(lines)


def _plot_selection(folds: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), sharex="col")
    for column, fold in enumerate(folds):
        series = fold["selection"]
        trees = np.asarray(series["trees"])
        top, bottom = axes[0, column], axes[1, column]
        _draw(top, trees, series["train_logloss"], series["val_logloss"], series["test_logloss"], fold["trees"])
        _draw(bottom, trees, series["train_error"], series["val_error"], series["test_error"], fold["trees"])
        top.set_title(f"Fold {fold['fold']}")
        bottom.set_xlabel("Trees")
        if column == 0:
            top.set_ylabel("Log loss")
            bottom.set_ylabel("Classification error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("Random forest model selection. Dashed line: tree count chosen by validation log loss.", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_refit(folds: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), sharex="col")
    for column, fold in enumerate(folds):
        series = fold["refit"]
        trees = np.asarray(series["trees"])
        top, bottom = axes[0, column], axes[1, column]
        top.plot(trees, series["train_logloss"], color="#1f4e79", label="Train (fit + validation nights)")
        top.plot(trees, series["test_logloss"], color="#9a3412", label="Test nights")
        bottom.plot(trees, series["train_error"], color="#1f4e79")
        bottom.plot(trees, series["test_error"], color="#9a3412")
        top.set_title(f"Fold {fold['fold']}, {fold['trees']} trees")
        bottom.set_xlabel("Trees")
        if column == 0:
            top.set_ylabel("Log loss")
            bottom.set_ylabel("Classification error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("Random forest reported model, refit on all training nights and scored on the test nights.", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _draw(axis, trees, train, validation, test, chosen) -> None:
    axis.plot(trees, train, color="#1f4e79", label="Train (fit nights)")
    axis.plot(trees, validation, color="#b45309", label="Validation nights")
    axis.plot(trees, test, color="#9a3412", label="Test nights")
    axis.axvline(chosen, color="#57534e", linewidth=0.8, linestyle="--")
    axis.set_xlim(1, max(int(trees[-1]), chosen) + 12)


if __name__ == "__main__":
    main()
