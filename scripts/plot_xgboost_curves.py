"""Plot train, validation, and test curves for the REM-detector XGBoost run.

The split matches scripts/train_xgboost_cv.py. Validation log loss still
chooses the parameters and the number of trees. Test nights are scored only
so the curves can be drawn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from remguard.boosting import (
    _classifier,
    _positive_weight,
    _select_params,
    inner_participant_split,
    participant_folds,
)
from train_xgboost_cv import GRID, NIGHTS, load_nights

OUT_DIR = ROOT / "reports"
CURVE_JSON = OUT_DIR / "rem_detector_learning_curves.json"
SELECTION_PNG = OUT_DIR / "rem_detector_learning_curves.png"
REFIT_PNG = OUT_DIR / "rem_detector_refit_curves.png"


def main() -> None:
    table, _skipped = load_nights(NIGHTS)
    participants = list(dict.fromkeys(table.recording_id.tolist()))
    folds = []
    for fold_index, test_ids in enumerate(participant_folds(participants, n_splits=5, seed=0)):
        print(f"fold {fold_index + 1}", flush=True)
        test_set = set(test_ids.tolist())
        train_ids = np.array([recording for recording in participants if recording not in test_set])
        fit_ids, val_ids = inner_participant_split(train_ids, seed=fold_index + 1)
        choice = _select_params(table, fit_ids, val_ids, GRID, n_estimators=400, early_stopping_rounds=30, n_jobs=4)
        selection = _selection_curves(table, fit_ids, val_ids, test_set, choice["params"])
        if selection["trees_chosen"] != choice["best_iteration"]:
            raise RuntimeError(
                f"fold {fold_index + 1} curve stop {selection['trees_chosen']} "
                f"disagrees with selection {choice['best_iteration']}"
            )
        refit = _refit_curves(table, train_ids, test_set, choice["params"], choice["best_iteration"])
        folds.append(
            {
                "fold": fold_index + 1,
                "params": choice["params"],
                "trees_chosen": choice["best_iteration"],
                "selection": selection,
                "refit": refit,
            }
        )
        stop = selection["trees_chosen"] - 1
        print(
            f"  trees {choice['best_iteration']}  "
            f"train {selection['train_logloss'][stop]:.3f} / {selection['train_error'][stop]:.3f}  "
            f"val {selection['val_logloss'][stop]:.3f} / {selection['val_error'][stop]:.3f}  "
            f"test {selection['test_logloss'][stop]:.3f} / {selection['test_error'][stop]:.3f}",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_JSON.write_text(json.dumps(folds), encoding="utf-8")
    _plot_selection(folds, SELECTION_PNG)
    _plot_refit(folds, REFIT_PNG)
    print(f"Wrote {SELECTION_PNG}", flush=True)
    print(f"Wrote {REFIT_PNG}", flush=True)


def _selection_curves(table, fit_ids, val_ids, test_set, params) -> dict:
    fit_mask = np.isin(table.recording_id, fit_ids)
    val_mask = np.isin(table.recording_id, val_ids)
    test_mask = np.array([recording in test_set for recording in table.recording_id])
    model = _classifier(params, 400, 30, _positive_weight(table.rem[fit_mask]), n_jobs=4)
    model.set_params(eval_metric=["error", "logloss"])
    model.fit(
        table.values[fit_mask],
        table.rem[fit_mask],
        eval_set=[
            (table.values[test_mask], table.rem[test_mask]),
            (table.values[fit_mask], table.rem[fit_mask]),
            (table.values[val_mask], table.rem[val_mask]),
        ],
        verbose=False,
    )
    history = model.evals_result()
    chosen = int(model.best_iteration) + 1
    return {
        "trees_chosen": chosen,
        "train_logloss": history["validation_1"]["logloss"],
        "val_logloss": history["validation_2"]["logloss"],
        "test_logloss": history["validation_0"]["logloss"],
        "train_error": history["validation_1"]["error"],
        "val_error": history["validation_2"]["error"],
        "test_error": history["validation_0"]["error"],
    }


def _refit_curves(table, train_ids, test_set, params, n_estimators) -> dict:
    train_mask = np.isin(table.recording_id, train_ids)
    test_mask = np.array([recording in test_set for recording in table.recording_id])
    model = _classifier(params, n_estimators, None, _positive_weight(table.rem[train_mask]), n_jobs=4)
    model.set_params(eval_metric=["error", "logloss"])
    model.fit(
        table.values[train_mask],
        table.rem[train_mask],
        eval_set=[
            (table.values[train_mask], table.rem[train_mask]),
            (table.values[test_mask], table.rem[test_mask]),
        ],
        verbose=False,
    )
    history = model.evals_result()
    return {
        "train_logloss": history["validation_0"]["logloss"],
        "test_logloss": history["validation_1"]["logloss"],
        "train_error": history["validation_0"]["error"],
        "test_error": history["validation_1"]["error"],
    }


def _plot_selection(folds: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), sharex="col")
    for column, fold in enumerate(folds):
        series = fold["selection"]
        trees = np.arange(1, len(series["train_logloss"]) + 1)
        chosen = fold["trees_chosen"]
        top = axes[0, column]
        bottom = axes[1, column]
        _draw(top, trees, series["train_logloss"], series["val_logloss"], series["test_logloss"], chosen)
        _draw(bottom, trees, series["train_error"], series["val_error"], series["test_error"], chosen)
        top.set_title(f"Fold {fold['fold']}")
        bottom.set_xlabel("Trees")
        if column == 0:
            top.set_ylabel("Log loss")
            bottom.set_ylabel("Classification error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("Model selection. Dashed line: tree count chosen by validation log loss.", fontsize=12, y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _plot_refit(folds: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(14.5, 6.2), sharex="col")
    for column, fold in enumerate(folds):
        series = fold["refit"]
        trees = np.arange(1, len(series["train_logloss"]) + 1)
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(trees, series["train_logloss"], color="#1f4e79", label="Train (fit + validation nights)")
        top.plot(trees, series["test_logloss"], color="#9a3412", label="Test nights")
        bottom.plot(trees, series["train_error"], color="#1f4e79")
        bottom.plot(trees, series["test_error"], color="#9a3412")
        top.set_title(f"Fold {fold['fold']}, {fold['trees_chosen']} trees")
        bottom.set_xlabel("Trees")
        if column == 0:
            top.set_ylabel("Log loss")
            bottom.set_ylabel("Classification error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("Reported model, refit on all training nights and scored on the test nights.", fontsize=12, y=1.02)
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
