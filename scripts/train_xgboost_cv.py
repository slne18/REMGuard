"""Train the REM detector with participant-level 80/20 cross-validation and XGBoost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from remguard.boosting import evaluate_xgboost
from remguard.features import FEATURE_NAMES, FeatureTable

NIGHTS = ROOT / "data" / "features" / "nights"
REPORT = ROOT / "reports" / "rem_detector_xgboost.md"
STATUS = ROOT / "data" / "features" / "status.json"

GRID = [
    {
        "learning_rate": learning_rate,
        "max_depth": depth,
        "min_child_weight": child,
        "subsample": subsample,
    }
    for learning_rate in (0.05, 0.1)
    for depth in (3, 6)
    for child in (1, 5)
    for subsample in (0.8, 1.0)
]


def main() -> None:
    table, skipped = load_nights(NIGHTS)
    participants = list(dict.fromkeys(table.recording_id.tolist()))
    if len(participants) < 5:
        raise RuntimeError(f"only {len(participants)} participants have features; extraction is not finished")
    print(f"{table.n_epochs} epochs from {len(participants)} participants", flush=True)
    evaluation = evaluate_xgboost(table, GRID, n_splits=5, n_estimators=400, early_stopping_rounds=30, n_jobs=4)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report(table, participants, skipped, evaluation), encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(
        f"error {evaluation.mean['error']:.3f} ± {evaluation.std['error']:.3f} "
        f"(variance {evaluation.variance['error']:.4f})",
        flush=True,
    )


def load_nights(directory: Path) -> tuple[FeatureTable, list[dict]]:
    tables = []
    skipped = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            reason = str(data["reason"])
            if data["rem"].shape[0] == 0:
                skipped.append({"recording_id": path.stem, "reason": reason})
                continue
            tables.append(
                FeatureTable(
                    recording_id=np.array([str(value) for value in data["recording_id"]], dtype=object),
                    onset_sec=np.asarray(data["onset_sec"], dtype=float),
                    rem=np.asarray(data["rem"], dtype=int),
                    values=np.asarray(data["values"], dtype=float),
                    feature_names=FEATURE_NAMES,
                )
            )
    if not tables:
        raise RuntimeError(f"no feature files in {directory}")
    return _stack(tables), skipped


def _stack(tables: list[FeatureTable]) -> FeatureTable:
    return FeatureTable(
        recording_id=np.concatenate([table.recording_id for table in tables]),
        onset_sec=np.concatenate([table.onset_sec for table in tables]),
        rem=np.concatenate([table.rem for table in tables]),
        values=np.vstack([table.values for table in tables]),
        feature_names=FEATURE_NAMES,
    )


def _report(table, participants, skipped, evaluation) -> str:
    rem_rate = float(np.mean(table.rem == 1))
    lines = [
        "# REM detector, all extracted nights",
        "",
        "The label is expert REM versus expert wake and non-REM. Chin EMG is not an input. Diagnosis is not an input. Each recording id is treated as one participant, which is how the CAP files are named.",
        "",
        "Cross-validation is five folds. Each fold holds out about 20 percent of the participants and trains on the other 80 percent. No participant appears in more than one test fold. Inside the training participants, a further 20 percent is used only to compare the parameter grid and to stop adding trees. The winning number of trees is then refit on all training participants. The test participants are not used for either choice.",
        "",
        f"Grid: learning rate 0.05 or 0.1, max depth 3 or 6, min child weight 1 or 5, subsample 0.8 or 1. Early stopping watches log loss, with at most 400 trees and a patience of 30.",
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
        lines.append(
            f"| {label} | {evaluation.mean[key]:.3f} | {evaluation.std[key]:.3f} | {evaluation.variance[key]:.4f} |"
        )
    lines.extend(
        [
            "",
            "Learning curves while the tree count is chosen: `rem_detector_learning_curves.png`.",
            "Learning curves of the refit reported model: `rem_detector_refit_curves.png`.",
            "",
            "## Folds",
            "",
        ]
    )
    for fold in evaluation.folds:
        metrics = fold.metrics
        lines.extend(
            [
                f"### Fold {fold.fold + 1}",
                "",
                f"- Test participants ({len(fold.test_participants)}): {', '.join(fold.test_participants)}.",
                f"- Train epochs {fold.n_train_epochs}, test epochs {fold.n_test_epochs}.",
                f"- Parameters: depth {fold.best_params['max_depth']}, learning rate {fold.best_params['learning_rate']}, "
                f"min child weight {fold.best_params['min_child_weight']}, subsample {fold.best_params['subsample']}, "
                f"trees {fold.best_iteration}.",
                f"- Error {metrics['error']:.3f}, log loss {metrics['log_loss']:.3f}, Brier {metrics['brier']:.3f}.",
                f"- REM recall {metrics['recall']:.3f}, specificity {metrics['specificity']:.3f}, "
                f"precision {metrics['precision']:.3f}, F1 {metrics['f1']:.3f}, "
                f"balanced accuracy {metrics['balanced_accuracy']:.3f}.",
                f"- Confusion matrix: true non-REM {metrics['tn']}, false REM {metrics['fp']}, "
                f"missed REM {metrics['fn']}, true REM {metrics['tp']}.",
                f"- Majority-class error on this fold: {fold.majority_error:.3f}.",
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


if __name__ == "__main__":
    main()
