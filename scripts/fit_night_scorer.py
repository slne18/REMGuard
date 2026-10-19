"""Fit the REM model used by the upload page and the healthy chin reference.

The reference uses the same no-score chin rule as a new night. Each healthy
night is scored by the fold that held it out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np

from remguard.boosting import _classifier, _positive_weight, participant_folds
from remguard.night_score import (
    MODEL_PARAMS,
    N_ESTIMATORS,
    model_path,
    reference_path,
    score_compact_folder,
)
from train_xgboost_cv import NIGHTS, load_nights

PREPROCESS = ROOT / "data" / "preprocess"


def main() -> None:
    table, _skipped = load_nights(NIGHTS)
    participants = list(dict.fromkeys(table.recording_id.tolist()))
    print(f"Fitting the model on {len(participants)} nights", flush=True)
    model = _fit(table.values, table.rem)
    held_out = _held_out_models(table, participants)
    healthy = []
    for folder in sorted(PREPROCESS.iterdir()):
        if not (folder / "preprocessing.json").is_file() or not _is_healthy(folder.name):
            continue
        if folder.name not in held_out:
            continue
        print(f"  {folder.name}", flush=True)
        try:
            scored = score_compact_folder(folder, held_out[folder.name])
        except ValueError as exc:
            print(f"    skipped ({exc})", flush=True)
            continue
        if scored.elevated_percent is None:
            print("    skipped (no detected REM or no quiet baseline)", flush=True)
            continue
        healthy.append({"recording_id": folder.name, "elevated_percent": round(scored.elevated_percent, 2)})
        print(f"    {scored.elevated_percent:.1f}% of detected REM", flush=True)
    if len(healthy) < 5:
        raise RuntimeError(f"only {len(healthy)} healthy nights could be scored")
    values = np.array([row["elevated_percent"] for row in healthy], dtype=float)
    reference = {
        "recording_ids": [row["recording_id"] for row in healthy],
        "elevated_percent": values.round(2).tolist(),
        "median": round(float(np.median(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "rule": (
            "Share of detected REM in which the chin exceeds twice the median chin size "
            "from the quieter half of epochs the model calls non-REM."
        ),
        "model": {
            "learning_rate": MODEL_PARAMS["learning_rate"],
            "max_depth": MODEL_PARAMS["max_depth"],
            "min_child_weight": MODEL_PARAMS["min_child_weight"],
            "subsample": MODEL_PARAMS["subsample"],
            "n_estimators": N_ESTIMATORS,
        },
    }
    model_path().parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "params": MODEL_PARAMS, "n_estimators": N_ESTIMATORS}, model_path())
    reference_path().write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"Healthy median {reference['median']}% , 90th percentile {reference['p90']}%", flush=True)
    print(f"Wrote {model_path()}", flush=True)
    print(f"Wrote {reference_path()}", flush=True)


def _fit(values: np.ndarray, labels: np.ndarray):
    model = _classifier(MODEL_PARAMS, N_ESTIMATORS, None, _positive_weight(labels), n_jobs=4)
    model.fit(values, labels, verbose=False)
    return model


def _held_out_models(table, participants: list[str]) -> dict:
    models = {}
    for fold_index, test_ids in enumerate(participant_folds(participants, n_splits=5, seed=0)):
        test_set = set(test_ids.tolist())
        train_ids = [recording for recording in participants if recording not in test_set]
        mask = np.isin(table.recording_id, train_ids)
        model = _fit(table.values[mask], table.rem[mask])
        for recording in test_set:
            if _is_healthy(str(recording)):
                models[str(recording)] = model
        print(f"  fold {fold_index + 1} ready", flush=True)
    return models


def _is_healthy(recording_id: str) -> bool:
    name = recording_id.lower()
    return name.startswith("n") and name[1:].isdigit()


if __name__ == "__main__":
    main()
