import numpy as np

from remguard.boosting import evaluate_xgboost, inner_participant_split, participant_folds
from remguard.features import FEATURE_NAMES, FeatureTable


def _table() -> FeatureTable:
    rng = np.random.default_rng(0)
    ids = []
    rem = []
    rows = []
    for index in range(10):
        for epoch in range(24):
            label = 1 if epoch < 8 else 0
            values = rng.normal(0, 0.05, size=len(FEATURE_NAMES))
            values[0] = 3.0 if label else -3.0
            if index == 9:
                values[-2:] = np.nan
            ids.append(f"p{index}")
            rem.append(label)
            rows.append(values)
    return FeatureTable(
        recording_id=np.array(ids, dtype=object),
        onset_sec=np.arange(len(ids), dtype=float),
        rem=np.array(rem, dtype=int),
        values=np.vstack(rows),
        feature_names=FEATURE_NAMES,
    )


def test_each_participant_is_held_out_once_and_the_test_share_is_about_one_fifth():
    folds = participant_folds([f"p{index}" for index in range(10)], n_splits=5, seed=1)
    held = [set(chunk.tolist()) for chunk in folds]
    assert [len(chunk) for chunk in held] == [2, 2, 2, 2, 2]
    assert set.union(*held) == {f"p{index}" for index in range(10)}
    assert not set.intersection(*held)


def test_inner_split_holds_out_twenty_percent_of_training_participants():
    fit_ids, val_ids = inner_participant_split(np.array([f"p{index}" for index in range(10)]), seed=2)
    assert len(val_ids) == 2
    assert len(fit_ids) == 8
    assert set(fit_ids).isdisjoint(set(val_ids))


def test_grouped_xgboost_reports_error_and_variance_without_leaking_participants():
    grid = [
        {"learning_rate": 0.2, "max_depth": 2, "min_child_weight": 1, "subsample": 1.0},
        {"learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1, "subsample": 0.8},
    ]
    evaluation = evaluate_xgboost(
        _table(),
        grid,
        n_splits=5,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
    )
    seen = []
    for fold in evaluation.folds:
        assert set(fold.test_participants).isdisjoint(fold.train_participants)
        assert fold.n_test_epochs == 48
        assert fold.best_iteration >= 1
        seen.extend(fold.test_participants)
    assert sorted(seen) == [f"p{index}" for index in range(10)]
    assert evaluation.mean["error"] < evaluation.mean["majority_error"]
    assert evaluation.variance["error"] >= 0
    assert evaluation.std["error"] == np.sqrt(evaluation.variance["error"])
