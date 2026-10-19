import numpy as np

from remguard.detector import (
    evaluate_leave_one_recording_out,
    load_detector,
    majority_predict,
    save_detector,
)
from remguard.features import FEATURE_NAMES, FeatureTable


def _separable_table() -> FeatureTable:
    rng = np.random.default_rng(0)
    ids = []
    rem = []
    rows = []
    for recording_id in ("a", "b", "c"):
        for _epoch in range(30):
            label = 1 if _epoch < 10 else 0
            values = rng.normal(0, 0.05, size=len(FEATURE_NAMES))
            values[0] = 2.0 if label else -2.0
            ids.append(recording_id)
            rem.append(label)
            rows.append(values)
    return FeatureTable(
        recording_id=np.array(ids, dtype=object),
        onset_sec=np.arange(len(ids), dtype=float) * 30,
        rem=np.array(rem, dtype=int),
        values=np.vstack(rows),
        feature_names=FEATURE_NAMES,
    )


def test_leave_one_recording_out_never_trains_on_the_held_out_night():
    evaluation = evaluate_leave_one_recording_out(_separable_table())
    assert [fold.held_out_recording for fold in evaluation.folds] == ["a", "b", "c"]
    for fold in evaluation.folds:
        assert fold.held_out_recording not in fold.train_recordings
        assert fold.n_test == 30
        assert fold.n_train == 60
        assert fold.forest["recall"] > 0.9
    assert evaluation.pooled_forest["recall"] > 0.9
    assert evaluation.pooled_majority["recall"] == 0


def test_majority_baseline_predicts_non_rem_on_a_tie_and_otherwise_the_training_mode():
    assert majority_predict(np.array([0, 1]), 3).tolist() == [0, 0, 0]
    assert majority_predict(np.array([1, 1, 0]), 2).tolist() == [1, 1]


def test_saved_detector_reloads_with_the_same_predictions(tmp_path):
    table = _separable_table()
    from remguard.detector import fit_detector

    pipeline = fit_detector(table)
    path = save_detector(tmp_path / "rem_detector.joblib", pipeline, table)
    loaded, config = load_detector(path)
    assert config["excluded_inputs"] == ["emg_chin"]
    assert tuple(config["feature_names"]) == FEATURE_NAMES
    assert loaded.predict(table.values).tolist() == pipeline.predict(table.values).tolist()
    assert (tmp_path / "rem_detector.json").exists()
