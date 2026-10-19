import numpy as np

from remguard.emg_features import chin_features


SFREQ = 100.0


def _signal(levels: list[tuple[float, float]]) -> np.ndarray:
    pieces = []
    for duration, value in levels:
        pieces.append(np.full(int(duration * SFREQ), value, dtype=float))
    return np.concatenate(pieces)


def test_known_burst_amplitude_duration_and_epoch_rates():
    # 10 s quiet REM, 10 s elevated REM, then a loud second that is outside REM,
    # then 30 s of quiet sleep for the baseline.
    samples = _signal([(10, 5.0), (10, 20.0), (1, 1000.0), (30, 5.0)])
    features = chin_features(samples, SFREQ, [(0.0, 10.0), (10.0, 20.0)], [(21.0, 51.0)])

    assert features.baseline_uv == 5.0
    assert features.threshold_uv == 10.0
    assert features.rem_duration_sec == 20.0
    assert features.valid_emg_duration_sec == 20.0
    assert features.mean_rectified_uv == 12.5
    assert features.median_rectified_uv == 12.5
    assert features.rms_uv == np.sqrt(212.5)
    assert features.elevated_time_percent == 50.0
    assert features.n_bursts == 1
    assert features.bursts[0].onset_sec == 10.0
    assert features.bursts[0].duration_sec == 10.0
    assert features.burst_duration_median_sec == 10.0
    assert features.bursts_per_hour == 180.0
    assert features.elevated_epoch_percent == 50.0
    assert features.tonic_like_epoch_percent == 50.0
    assert "baseline_overlaps_rem" not in features.notes


def test_activity_outside_rem_does_not_enter_the_features():
    quiet_rem = chin_features(
        _signal([(10, 5.0), (10, 1000.0), (30, 5.0)]),
        SFREQ,
        [(0.0, 10.0)],
        [(20.0, 50.0)],
    )
    assert quiet_rem.mean_rectified_uv == 5.0
    assert quiet_rem.rms_uv == 5.0
    assert quiet_rem.n_bursts == 0
    assert quiet_rem.elevated_time_percent == 0.0


def test_short_spike_is_not_a_burst_and_a_dropout_is_removed():
    samples = _signal([(10, 5.0), (0.05, 40.0), (9.95, 5.0), (30, 5.0)])
    short = chin_features(samples, SFREQ, [(0.0, 20.0)], [(20.0, 50.0)])
    assert short.n_bursts == 0

    held = _signal([(0.1, 40.0), (9.9, 5.0), (30, 5.0)])
    long_enough = chin_features(held, SFREQ, [(0.0, 10.0)], [(10.0, 40.0)])
    assert long_enough.n_bursts == 1
    assert long_enough.bursts[0].duration_sec == 0.1

    with_gap = _signal([(2, 0.0), (8, 5.0), (30, 5.0)])
    gapped = chin_features(with_gap, SFREQ, [(0.0, 10.0)], [(10.0, 40.0)])
    assert gapped.valid_emg_duration_sec == 8.0
    assert gapped.mean_rectified_uv == 5.0
    assert gapped.rem_duration_sec == 10.0


def test_locked_baseline_stays_put_when_the_intervals_overlap_it():
    samples = _signal([(10, 5.0), (30, 100.0)])
    features = chin_features(
        samples,
        SFREQ,
        [(0.0, 10.0)],
        [(0.0, 40.0)],
        locked_baseline=(5.0, 10.0),
    )
    assert features.baseline_uv == 5.0
    assert features.threshold_uv == 10.0
    assert features.mean_rectified_uv == 5.0
    assert "baseline_overlaps_rem" not in features.notes


def test_missing_baseline_leaves_amplitude_and_blocks_the_threshold():
    samples = _signal([(10, 6.0)])
    features = chin_features(samples, SFREQ, [(0.0, 10.0)], [])
    assert features.mean_rectified_uv == 6.0
    assert features.rms_uv == 6.0
    assert np.isnan(features.threshold_uv)
    assert np.isnan(features.elevated_time_percent)
    assert features.n_bursts == 0
    assert "missing_baseline" in features.notes
