import numpy as np
import pytest

from remguard.features import FEATURE_NAMES, epoch_feature_vector, filter_eog_for_features


def _sine(sfreq: float, seconds: float, freq: float, amplitude_uv: float) -> np.ndarray:
    times = np.arange(int(sfreq * seconds)) / sfreq
    return (amplitude_uv * np.sin(2 * np.pi * freq * times))[np.newaxis, :] * 1e-6


def test_feature_names_do_not_include_chin_emg():
    assert not any("emg" in name for name in FEATURE_NAMES)
    assert FEATURE_NAMES[-2:] == ("eog_std_uv", "eog_abs_mean_uv")


def test_relative_power_follows_the_sine_frequency():
    delta = epoch_feature_vector(_sine(100, 30, 2, 20), 100, None)
    alpha = epoch_feature_vector(_sine(100, 30, 10, 20), 100, None)
    assert delta[0] > 0.7
    assert alpha[2] > 0.7
    assert delta[0] > delta[2]
    assert alpha[2] > alpha[0]


def test_a_pure_tone_has_lower_entropy_than_noise():
    tone = epoch_feature_vector(_sine(100, 30, 10, 20), 100, None)
    noise = np.random.default_rng(0).normal(0, 20e-6, size=(1, 3000))
    broad = epoch_feature_vector(noise, 100, None)
    assert tone[5] < broad[5]


def test_amplitude_is_preserved_in_microvolts_and_missing_eog_is_nan():
    features = epoch_feature_vector(_sine(100, 30, 10, 20), 100, None)
    assert features[6] == pytest.approx(20 / np.sqrt(2), rel=0.05)
    assert np.isnan(features[8]) and np.isnan(features[9])
    eog = _sine(100, 30, 2, 50)
    with_eog = epoch_feature_vector(_sine(100, 30, 10, 20), 100, eog)
    assert with_eog[8] == pytest.approx(50 / np.sqrt(2), rel=0.05)


def test_eog_feature_filter_keeps_slow_eye_movement_and_drops_a_fast_tone():
    import mne

    sfreq = 100
    seconds = 60
    slow = _sine(sfreq, seconds, 2, 40)[0]
    fast = _sine(sfreq, seconds, 20, 40)[0]
    info = mne.create_info(["EOG ROC"], sfreq, ch_types=["eog"])
    raw = mne.io.RawArray(np.vstack([slow + fast]), info, verbose="ERROR")
    before = raw.get_data().copy()
    filtered = filter_eog_for_features(raw)
    assert np.allclose(raw.get_data(), before)

    def _bin(values: np.ndarray, freq: float) -> float:
        trim = int(10 * sfreq)
        segment = values[trim:-trim]
        spectrum = np.fft.rfft(segment * np.hanning(segment.size))
        freqs = np.fft.rfftfreq(segment.size, 1.0 / sfreq)
        return float(np.abs(spectrum[np.argmin(np.abs(freqs - freq))]) ** 2)

    original = before[0]
    output = filtered.get_data()[0]
    assert _bin(output, 2) > 0.5 * _bin(original, 2)
    assert _bin(output, 20) < 0.05 * _bin(original, 20)
