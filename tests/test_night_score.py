import mne
import numpy as np

from remguard.night_score import _mne_store, chin_elevated_percent
from remguard.preprocessing import filter_modality


def test_quiet_half_sets_the_threshold_and_scores_only_detected_rem():
    # Two quiet epochs at 5 µV, one loud non-REM epoch at 50 µV, one REM epoch half above 10 µV.
    quiet = np.full(300, 5.0)
    loud = np.full(300, 50.0)
    rem = np.concatenate([np.full(150, 4.0), np.full(150, 30.0)])
    chin = np.concatenate([quiet, loud, quiet, rem])
    onsets = np.array([0.0, 30.0, 60.0, 90.0])
    is_rem = np.array([False, False, False, True])
    elevated, level, threshold = chin_elevated_percent(chin, 10.0, onsets, is_rem)
    assert level == 5.0
    assert threshold == 10.0
    assert elevated == 50.0


class _Modality:
    def __init__(self, role: str, processed, analysis: list[str]):
        self.role = role
        self.processed = processed
        self.analysis_channels = analysis


def _tone(sfreq: float, seconds: float, freq: float, amplitude_uv: float) -> np.ndarray:
    times = np.arange(int(sfreq * seconds)) / sfreq
    return amplitude_uv * np.sin(2 * np.pi * freq * times) * 1e-6


def _power(samples: np.ndarray, sfreq: float, freq: float) -> float:
    trim = int(5 * sfreq)
    segment = samples[trim:-trim]
    window = segment * np.hanning(segment.size)
    spectrum = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(segment.size, 1.0 / sfreq)
    return float(np.abs(spectrum[np.argmin(np.abs(freqs - freq))]) ** 2)


def test_uploaded_signals_are_filtered_and_resampled_with_mne():
    mne.set_log_level("ERROR")
    seconds = 40.0

    eeg_rate = 256.0
    eeg = _tone(eeg_rate, seconds, 10, 20) + _tone(eeg_rate, seconds, 80, 20)
    eeg_raw = mne.io.RawArray(eeg[np.newaxis, :], mne.create_info(["EEG C3"], eeg_rate, ["eeg"]), verbose="ERROR")
    eeg_filtered, _applied = filter_modality(eeg_raw, "eeg")
    eeg_stored = _mne_store(_Modality("eeg", eeg_filtered, ["EEG C3"]))
    assert eeg_stored.info["sfreq"] == 100.0
    eeg_out = eeg_stored.get_data()[0]
    assert _power(eeg_out, 100.0, 10) > 10 * _power(eeg_out, 100.0, 40)

    eog_rate = 256.0
    eog = _tone(eog_rate, seconds, 2, 40) + _tone(eog_rate, seconds, 20, 40)
    eog_raw = mne.io.RawArray(eog[np.newaxis, :], mne.create_info(["EOG ROC"], eog_rate, ["eog"]), verbose="ERROR")
    eog_stored = _mne_store(_Modality("eog", eog_raw, ["EOG ROC"]))
    assert eog_stored.info["sfreq"] == 100.0
    eog_out = eog_stored.get_data()[0]
    assert _power(eog_out, 100.0, 2) > 10 * _power(eog_out, 100.0, 20)

    chin_rate = 512.0
    chin = _tone(chin_rate, seconds, 4, 20) + _tone(chin_rate, seconds, 40, 20)
    chin_raw = mne.io.RawArray(chin[np.newaxis, :], mne.create_info(["EMG chin"], chin_rate, ["emg"]), verbose="ERROR")
    chin_filtered, _applied = filter_modality(chin_raw, "emg_chin")
    chin_stored = _mne_store(_Modality("emg_chin", chin_filtered, ["EMG chin"]))
    assert chin_stored.info["sfreq"] == 256.0
    chin_out = chin_stored.get_data()[0]
    assert _power(chin_out, 256.0, 40) > 10 * _power(chin_out, 256.0, 4)
