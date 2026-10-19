"""Epoch features for the REM detector.

Chin EMG is not an input. The detector has to recognize REM from EEG and eye
movements before any muscle-activity score is calculated.

EEG features use the Phase 2 analysis signal, averaged across the channels
kept for sleep staging. Relative power uses a 0.5–30 Hz total so a channel
that was already low-passed near 30 Hz is not penalized for empty bins above
that. EOG stays unfiltered in Phase 2; this module filters a copy to 0.3–10 Hz
only while measuring eye-movement amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from scipy.signal import welch

from remguard.data.epochs import EpochTimeline
from remguard.preprocessing import EPOCH_SEC, PreprocessedRecording

EEG_BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)
TOTAL_BAND = (0.5, 30.0)
WELCH_WINDOW_SEC = 4.0
EOG_L_FREQ = 0.3
EOG_H_FREQ = 10.0
EOG_L_TRANS_HZ = 0.3
EOG_H_TRANS_HZ = 2.0

FEATURE_NAMES = (
    "eeg_rel_delta",
    "eeg_rel_theta",
    "eeg_rel_alpha",
    "eeg_rel_sigma",
    "eeg_rel_beta",
    "eeg_spectral_entropy",
    "eeg_std_uv",
    "eeg_abs_mean_uv",
    "eog_std_uv",
    "eog_abs_mean_uv",
)


@dataclass
class FeatureTable:
    recording_id: np.ndarray
    onset_sec: np.ndarray
    rem: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def n_epochs(self) -> int:
        return int(self.values.shape[0])


def empty_table() -> FeatureTable:
    return FeatureTable(
        recording_id=np.array([], dtype=object),
        onset_sec=np.array([], dtype=float),
        rem=np.array([], dtype=int),
        values=np.zeros((0, len(FEATURE_NAMES))),
        feature_names=FEATURE_NAMES,
    )


def concatenate_tables(tables: list[FeatureTable]) -> FeatureTable:
    present = [table for table in tables if table.n_epochs]
    if not present:
        return empty_table()
    return FeatureTable(
        recording_id=np.concatenate([table.recording_id for table in present]),
        onset_sec=np.concatenate([table.onset_sec for table in present]),
        rem=np.concatenate([table.rem for table in present]),
        values=np.vstack([table.values for table in present]),
        feature_names=FEATURE_NAMES,
    )


def extract_recording_features(
    recording: PreprocessedRecording,
    timeline: EpochTimeline,
) -> FeatureTable:
    """Features for epochs that have a clean REM or non-REM expert label."""
    eeg = recording.modality("eeg")
    if eeg is None:
        return empty_table()
    labeled = [epoch for epoch in timeline.epochs if epoch.rem is not None]
    if not labeled:
        return empty_table()

    picks = eeg.analysis_channels or list(eeg.processed.ch_names)
    eeg_sfreq = float(eeg.processed.info["sfreq"])
    eog_raw = _eog_for_features(recording)
    eog_sfreq = None if eog_raw is None else float(eog_raw.info["sfreq"])

    rows = []
    onsets = []
    rem = []
    for epoch in labeled:
        eeg_data = _slice(eeg.processed, picks, epoch.onset_sec, eeg_sfreq)
        if eeg_data is None:
            continue
        eog_data = None
        if eog_raw is not None and eog_sfreq is not None:
            eog_data = _slice(eog_raw, list(eog_raw.ch_names), epoch.onset_sec, eog_sfreq)
        rows.append(epoch_feature_vector(eeg_data, eeg_sfreq, eog_data))
        onsets.append(epoch.onset_sec)
        rem.append(int(epoch.rem))
    if not rows:
        return empty_table()
    return FeatureTable(
        recording_id=np.array([recording.recording_id] * len(rows), dtype=object),
        onset_sec=np.asarray(onsets, dtype=float),
        rem=np.asarray(rem, dtype=int),
        values=np.vstack(rows),
        feature_names=FEATURE_NAMES,
    )


def epoch_feature_vector(
    eeg_volts: np.ndarray,
    eeg_sfreq: float,
    eog_volts: np.ndarray | None,
) -> np.ndarray:
    """Average per-channel EEG features, then append EOG amplitude features."""
    channel_rows = [_eeg_channel_features(channel, eeg_sfreq) for channel in np.atleast_2d(eeg_volts)]
    eeg_features = np.mean(np.vstack(channel_rows), axis=0)
    if eog_volts is None:
        eog_features = np.array([np.nan, np.nan])
    else:
        eog_rows = [_amplitude_features(channel) for channel in np.atleast_2d(eog_volts)]
        eog_features = np.mean(np.vstack(eog_rows), axis=0)
    return np.concatenate([eeg_features, eog_features])


def filter_eog_for_features(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a 0.3–10 Hz copy. ``raw`` is not modified."""
    processed = raw.copy()
    nyquist = float(raw.info["sfreq"]) / 2.0
    highpass = EOG_L_FREQ if nyquist > EOG_L_FREQ + 1.0 else None
    lowpass = EOG_H_FREQ if nyquist > EOG_H_FREQ + 5.0 else None
    if highpass is None and lowpass is None:
        return processed
    kwargs = {
        "l_freq": highpass,
        "h_freq": lowpass,
        "picks": "all",
        "phase": "zero",
        "fir_design": "firwin",
        "fir_window": "hamming",
        "n_jobs": 1,
        "verbose": "ERROR",
    }
    if highpass is not None:
        kwargs["l_trans_bandwidth"] = EOG_L_TRANS_HZ
    if lowpass is not None:
        kwargs["h_trans_bandwidth"] = EOG_H_TRANS_HZ
    processed.filter(**kwargs)
    return processed


def _eog_for_features(recording: PreprocessedRecording) -> mne.io.BaseRaw | None:
    eog = recording.modality("eog")
    if eog is None:
        return None
    return filter_eog_for_features(eog.original)


def _slice(raw: mne.io.BaseRaw, picks: list[str], onset_sec: float, sfreq: float) -> np.ndarray | None:
    start = int(np.round(onset_sec * sfreq))
    stop = start + int(np.round(EPOCH_SEC * sfreq))
    if start < 0 or stop > raw.n_times:
        return None
    return raw.get_data(picks=picks, start=start, stop=stop)


def _eeg_channel_features(samples_volts: np.ndarray, sfreq: float) -> np.ndarray:
    amplitude = _amplitude_features(samples_volts)
    nperseg = min(samples_volts.size, max(32, int(WELCH_WINDOW_SEC * sfreq)))
    freqs, power = welch(samples_volts, fs=sfreq, nperseg=nperseg)
    total = _band_power(freqs, power, TOTAL_BAND[0], TOTAL_BAND[1])
    relative = []
    for _name, low, high in EEG_BANDS:
        band = _band_power(freqs, power, low, high)
        relative.append(0.0 if total <= 0 else band / total)
    return np.array(relative + [_spectral_entropy(freqs, power), amplitude[0], amplitude[1]], dtype=float)


def _amplitude_features(samples_volts: np.ndarray) -> np.ndarray:
    microvolts = np.asarray(samples_volts, dtype=float) * 1e6
    return np.array([float(np.std(microvolts)), float(np.mean(np.abs(microvolts)))], dtype=float)


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high if high < TOTAL_BAND[1] else freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask]))


def _spectral_entropy(freqs: np.ndarray, power: np.ndarray) -> float:
    mask = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    selected = np.asarray(power[mask], dtype=float)
    total = float(np.sum(selected))
    if selected.size < 2 or total <= 0:
        return 0.0
    probability = selected / total
    probability = probability[probability > 0]
    entropy = float(-np.sum(probability * np.log(probability)))
    return entropy / float(np.log(selected.size))
