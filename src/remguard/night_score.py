"""Score a new night with the saved REM model and the chin rule.

REM comes from EEG and EOG only. The quiet chin level is the median size
during the quieter half of the epochs the model calls non-REM. A sample in
detected REM is elevated when it is above twice that level. No expert score
is read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

from remguard.features import EPOCH_SEC, epoch_feature_vector, filter_eog_for_features
from remguard.preprocessing import PreprocessedRecording

# Same stored rates as the compact nights the model was trained on.
STORE_HZ = {"eeg": 100.0, "eog": 100.0, "emg_chin": 256.0}
EEG_RATE_HZ = STORE_HZ["eeg"]
CHIN_RATE_HZ = STORE_HZ["emg_chin"]
MODEL_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
}
N_ESTIMATORS = 400


def repo_root() -> Path:
    override = os.environ.get("REMGUARD_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def model_path() -> Path:
    return repo_root() / "models" / "rem_xgboost.joblib"


def reference_path() -> Path:
    return repo_root() / "models" / "healthy_reference.json"


@dataclass
class NightScore:
    elevated_percent: float | None
    rem_minutes: float
    quiet_level_uv: float | None
    threshold_uv: float | None
    duration_sec: float
    rem_blocks: list[tuple[float, float]]
    preview_times: list[float]
    preview_uv: list[float]
    n_rem_epochs: int
    n_epochs: int


def chin_elevated_percent(
    chin_uv: np.ndarray,
    sfreq: float,
    onsets: np.ndarray,
    is_rem: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Return elevated percent, quiet level, and threshold.

    Elevated percent is missing when the model calls no REM. Quiet level is
    missing when the model calls no non-REM.
    """
    quiet_pieces = []
    quiet_medians = []
    rem_pieces = []
    for onset, rem in zip(onsets, is_rem):
        piece = _slice(chin_uv, sfreq, float(onset))
        if piece is None or piece.size == 0:
            continue
        if rem:
            rem_pieces.append(piece)
            continue
        quiet_medians.append(float(np.median(np.abs(piece))))
        quiet_pieces.append(piece)
    if not quiet_pieces:
        return None, None, None
    cutoff = float(np.median(quiet_medians))
    chosen = [piece for piece, median in zip(quiet_pieces, quiet_medians) if median <= cutoff]
    quiet_level = float(np.median(np.abs(np.concatenate(chosen))))
    threshold = 2.0 * quiet_level
    if not rem_pieces:
        return None, quiet_level, threshold
    rem = np.concatenate(rem_pieces)
    elevated = float(100.0 * np.mean(np.abs(rem) > threshold))
    return elevated, quiet_level, threshold


def score_recording(recording: PreprocessedRecording, model) -> NightScore:
    """Score one EDF after the same MNE filtering and resampling as training.

    EEG is already band-passed at 0.3–35 Hz and chin EMG at 10 Hz by
    ``preprocess_recording``. EOG is band-passed here at 0.3–10 Hz. EEG and
    EOG are then resampled to 100 Hz, and chin EMG to 256 Hz, when the native
    rate is higher. ``raw.resample`` is MNE's resampler.
    """
    eeg = _stored_matrix(recording, "eeg", volts=True)
    if eeg is None:
        raise ValueError("the recording has no EEG channel the scorer can use")
    eog = _stored_matrix(recording, "eog", volts=True)
    chin = _stored_matrix(recording, "emg_chin", volts=False)
    if chin is None:
        raise ValueError("the recording has no chin EMG channel")
    return score_signals(
        eeg[0],
        eeg[1],
        None if eog is None else eog[0],
        None if eog is None else eog[1],
        chin[0],
        chin[1],
        model,
    )


def score_compact_folder(folder: Path, model) -> NightScore:
    import json

    import mne

    mne.set_log_level("ERROR")
    payload = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    eeg = _fif_matrix(folder, payload, "eeg", volts=True)
    eog = _fif_matrix(folder, payload, "eog", volts=True)
    chin = _fif_matrix(folder, payload, "emg_chin", volts=False)
    if eeg is None:
        raise ValueError(f"{folder.name} has no EEG")
    if eog is None:
        raise ValueError(f"{folder.name} has no EOG")
    if chin is None:
        raise ValueError(f"{folder.name} has no chin EMG")
    return score_signals(eeg[0], eeg[1], eog[0], eog[1], chin[0], chin[1], model)


def _stored_matrix(recording: PreprocessedRecording, role: str, volts: bool) -> tuple[np.ndarray, float] | None:
    parts = []
    for modality in recording.modalities:
        if modality.role != role:
            continue
        stored = _mne_store(modality)
        if stored is None:
            continue
        data = stored.get_data()
        if not volts:
            data = data * 1e6
        parts.append((data, float(stored.info["sfreq"])))
    return _align(parts, STORE_HZ[role])


def _mne_store(modality) -> mne.io.BaseRaw | None:
    """Filter EOG, pick analysis channels, and resample with MNE."""
    source = modality.processed
    if modality.role == "eog":
        source = filter_eog_for_features(source)
    picks = list(modality.analysis_channels) or list(source.ch_names)
    if modality.role == "eeg" and not modality.analysis_channels:
        return None
    stored = source.copy().pick(picks)
    target = STORE_HZ[modality.role]
    if float(stored.info["sfreq"]) > target + 0.1:
        stored.resample(target, npad="auto", verbose="ERROR")
    return stored


def _fif_matrix(folder: Path, payload: dict, role: str, volts: bool) -> tuple[np.ndarray, float] | None:
    import mne

    parts = []
    for item in payload["modalities"]:
        if item["role"] != role:
            continue
        raw = mne.io.read_raw_fif(folder / item["file"], preload=True, verbose="ERROR")
        picks = item.get("analysis_channels") or list(raw.ch_names)
        data = raw.get_data(picks=picks)
        if not volts:
            data = data * 1e6
        parts.append((data, float(raw.info["sfreq"])))
    return _align(parts, EEG_RATE_HZ if volts else CHIN_RATE_HZ)


def _align(parts: list[tuple[np.ndarray, float]], target: float) -> tuple[np.ndarray, float] | None:
    if not parts:
        return None
    converted = []
    rate = target
    for data, sfreq in parts:
        values, rate = _resample(np.atleast_2d(data), sfreq, target)
        converted.append(values)
    width = min(values.shape[1] for values in converted)
    return np.vstack([values[:, :width] for values in converted]), rate


def score_signals(
    eeg_volts: np.ndarray,
    eeg_sfreq: float,
    eog_volts: np.ndarray | None,
    eog_sfreq: float | None,
    chin_uv: np.ndarray,
    chin_sfreq: float,
    model,
) -> NightScore:
    eeg, eeg_rate = _resample(np.atleast_2d(eeg_volts), eeg_sfreq, EEG_RATE_HZ)
    eog = None
    eog_rate = None
    if eog_volts is not None and eog_sfreq is not None:
        eog, eog_rate = _resample(np.atleast_2d(eog_volts), eog_sfreq, EEG_RATE_HZ)
    chin, chin_rate = _resample(np.atleast_2d(chin_uv), chin_sfreq, CHIN_RATE_HZ)
    chin = np.mean(chin, axis=0)
    duration = min(_duration(eeg, eeg_rate), _duration(chin, chin_rate))
    if eog is not None and eog_rate is not None:
        duration = min(duration, _duration(eog, eog_rate))
    onsets = np.arange(0.0, duration - EPOCH_SEC + 1e-6, EPOCH_SEC)
    if onsets.size == 0:
        raise ValueError("the recording is shorter than one 30-second epoch")
    rows = []
    kept = []
    for onset in onsets:
        eeg_piece = _slice_2d(eeg, eeg_rate, float(onset))
        if eeg_piece is None:
            continue
        eog_piece = None if eog is None or eog_rate is None else _slice_2d(eog, eog_rate, float(onset))
        rows.append(epoch_feature_vector(eeg_piece, eeg_rate, eog_piece))
        kept.append(float(onset))
    if not rows:
        raise ValueError("no complete 30-second epochs could be read")
    predicted = model.predict(np.vstack(rows)).astype(int)
    kept_onsets = np.array(kept, dtype=float)
    is_rem = predicted == 1
    elevated, quiet, threshold = chin_elevated_percent(chin, chin_rate, kept_onsets, is_rem)
    blocks = _blocks(kept_onsets[is_rem])
    times, values = _preview(chin, chin_rate, blocks)
    return NightScore(
        elevated_percent=elevated,
        rem_minutes=float(np.sum(is_rem) * EPOCH_SEC / 60.0),
        quiet_level_uv=quiet,
        threshold_uv=threshold,
        duration_sec=float(duration),
        rem_blocks=blocks,
        preview_times=times,
        preview_uv=values,
        n_rem_epochs=int(np.sum(is_rem)),
        n_epochs=int(kept_onsets.size),
    )


def _resample(values: np.ndarray, sfreq: float, target: float) -> tuple[np.ndarray, float]:
    if sfreq <= target + 0.5:
        return values, float(sfreq)
    up = int(round(target))
    down = int(round(sfreq))
    divisor = np.gcd(up, down)
    up //= int(divisor)
    down //= int(divisor)
    converted = np.vstack([resample_poly(channel, up, down) for channel in values])
    return converted, float(sfreq) * up / down


def _duration(values: np.ndarray, sfreq: float) -> float:
    return float(values.shape[-1] / sfreq)


def _slice(samples: np.ndarray, sfreq: float, onset: float) -> np.ndarray | None:
    start = int(np.round(onset * sfreq))
    stop = start + int(np.round(EPOCH_SEC * sfreq))
    if start < 0 or stop > samples.shape[-1]:
        return None
    return np.asarray(samples[start:stop], dtype=float)


def _slice_2d(samples: np.ndarray, sfreq: float, onset: float) -> np.ndarray | None:
    start = int(np.round(onset * sfreq))
    stop = start + int(np.round(EPOCH_SEC * sfreq))
    if start < 0 or stop > samples.shape[-1]:
        return None
    return np.asarray(samples[:, start:stop], dtype=float)


def _blocks(onsets: np.ndarray) -> list[tuple[float, float]]:
    if onsets.size == 0:
        return []
    ordered = np.sort(onsets)
    blocks = []
    start = float(ordered[0])
    previous = start
    for onset in ordered[1:]:
        if float(onset) - previous > EPOCH_SEC + 0.5:
            blocks.append((start, previous + EPOCH_SEC))
            start = float(onset)
        previous = float(onset)
    blocks.append((start, previous + EPOCH_SEC))
    return blocks


def _preview(
    chin: np.ndarray,
    sfreq: float,
    blocks: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    if not blocks:
        return [], []
    start, end = max(blocks, key=lambda block: block[1] - block[0])
    window_end = min(end, start + 20.0)
    i0 = int(np.round(start * sfreq))
    i1 = int(np.round(window_end * sfreq))
    piece = chin[i0:i1]
    if piece.size == 0:
        return [], []
    step = max(1, int(np.ceil(piece.size / 600)))
    shown = piece[::step]
    times = (start + np.arange(shown.size) * step / sfreq).tolist()
    return [float(value) for value in times], [float(value) for value in shown]
