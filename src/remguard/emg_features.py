"""Chin-EMG features measured inside REM intervals.

Amplitude features use only samples that fall inside the REM intervals passed
in. A quiet-sleep baseline, taken from outside REM, sets the elevated-activity
threshold. Wake samples are not used for either piece.

The threshold is twice the median rectified amplitude of that baseline. It is
not a fixed microvolt cutoff. Absolute microvolts are not treated as comparable
across recordings, because the CAP files do not share one calibration.

Bursts are contiguous stretches of at least 0.1 s where the rectified signal
exceeds the threshold. Formal clinical RBD rules are not applied: those rules
need a calibrated reference these headers do not provide. ``tonic_like_epoch_percent``
is the share of REM epochs in which at least half of the valid samples are
above the same threshold. It is an exploratory description, not a tonic score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THRESHOLD_MULTIPLE = 2.0
MIN_BURST_SEC = 0.1
MIN_BASELINE_SEC = 30.0
FLAT_CHANNEL_STD_UV = 1.0
DROPOUT_UV = 0.01
TONIC_LIKE_FRACTION = 0.5


@dataclass(frozen=True)
class Burst:
    onset_sec: float
    duration_sec: float


@dataclass(frozen=True)
class EmgFeatures:
    rem_duration_sec: float
    valid_emg_duration_sec: float
    mean_rectified_uv: float
    median_rectified_uv: float
    rms_uv: float
    variability_uv: float
    elevated_time_percent: float
    bursts_per_hour: float
    burst_duration_median_sec: float
    burst_duration_p95_sec: float
    elevated_epoch_percent: float
    tonic_like_epoch_percent: float
    n_bursts: int
    baseline_uv: float
    threshold_uv: float
    bursts: tuple[Burst, ...]
    notes: tuple[str, ...]


def chin_features(
    samples_uv: np.ndarray,
    sfreq: float,
    rem_intervals: list[tuple[float, float]],
    baseline_intervals: list[tuple[float, float]],
    locked_baseline: tuple[float, float] | None = None,
) -> EmgFeatures:
    """Features of one chin channel. ``samples_uv`` is the whole recording.

    ``locked_baseline`` is ``(baseline_uv, threshold_uv)`` from an earlier call.
    Predicted-REM features pass the expert-S2 baseline so a false REM label
    cannot move the threshold.
    """
    samples = np.asarray(samples_uv, dtype=float)
    notes: list[str] = []
    if samples.size and float(np.nanstd(samples)) < FLAT_CHANNEL_STD_UV:
        notes.append("flat_channel")

    rem_mask = _interval_mask(samples.size, sfreq, rem_intervals)
    dropout = _dropout_mask(samples, sfreq)
    valid_rem = rem_mask & ~dropout & np.isfinite(samples)
    rem_duration = float(sum(max(0.0, end - start) for start, end in rem_intervals))
    valid_duration = float(np.sum(valid_rem) / sfreq) if sfreq else 0.0
    if locked_baseline is None:
        baseline_mask = _interval_mask(samples.size, sfreq, baseline_intervals)
        valid_baseline = baseline_mask & ~dropout & np.isfinite(samples)
        if np.any(baseline_mask & rem_mask):
            notes.append("baseline_overlaps_rem")
            valid_baseline &= ~rem_mask
        baseline_uv, threshold_uv = _threshold(samples, valid_baseline, sfreq, notes)
    else:
        baseline_uv, threshold_uv = locked_baseline
    rectified = np.abs(samples)

    if not np.any(valid_rem):
        notes.append("no_valid_rem_emg")
        return _empty(rem_duration, baseline_uv, threshold_uv, notes)

    used = rectified[valid_rem]
    raw = samples[valid_rem]
    bursts = () if threshold_uv != threshold_uv else _bursts(rectified, sfreq, threshold_uv, valid_rem)
    durations = np.array([burst.duration_sec for burst in bursts], dtype=float)
    elevated_epoch, tonic_like = _epoch_rates(samples, sfreq, rem_intervals, dropout, threshold_uv)
    return EmgFeatures(
        rem_duration_sec=rem_duration,
        valid_emg_duration_sec=valid_duration,
        mean_rectified_uv=float(np.mean(used)),
        median_rectified_uv=float(np.median(used)),
        rms_uv=float(np.sqrt(np.mean(raw * raw))),
        variability_uv=float(np.std(used)),
        elevated_time_percent=_elevated_percent(used, threshold_uv),
        bursts_per_hour=_bursts_per_hour(len(bursts), valid_duration),
        burst_duration_median_sec=float(np.median(durations)) if durations.size else float("nan"),
        burst_duration_p95_sec=float(np.percentile(durations, 95)) if durations.size else float("nan"),
        elevated_epoch_percent=elevated_epoch,
        tonic_like_epoch_percent=tonic_like,
        n_bursts=len(bursts),
        baseline_uv=baseline_uv,
        threshold_uv=threshold_uv,
        bursts=bursts,
        notes=tuple(notes),
    )


def baseline_intervals_for(timeline, stages: tuple[str, ...] = ("S2", "S3", "S4")) -> tuple[list[tuple[float, float]], str | None]:
    """Quiet-sleep intervals used only to set the threshold. Wake is not included."""
    for stage in stages:
        intervals = [
            (epoch.onset_sec, epoch.end_sec)
            for epoch in timeline.epochs
            if epoch.fully_annotated and epoch.stage == stage and epoch.rem is False
        ]
        duration = sum(end - start for start, end in intervals)
        if duration >= MIN_BASELINE_SEC:
            return intervals, stage
    return [], None


def expert_rem_intervals(timeline) -> list[tuple[float, float]]:
    return [
        (epoch.onset_sec, epoch.end_sec)
        for epoch in timeline.epochs
        if epoch.rem is True
    ]


def _threshold(samples: np.ndarray, valid_baseline: np.ndarray, sfreq: float, notes: list[str]) -> tuple[float, float]:
    baseline_sec = float(np.sum(valid_baseline) / sfreq) if sfreq else 0.0
    if baseline_sec < 1.0:
        notes.append("missing_baseline")
        return float("nan"), float("nan")
    if baseline_sec < MIN_BASELINE_SEC:
        notes.append("short_baseline")
    baseline = float(np.median(np.abs(samples[valid_baseline])))
    return baseline, THRESHOLD_MULTIPLE * baseline


def _elevated_percent(rectified_valid: np.ndarray, threshold_uv: float) -> float:
    if threshold_uv != threshold_uv or rectified_valid.size == 0:
        return float("nan")
    return float(100.0 * np.mean(rectified_valid > threshold_uv))


def _bursts_per_hour(n_bursts: int, valid_duration_sec: float) -> float:
    if valid_duration_sec <= 0:
        return float("nan")
    return float(n_bursts / (valid_duration_sec / 3600.0))


def _bursts(rectified: np.ndarray, sfreq: float, threshold_uv: float, valid: np.ndarray) -> tuple[Burst, ...]:
    above = valid & (rectified > threshold_uv)
    padded = np.concatenate(([False], above, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    min_samples = max(1, int(round(MIN_BURST_SEC * sfreq)))
    bursts = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            bursts.append(Burst(onset_sec=start / sfreq, duration_sec=(end - start) / sfreq))
    return tuple(bursts)


def _epoch_rates(
    samples: np.ndarray,
    sfreq: float,
    rem_intervals: list[tuple[float, float]],
    dropout: np.ndarray,
    threshold_uv: float,
) -> tuple[float, float]:
    if threshold_uv != threshold_uv or not rem_intervals:
        return float("nan"), float("nan")
    considered = 0
    elevated = 0
    tonic = 0
    for start, end in rem_intervals:
        i0 = max(0, int(np.round(start * sfreq)))
        i1 = min(samples.size, int(np.round(end * sfreq)))
        if i1 <= i0:
            continue
        region = np.isfinite(samples[i0:i1]) & ~dropout[i0:i1]
        if int(np.sum(region)) < 0.5 * (i1 - i0):
            continue
        segment = samples[i0:i1][region]
        considered += 1
        rms = float(np.sqrt(np.mean(segment * segment)))
        if rms > threshold_uv:
            elevated += 1
        if float(np.mean(np.abs(segment) > threshold_uv)) >= TONIC_LIKE_FRACTION:
            tonic += 1
    if considered == 0:
        return float("nan"), float("nan")
    return 100.0 * elevated / considered, 100.0 * tonic / considered


def _dropout_mask(samples: np.ndarray, sfreq: float) -> np.ndarray:
    """Mark 1 s stretches that are numerically zero. Quiet atonia stays in."""
    dropout = np.zeros(samples.size, dtype=bool)
    window = int(round(sfreq))
    if window < 1:
        return dropout
    for start in range(0, samples.size, window):
        stop = min(samples.size, start + window)
        segment = samples[start:stop]
        finite = segment[np.isfinite(segment)]
        if finite.size < 0.5 * window:
            dropout[start:stop] = True
            continue
        if float(np.mean(np.abs(finite))) < DROPOUT_UV and float(np.std(finite)) < DROPOUT_UV:
            dropout[start:stop] = True
    return dropout


def _interval_mask(n_samples: int, sfreq: float, intervals: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    if sfreq <= 0:
        return mask
    for start, end in intervals:
        i0 = max(0, int(np.round(start * sfreq)))
        i1 = min(n_samples, int(np.round(end * sfreq)))
        if i1 > i0:
            mask[i0:i1] = True
    return mask


def _empty(rem_duration: float, baseline_uv: float, threshold_uv: float, notes: list[str]) -> EmgFeatures:
    nan = float("nan")
    return EmgFeatures(
        rem_duration_sec=rem_duration,
        valid_emg_duration_sec=0.0,
        mean_rectified_uv=nan,
        median_rectified_uv=nan,
        rms_uv=nan,
        variability_uv=nan,
        elevated_time_percent=nan,
        bursts_per_hour=nan,
        burst_duration_median_sec=nan,
        burst_duration_p95_sec=nan,
        elevated_epoch_percent=nan,
        tonic_like_epoch_percent=nan,
        n_bursts=0,
        baseline_uv=baseline_uv,
        threshold_uv=threshold_uv,
        bursts=(),
        notes=tuple(notes),
    )
