"""EEG and chin-EMG preprocessing for overnight CAP recordings.

Decisions, kept with the code so later phases use the same settings:

- EEG and chin EMG are filtered separately. EOG is loaded and typed, and is
  left unfiltered until the REM detector defines an eye-movement band.
- Filters are zero-phase FIR (MNE ``phase="zero"``). A centered FIR uses
  future samples within the filter half-length, so these outputs are for
  completed recordings and are not a real-time implementation.
- No project-wide resampling. Channels are read in groups that share a native
  sampling frequency, so a 128 Hz EOG channel is not upsampled to 512 Hz.
- Signals stay in the physical units MNE returns (volts). Reports and plots
  convert volts to microvolts for reading. Nothing is z-scored.
- EEG bandpass is 0.3–35 Hz (sleep-stage range). Transition widths are 0.3 Hz
  and 5 Hz.
- Chin EMG highpass is 10 Hz. The lowpass is 100 Hz when the sampling rate
  allows it, otherwise 10 Hz below Nyquist. A 50 Hz notch is applied only when
  that frequency is still inside the EMG band, at least 5 Hz below Nyquist,
  and a middle-of-night spectrum shows a line-noise peak (power ratio >= 3).
- 30 s stage epochs follow expert SLEEP-* rows from the REMlogic file. Rows
  that are ambiguous, overlapping, not 30 s, or outside the EDF are listed
  and omitted. Epoch baselines are not subtracted.
- Inside expert REM intervals, chin EMG is also cut into non-overlapping 1 s
  windows. Windows that would run past the interval are dropped.
- Flat channels (standard deviation below 1 µV) are flagged. Header physical
  rails count as saturation only when the digital minimum is below the digital
  maximum and the samples stay inside that physical range. A reversed digital
  range is reported as a header inconsistency. High EEG amplitude is flagged.
  High chin-EMG amplitude is not treated as noise, because that is the
  activity later phases measure.
- The EDF prefiltering field is stored as an acquisition property. A channel is
  not filtered again when that field already matches the analysis band. EEG
  stored with a highpass at or above 5 Hz is kept for inspection and left out
  of the sleep-stage epochs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

from remguard.data.annotations import ScoringFile, SleepEvent, parse_scoring_file
from remguard.data.edf_header import EdfHeader, read_edf_header

EEG_L_FREQ = 0.3
EEG_H_FREQ = 35.0
EEG_L_TRANS_HZ = 0.3
EEG_H_TRANS_HZ = 5.0
EMG_L_FREQ = 10.0
EMG_PREFERRED_H_FREQ = 100.0
EMG_L_TRANS_HZ = 2.0
EMG_H_TRANS_HZ = 5.0
LINE_HZ = 50.0
LINE_NOISE_RATIO = 3.0
NOTCH_NYQUIST_MARGIN_HZ = 5.0
EPOCH_SEC = 30.0
REM_EMG_WINDOW_SEC = 1.0
FLAT_STD_UV = 1.0
SATURATION_FRACTION = 0.01
EEG_HIGH_AMPLITUDE_UV = 500.0

_LIMB = re.compile(r"(tib|delto|dx\d|sx\d)", re.IGNORECASE)
_CHIN = re.compile(r"(chin|milo|subment|emg\s*-?\s*emg|emg1\s*-?\s*emg2)", re.IGNORECASE)
_BARE_EMG = re.compile(r"^emg$", re.IGNORECASE)
_EOG = re.compile(r"(eog|loc|roc)", re.IGNORECASE)
_ECG = re.compile(r"(ecg|ekg|heart)", re.IGNORECASE)
_OXYGEN = re.compile(r"s[ap]\s*o\s*2", re.IGNORECASE)
# Electrode names may be glued to the reference, as in F3A2. They still have to
# start at a boundary so the O2 inside SAO2 is not read as occipital EEG.
_EEG = re.compile(
    r"(?:^|[^a-z0-9])(fp[12]|f[3478z]|c[34z]|p[34z]|o[12z]|t[3456])",
    re.IGNORECASE,
)

MNE_TYPES = {"eeg": "eeg", "eog": "eog", "emg_chin": "emg"}
STAGE_CODES = {"W": 1, "S1": 2, "S2": 3, "S3": 4, "S4": 5, "R": 6, "MT": 7}


@dataclass(frozen=True)
class ChannelRole:
    role: str
    assumed: bool = False
    ambiguous_label: bool = False


@dataclass(frozen=True)
class ChannelAssignment:
    edf_label: str
    mne_name: str
    canonical_name: str
    role: str
    native_sfreq_hz: float
    original_unit: str
    physical_min_volts: float | None
    physical_max_volts: float | None
    assumed: bool
    ambiguous_label: bool
    digital_min: float | None = None
    digital_max: float | None = None
    prefiltering: str = ""


@dataclass(frozen=True)
class AppliedFilter:
    role: str
    sfreq_hz: float
    l_freq_hz: float | None
    h_freq_hz: float | None
    notch_hz: float | None
    phase: str
    method: str
    line_noise_ratio: float | None
    channels_filtered: tuple[str, ...] = ()
    channels_retained: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChannelQuality:
    canonical_name: str
    role: str
    std_uv: float
    saturation_fraction: float
    flat: bool
    saturated: bool
    notes: tuple[str, ...]


@dataclass
class PreprocessedModality:
    role: str
    sfreq_hz: float
    original: mne.io.BaseRaw
    processed: mne.io.BaseRaw
    channels: list[ChannelAssignment]
    quality: list[ChannelQuality]
    applied_filter: AppliedFilter
    analysis_channels: list[str] = field(default_factory=list)


@dataclass
class PreprocessedRecording:
    recording_id: str
    group: str
    edf_path: str
    annotation_path: str | None
    header: EdfHeader
    scoring: ScoringFile | None
    modalities: list[PreprocessedModality]
    ignored_channels: list[ChannelAssignment]
    stage_epochs: list[mne.Epochs]
    stage_events_kept: list[SleepEvent]
    stage_events_skipped: list[tuple[str, str, str]]
    rem_emg_epochs: mne.Epochs | None
    issues: list[str] = field(default_factory=list)
    gaps: list[dict[str, float | str]] = field(default_factory=list)

    def modality(self, role: str) -> PreprocessedModality | None:
        return next((item for item in self.modalities if item.role == role), None)


def classify_label(label: str) -> ChannelRole:
    """Map one EDF label onto a role. Unknown labels become misc."""
    if _LIMB.search(label):
        return ChannelRole("emg_limb")
    if _ECG.search(label):
        return ChannelRole("ecg")
    if _CHIN.search(label):
        ambiguous = "milo" in label.lower()
        return ChannelRole("emg_chin", ambiguous_label=ambiguous)
    if _BARE_EMG.search(label.strip()):
        return ChannelRole("emg_chin", assumed=True)
    if _EOG.search(label):
        return ChannelRole("eog")
    if _OXYGEN.search(label):
        return ChannelRole("misc")
    if _EEG.search(label):
        return ChannelRole("eeg")
    return ChannelRole("misc")


def group_label(recording_id: str) -> str:
    if recording_id.lower().startswith("rbd"):
        return "rbd"
    if re.fullmatch(r"n\d+", recording_id.lower()):
        return "control"
    return "unknown"


def emg_h_freq(sfreq_hz: float) -> float | None:
    """Upper EMG cutoff that the native sampling rate can support."""
    nyquist = sfreq_hz / 2.0
    if nyquist - 10.0 >= EMG_PREFERRED_H_FREQ:
        return EMG_PREFERRED_H_FREQ
    cutoff = nyquist - 10.0
    if cutoff <= EMG_L_FREQ + 5.0:
        return None
    return cutoff


def preprocess_recording(
    edf_path: str | Path,
    annotation_path: str | Path | None = None,
) -> PreprocessedRecording:
    edf_path = Path(edf_path)
    header = read_edf_header(edf_path)
    recording_id = edf_path.stem
    issues: list[str] = []
    if header.has_embedded_annotations:
        issues.append("edf_contains_annotations")
    else:
        issues.append("scores_not_embedded_in_edf")
    if header.n_records <= 0:
        issues.append("edf_record_count_unusable")

    loaded, ignored = _load_modalities(edf_path, header)
    loaded_roles = {item[0] for item in loaded}
    if "eeg" not in loaded_roles:
        issues.append("missing_eeg")
    if "emg_chin" not in loaded_roles:
        issues.append("missing_emg_chin")
    if "eog" not in loaded_roles:
        issues.append("missing_eog")

    modalities = [_process_modality(item) for item in loaded]
    scoring = None
    annotation_text_path = None
    if annotation_path is not None:
        annotation_text_path = str(annotation_path)
        scoring = parse_scoring_file(annotation_path, header.start_time)
        if _calendar_dates_disagree(header, scoring.recording_date):
            issues.append("annotation_calendar_date_disagrees_with_edf")

    duration = _duration_sec(modalities, header)
    gaps = find_gaps(scoring.events, duration) if scoring is not None else []
    kept, skipped = _select_stage_events(scoring, duration) if scoring is not None else ([], [])
    stage_epochs = _stage_epochs(modalities, kept)
    rem_epochs = _rem_emg_epochs(modalities, scoring, duration) if scoring is not None else None
    return PreprocessedRecording(
        recording_id=recording_id,
        group=group_label(recording_id),
        edf_path=str(edf_path),
        annotation_path=annotation_text_path,
        header=header,
        scoring=scoring,
        modalities=modalities,
        ignored_channels=ignored,
        stage_epochs=stage_epochs,
        stage_events_kept=kept,
        stage_events_skipped=skipped,
        rem_emg_epochs=rem_epochs,
        issues=issues,
        gaps=gaps,
    )


def find_gaps(
    events: list[SleepEvent],
    duration_sec: float,
    minimum_sec: float = 0.5,
) -> list[dict[str, float | str]]:
    timed = [event for event in events if event.onset_sec == event.onset_sec]
    gaps: list[dict[str, float | str]] = []
    if not timed:
        return [{"start_sec": 0.0, "end_sec": duration_sec, "kind": "no_scores"}]
    if timed[0].onset_sec > minimum_sec:
        gaps.append({"start_sec": 0.0, "end_sec": timed[0].onset_sec, "kind": "leading"})
    for previous, current in zip(timed, timed[1:]):
        gap = current.onset_sec - previous.end_sec
        if gap > minimum_sec:
            gaps.append(
                {
                    "start_sec": previous.end_sec,
                    "end_sec": current.onset_sec,
                    "kind": "internal",
                }
            )
    if duration_sec - timed[-1].end_sec > minimum_sec:
        gaps.append(
            {
                "start_sec": timed[-1].end_sec,
                "end_sec": duration_sec,
                "kind": "trailing",
            }
        )
    return gaps


def acquisition_band(text: str) -> tuple[float | None, float | None, float | None]:
    """Return highpass, lowpass, and notch frequencies parsed from an EDF prefilter field."""
    highpass = lowpass = notch = None
    for kind, value in re.findall(r"(HP|LP|NOTCH)\s*:?\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE):
        number = float(value)
        token = kind.upper()
        if token == "HP":
            highpass = number
        elif token == "LP":
            lowpass = number
        else:
            notch = number
    bandpass = re.search(
        r"BP\s*:?\s*([0-9]*\.?[0-9]+)\s*HZ\s*-\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if bandpass:
        highpass = float(bandpass.group(1))
        lowpass = float(bandpass.group(2))
    return highpass, lowpass, notch


def eeg_channel_action(channel: ChannelAssignment) -> str:
    """Return filter, retain, or exclude for one EEG channel."""
    highpass, lowpass, _notch = acquisition_band(channel.prefiltering)
    if highpass is not None and highpass >= 5.0:
        return "exclude"
    if highpass is not None and lowpass is not None and highpass <= 0.5 and 20.0 <= lowpass <= 40.0:
        return "retain"
    return "filter"


def emg_channel_action(channel: ChannelAssignment, sfreq_hz: float) -> str:
    """Return filter or retain for one chin-EMG channel."""
    highpass, lowpass, _notch = acquisition_band(channel.prefiltering)
    if highpass is None or highpass < EMG_L_FREQ - 0.5:
        return "filter"
    target = emg_h_freq(sfreq_hz)
    if target is None:
        return "retain"
    if lowpass is None:
        return "filter"
    if abs(lowpass - target) <= max(5.0, 0.2 * target):
        return "retain"
    return "filter"


def filter_modality(
    raw: mne.io.BaseRaw,
    role: str,
    channels: list[ChannelAssignment] | None = None,
) -> tuple[mne.io.BaseRaw, AppliedFilter]:
    """Return a filtered copy. ``raw`` is not modified."""
    sfreq = float(raw.info["sfreq"])
    if role == "eeg":
        return _filter_selected(raw, role, channels, eeg_channel_action)
    if role == "emg_chin":
        return _filter_emg(raw, channels)
    processed = raw.copy()
    return processed, AppliedFilter(
        role=role,
        sfreq_hz=sfreq,
        l_freq_hz=None,
        h_freq_hz=None,
        notch_hz=None,
        phase="none",
        method="none",
        line_noise_ratio=None,
    )


def make_fixed_epochs(
    raw: mne.io.BaseRaw,
    onsets_sec: np.ndarray,
    codes: np.ndarray,
    epoch_sec: float,
    event_id: dict[str, int],
) -> mne.Epochs:
    sfreq = float(raw.info["sfreq"])
    samples = np.round(np.asarray(onsets_sec, dtype=float) * sfreq).astype(int)
    events = np.column_stack(
        [samples, np.zeros(len(samples), dtype=int), np.asarray(codes, dtype=int)]
    )
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_sec - 1.0 / sfreq,
        baseline=None,
        preload=True,
        reject=None,
        proj=False,
        verbose="ERROR",
    )


def assess_channel(
    data_volts: np.ndarray,
    assignment: ChannelAssignment,
) -> ChannelQuality:
    microvolts = np.asarray(data_volts, dtype=float) * 1e6
    std_uv = float(np.std(microvolts))
    notes: list[str] = []
    if assignment.assumed:
        notes.append("chin_label_assumed")
    if assignment.ambiguous_label:
        notes.append("ambiguous_channel_label")
    if assignment.original_unit.strip().lower().replace("µ", "u") not in {"uv", "mv", "v"}:
        notes.append("physical_unit_not_recognized")
    flat = std_uv < FLAT_STD_UV
    if flat:
        notes.append("flat")
    fraction = _saturation_fraction(data_volts, assignment)
    reversed_digits = (
        assignment.digital_min is not None
        and assignment.digital_max is not None
        and assignment.digital_min >= assignment.digital_max
    )
    if reversed_digits:
        notes.append("digital_range_reversed")
    outside = _outside_fraction(data_volts, assignment)
    if outside > SATURATION_FRACTION:
        notes.append(f"exceeds_header_physical_range:{outside:.1%}")
    # A reversed or non-bounding header range is not evidence of amplifier saturation.
    range_is_usable = not reversed_digits and (outside != outside or outside <= SATURATION_FRACTION)
    saturated = range_is_usable and fraction > SATURATION_FRACTION
    if saturated:
        notes.append("saturated")
    if assignment.role == "eeg" and np.percentile(np.abs(microvolts), 99) > EEG_HIGH_AMPLITUDE_UV:
        notes.append("high_eeg_amplitude")
    return ChannelQuality(
        canonical_name=assignment.canonical_name,
        role=assignment.role,
        std_uv=std_uv,
        saturation_fraction=fraction,
        flat=flat,
        saturated=saturated,
        notes=tuple(notes),
    )


def plot_diagnostics(
    result: PreprocessedRecording,
    out_path: str | Path,
    segment_sec: float = 10.0,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eeg = result.modality("eeg")
    emg = result.modality("emg_chin")
    onset, stage = _example_onset(result)
    rows = 2 if emg is not None else 1
    figure, axes = plt.subplots(rows, 2, figsize=(11, 3.2 * rows), sharex=True, squeeze=False)
    if eeg is not None:
        _plot_pair(axes[0], eeg, onset, segment_sec, f"EEG during {stage}")
    if emg is not None:
        _plot_pair(axes[1], emg, onset, segment_sec, f"Chin EMG during {stage}")
    figure.suptitle(
        f"{result.recording_id} ({result.group})  onset {onset:.1f} s  stage {stage}"
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path


def _load_modalities(
    path: Path,
    header: EdfHeader,
) -> tuple[list[tuple[str, float, mne.io.BaseRaw, list[ChannelAssignment]]], list[ChannelAssignment]]:
    assignments = _assign_header_channels(header)
    wanted = [item for item in assignments if item.role in MNE_TYPES]
    ignored = [item for item in assignments if item.role not in MNE_TYPES]
    grouped: dict[float, list[ChannelAssignment]] = {}
    for item in wanted:
        grouped.setdefault(item.native_sfreq_hz, []).append(item)

    loaded = []
    for sfreq, channels in grouped.items():
        include = list(dict.fromkeys(item.edf_label for item in channels))
        raw = mne.io.read_raw_edf(path, include=include, preload=True, verbose="ERROR")
        if abs(float(raw.info["sfreq"]) - sfreq) > 1e-3:
            raise RuntimeError(
                f"{path.name} sampling rate changed from {sfreq} Hz to {raw.info['sfreq']} Hz"
            )
        if len(raw.ch_names) != len(channels):
            raise RuntimeError(
                f"{path.name} channel count mismatch: header {len(channels)} vs MNE {raw.ch_names}"
            )
        paired = _pair_loaded_names(channels, raw.ch_names)
        rename = {item.mne_name: item.canonical_name for item in paired}
        types = {item.mne_name: MNE_TYPES[item.role] for item in paired}
        raw.set_channel_types(types, verbose="ERROR")
        raw.rename_channels(rename)
        for role in ("eeg", "eog", "emg_chin"):
            role_channels = [item for item in paired if item.role == role]
            if not role_channels:
                continue
            role_raw = raw.copy().pick([item.canonical_name for item in role_channels])
            loaded.append((role, sfreq, role_raw, role_channels))
        del raw
    return loaded, ignored


def _infer_unlabeled_chin(
    path: Path,
    header: EdfHeader,
    roles: list[ChannelRole],
) -> int | None:
    """Pick a chin channel from the signal when no label says chin.

    Broadband muscle stays. A spiky heart trace and a narrow snore tone do not,
    and leg channels are quiet except in bursts, so the chin is the broadband
    channel with the higher ordinary level.
    """
    if not path.is_file():
        return None
    candidates = [
        index
        for index, (signal, role) in enumerate(zip(header.signals, roles))
        if role.role == "misc" and signal.sampling_frequency_hz >= 100 and not _OXYGEN.search(signal.label)
    ]
    if not candidates:
        return None
    labels = [header.signals[index].label for index in candidates]
    raw = mne.io.read_raw_edf(path, include=labels, preload=False, verbose="ERROR")
    ordered = []
    for label in labels:
        match = next((name for name in raw.ch_names if name == label or name.startswith(label)), None)
        if match is None:
            return None
        ordered.append(match)
    sfreq = float(raw.info["sfreq"])
    signals = []
    for name in ordered:
        signals.append(_spread_windows(raw, name, sfreq))
    chosen = _choose_chin_index(signals, sfreq)
    if chosen is None:
        return None
    return candidates[chosen]


def _spread_windows(raw: mne.io.BaseRaw, name: str, sfreq: float) -> np.ndarray:
    window = int(8 * sfreq)
    if raw.n_times <= window:
        return raw.get_data(picks=[name])[0]
    count = min(24, raw.n_times // window)
    span = raw.n_times - window
    starts = np.linspace(0, span, count).astype(int)
    pieces = [raw.get_data(picks=[name], start=int(start), stop=int(start) + window)[0] for start in starts]
    return np.concatenate(pieces)


def _choose_chin_index(signals: list[np.ndarray], sfreq: float) -> int | None:
    scored = [(index, _muscle_score(samples, sfreq)) for index, samples in enumerate(signals)]
    muscle = [
        (index, level)
        for index, (entropy, kurtosis, level) in scored
        if entropy >= 0.92 and kurtosis <= 8.0 and level > 0
    ]
    if not muscle:
        return None
    return max(muscle, key=lambda item: item[1])[0]


def _muscle_score(samples: np.ndarray, sfreq: float) -> tuple[float, float, float]:
    """Return entropy from 15–70 Hz, kurtosis, and the ordinary RMS level."""
    window = max(32, int(8 * sfreq))
    if samples.size < window:
        pieces = [np.asarray(samples, dtype=float)]
    else:
        count = min(24, samples.size // window)
        pieces = [samples[index * window : (index + 1) * window] for index in range(count)]
    entropies = []
    kurtoses = []
    levels = []
    for piece in pieces:
        deviation = float(np.std(piece))
        if deviation <= 0:
            continue
        entropies.append(_band_entropy(piece, sfreq))
        centered = (piece - np.mean(piece)) / deviation
        kurtoses.append(float(np.mean(np.power(centered, 4))))
        levels.append(float(np.sqrt(np.mean(np.square(piece)))))
    if not entropies:
        return 0.0, 0.0, 0.0
    return float(np.median(entropies)), float(np.median(kurtoses)), float(np.percentile(levels, 40))


def _band_entropy(samples: np.ndarray, sfreq: float) -> float:
    nperseg = min(samples.size, max(32, int(2 * sfreq)))
    freqs, power = welch(samples, fs=sfreq, nperseg=nperseg)
    band = power[(freqs >= 15.0) & (freqs < min(70.0, sfreq / 2.0 - 1.0))]
    total = float(np.sum(band))
    if band.size < 4 or total <= 0:
        return 0.0
    share = band / total
    return float(-np.sum(share * np.log(share + 1e-20)) / np.log(share.size))


def _assign_header_channels(header: EdfHeader) -> list[ChannelAssignment]:
    labels = [signal.label for signal in header.signals]
    roles = [classify_label(label) for label in labels]
    if not any(role.role == "emg_chin" for role in roles):
        chosen = _infer_unlabeled_chin(Path(header.path), header, roles)
        if chosen is not None:
            roles[chosen] = ChannelRole("emg_chin", assumed=True)
    chin_count = sum(1 for role in roles if role.role == "emg_chin")
    chin_seen = 0
    assigned: list[ChannelAssignment] = []
    for signal, role in zip(header.signals, roles):
        canonical = _canonical_name(signal.label, role, chin_seen, chin_count)
        if role.role == "emg_chin":
            chin_seen += 1
        assigned.append(
            ChannelAssignment(
                edf_label=signal.label,
                mne_name=signal.label,
                canonical_name=canonical,
                role=role.role,
                native_sfreq_hz=signal.sampling_frequency_hz,
                original_unit=signal.physical_dimension,
                physical_min_volts=signal.physical_min_volts,
                physical_max_volts=signal.physical_max_volts,
                assumed=role.assumed,
                ambiguous_label=role.ambiguous_label,
                digital_min=signal.digital_min,
                digital_max=signal.digital_max,
                prefiltering=signal.prefiltering,
            )
        )
    return assigned


def _canonical_name(label: str, role: ChannelRole, chin_seen: int, chin_count: int) -> str:
    if role.role == "eeg":
        return f"EEG {label}"
    if role.role == "eog":
        return f"EOG {label}"
    if role.role == "emg_chin":
        if chin_count > 1:
            return f"EMG chin-{chin_seen + 1}"
        return "EMG chin"
    if role.role == "ecg":
        return f"ECG {label}"
    if role.role == "emg_limb":
        return f"EMG limb {label}"
    return label


def _pair_loaded_names(
    channels: list[ChannelAssignment],
    mne_names: list[str],
) -> list[ChannelAssignment]:
    paired: list[ChannelAssignment] = []
    for channel, mne_name in zip(channels, mne_names):
        if mne_name != channel.edf_label and not mne_name.startswith(channel.edf_label):
            raise RuntimeError(
                f"Could not pair EDF label {channel.edf_label!r} with loaded name {mne_name!r}"
            )
        paired.append(
            ChannelAssignment(
                edf_label=channel.edf_label,
                mne_name=mne_name,
                canonical_name=channel.canonical_name,
                role=channel.role,
                native_sfreq_hz=channel.native_sfreq_hz,
                original_unit=channel.original_unit,
                physical_min_volts=channel.physical_min_volts,
                physical_max_volts=channel.physical_max_volts,
                assumed=channel.assumed,
                ambiguous_label=channel.ambiguous_label,
                digital_min=channel.digital_min,
                digital_max=channel.digital_max,
                prefiltering=channel.prefiltering,
            )
        )
    return paired


def _process_modality(
    item: tuple[str, float, mne.io.BaseRaw, list[ChannelAssignment]],
) -> PreprocessedModality:
    role, sfreq, raw, channels = item
    processed, applied = filter_modality(raw, role, channels)
    quality = []
    data = raw.get_data()
    actions = _channel_actions(role, channels, sfreq)
    for index, channel in enumerate(channels):
        checked = assess_channel(data[index], channel)
        extra = _action_note(actions[index])
        if extra:
            checked = replace(checked, notes=checked.notes + (extra,))
        quality.append(checked)
    analysis = [
        channel.canonical_name
        for channel, action in zip(channels, actions)
        if action != "exclude"
    ]
    return PreprocessedModality(
        role=role,
        sfreq_hz=sfreq,
        original=raw,
        processed=processed,
        channels=channels,
        quality=quality,
        applied_filter=applied,
        analysis_channels=analysis,
    )


def _channel_actions(
    role: str,
    channels: list[ChannelAssignment],
    sfreq_hz: float,
) -> list[str]:
    if role == "eeg":
        actions = [eeg_channel_action(channel) for channel in channels]
        if sfreq_hz / 2.0 <= EEG_H_FREQ:
            return ["exclude"] * len(actions)
        return actions
    if role == "emg_chin":
        return [emg_channel_action(channel, sfreq_hz) for channel in channels]
    return ["unfiltered"] * len(channels)


def _action_note(action: str) -> str | None:
    if action == "exclude":
        return "acquisition_highpass_unsuitable_for_sleep_eeg"
    if action == "retain":
        return "acquisition_filter_retained"
    return None


def _filter_selected(raw, role: str, channels: list[ChannelAssignment] | None, decide) -> tuple[mne.io.BaseRaw, AppliedFilter]:
    sfreq = float(raw.info["sfreq"])
    if not channels:
        picks = list(raw.ch_names)
        retained: list[ChannelAssignment] = []
    else:
        actions = [decide(channel) for channel in channels]
        picks = [
            channel.canonical_name
            for channel, action in zip(channels, actions)
            if action == "filter"
        ]
        retained = [
            channel for channel, action in zip(channels, actions) if action == "retain"
        ]
    processed = raw.copy()
    if picks and sfreq / 2.0 <= EEG_H_FREQ:
        picks = []
    if picks:
        processed.filter(
            l_freq=EEG_L_FREQ,
            h_freq=EEG_H_FREQ,
            picks=picks,
            phase="zero",
            fir_design="firwin",
            fir_window="hamming",
            l_trans_bandwidth=EEG_L_TRANS_HZ,
            h_trans_bandwidth=EEG_H_TRANS_HZ,
            n_jobs=1,
            verbose="ERROR",
        )
    if picks and retained:
        method = "fir+acquisition"
        phase = "zero"
        low, high, notch = EEG_L_FREQ, EEG_H_FREQ, None
    elif picks:
        method = "fir"
        phase = "zero"
        low, high, notch = EEG_L_FREQ, EEG_H_FREQ, None
    else:
        method = "acquisition"
        phase = "acquisition"
        low, high, notch = acquisition_band(retained[0].prefiltering) if retained else (None, None, None)
    return processed, AppliedFilter(
        role=role,
        sfreq_hz=sfreq,
        l_freq_hz=low,
        h_freq_hz=high,
        notch_hz=notch,
        phase=phase,
        method=method,
        line_noise_ratio=None,
        channels_filtered=tuple(picks),
        channels_retained=tuple(channel.canonical_name for channel in retained),
    )


def _filter_emg(
    raw: mne.io.BaseRaw,
    channels: list[ChannelAssignment] | None = None,
) -> tuple[mne.io.BaseRaw, AppliedFilter]:
    sfreq = float(raw.info["sfreq"])
    if channels:
        actions = [emg_channel_action(channel, sfreq) for channel in channels]
        picks = [
            channel.canonical_name
            for channel, action in zip(channels, actions)
            if action == "filter"
        ]
        retained = [channel for channel, action in zip(channels, actions) if action == "retain"]
    else:
        picks = list(raw.ch_names)
        retained = []
    if not picks:
        highpass, lowpass, notch = acquisition_band(retained[0].prefiltering)
        return raw.copy(), AppliedFilter(
            role="emg_chin",
            sfreq_hz=sfreq,
            l_freq_hz=highpass,
            h_freq_hz=lowpass,
            notch_hz=notch,
            phase="acquisition",
            method="acquisition",
            line_noise_ratio=None,
            channels_retained=tuple(channel.canonical_name for channel in retained),
        )
    h_freq = emg_h_freq(sfreq)
    ratio = _line_noise_ratio(raw.copy().pick(picks))
    nyquist = sfreq / 2.0
    use_notch = (
        ratio >= LINE_NOISE_RATIO
        and nyquist > LINE_HZ + NOTCH_NYQUIST_MARGIN_HZ
        and (h_freq is None or h_freq > LINE_HZ)
    )
    processed = raw.copy()
    notch = None
    if use_notch:
        processed.notch_filter(
            freqs=[LINE_HZ],
            picks=picks,
            phase="zero",
            method="fir",
            fir_design="firwin",
            n_jobs=1,
            verbose="ERROR",
        )
        notch = LINE_HZ
    filter_kwargs = {
        "l_freq": EMG_L_FREQ,
        "h_freq": h_freq,
        "picks": picks,
        "phase": "zero",
        "fir_design": "firwin",
        "fir_window": "hamming",
        "l_trans_bandwidth": EMG_L_TRANS_HZ,
        "n_jobs": 1,
        "verbose": "ERROR",
    }
    if h_freq is not None:
        filter_kwargs["h_trans_bandwidth"] = EMG_H_TRANS_HZ
    processed.filter(**filter_kwargs)
    return processed, AppliedFilter(
        role="emg_chin",
        sfreq_hz=sfreq,
        l_freq_hz=EMG_L_FREQ,
        h_freq_hz=h_freq,
        notch_hz=notch,
        phase="zero",
        method="fir" if not retained else "fir+acquisition",
        line_noise_ratio=ratio,
        channels_filtered=tuple(picks),
        channels_retained=tuple(channel.canonical_name for channel in retained),
    )


def _line_noise_ratio(raw: mne.io.BaseRaw, line_hz: float = LINE_HZ) -> float:
    sfreq = float(raw.info["sfreq"])
    snippet = _middle(raw, seconds=60.0)
    nperseg = min(snippet.size, int(sfreq * 4))
    if nperseg < 32:
        return 1.0
    freqs, power = welch(snippet, fs=sfreq, nperseg=nperseg)
    index = int(np.argmin(np.abs(freqs - line_hz)))
    if abs(freqs[index] - line_hz) > 1.0:
        return 1.0
    neighbors = (freqs >= line_hz - 10) & (freqs <= line_hz + 10) & (np.abs(freqs - line_hz) >= 2)
    if not np.any(neighbors) or power[index] <= 0:
        return 1.0
    baseline = float(np.median(power[neighbors]))
    if baseline <= 0:
        return float("inf")
    return float(power[index] / baseline)


def _middle(raw: mne.io.BaseRaw, seconds: float) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    n_take = min(raw.n_times, int(seconds * sfreq))
    start = max(0, (raw.n_times - n_take) // 2)
    return raw.get_data(picks=[raw.ch_names[0]])[0, start : start + n_take]


def _calendar_dates_disagree(header: EdfHeader, annotation_date: str | None) -> bool:
    if not annotation_date:
        return False
    parts = re.split(r"[./-]", annotation_date.strip())
    if len(parts) != 3:
        return True
    day, month, year = (int(part) for part in parts)
    if year < 100:
        year += 2000 if year < 85 else 1900
    recorded = header.start_datetime_naive
    return (day, month, year) != (recorded.day, recorded.month, recorded.year)


def _duration_sec(modalities: list[PreprocessedModality], header: EdfHeader) -> float:
    if modalities:
        raw = modalities[0].original
        return raw.n_times / float(raw.info["sfreq"])
    return header.duration_sec


def _select_stage_events(
    scoring: ScoringFile,
    duration_sec: float,
) -> tuple[list[SleepEvent], list[tuple[str, str, str]]]:
    kept: list[SleepEvent] = []
    skipped: list[tuple[str, str, str]] = []
    for event in scoring.events:
        reason = _skip_reason(event, duration_sec)
        if reason is None:
            kept.append(event)
        else:
            skipped.append((event.clock_time, event.stage, reason))
    return kept, skipped


def _skip_reason(event: SleepEvent, duration_sec: float) -> str | None:
    if event.stage in {"AMBIGUOUS", "UNSCORED"} or "ambiguous_stage" in event.issues:
        return "ambiguous_or_unscored"
    if "overlap" in event.issues:
        return "overlap"
    if abs(event.duration_sec - EPOCH_SEC) > 1e-3:
        return "duration_not_30s"
    if not np.isfinite(event.onset_sec):
        return "bad_onset"
    if event.onset_sec < -1e-6:
        return "starts_before_recording"
    if event.onset_sec + EPOCH_SEC > duration_sec + 1e-3:
        return "extends_past_recording"
    return None


def _stage_epochs(
    modalities: list[PreprocessedModality],
    kept: list[SleepEvent],
) -> list[mne.Epochs]:
    if not kept:
        return []
    onsets = np.array([event.onset_sec for event in kept], dtype=float)
    codes = np.array([STAGE_CODES[event.stage] for event in kept], dtype=int)
    event_id = {stage: code for stage, code in STAGE_CODES.items() if code in set(codes)}
    epochs = []
    for modality in modalities:
        if modality.role != "eeg":
            continue
        source = modality.processed
        if modality.analysis_channels and len(modality.analysis_channels) != len(source.ch_names):
            source = source.copy().pick(modality.analysis_channels)
        epochs.append(make_fixed_epochs(source, onsets, codes, EPOCH_SEC, event_id))
    return epochs


def _rem_emg_epochs(
    modalities: list[PreprocessedModality],
    scoring: ScoringFile,
    duration_sec: float,
) -> mne.Epochs | None:
    emg = next((item for item in modalities if item.role == "emg_chin"), None)
    if emg is None:
        return None
    onsets = _rem_window_onsets(scoring.rem_intervals(), duration_sec)
    if onsets.size == 0:
        return None
    codes = np.ones(onsets.size, dtype=int)
    return make_fixed_epochs(
        emg.processed,
        onsets,
        codes,
        REM_EMG_WINDOW_SEC,
        {"REM_EMG": 1},
    )


def _rem_window_onsets(intervals: list[tuple[float, float]], duration_sec: float) -> np.ndarray:
    onsets: list[float] = []
    for start, end in intervals:
        cursor = start
        while cursor + REM_EMG_WINDOW_SEC <= min(end, duration_sec) + 1e-6:
            if cursor >= -1e-6:
                onsets.append(max(cursor, 0.0))
            cursor += REM_EMG_WINDOW_SEC
    return np.asarray(onsets, dtype=float)


def _outside_fraction(data_volts: np.ndarray, assignment: ChannelAssignment) -> float:
    low = assignment.physical_min_volts
    high = assignment.physical_max_volts
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float("nan")
    return float(np.mean((data_volts < low) | (data_volts > high)))


def _saturation_fraction(data_volts: np.ndarray, assignment: ChannelAssignment) -> float:
    low = assignment.physical_min_volts
    high = assignment.physical_max_volts
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high):
        return float("nan")
    span = high - low
    if span <= 0:
        return float("nan")
    tolerance = span * 0.001
    at_rail = (data_volts <= low + tolerance) | (data_volts >= high - tolerance)
    return float(np.mean(at_rail))


def _example_onset(result: PreprocessedRecording) -> tuple[float, str]:
    for event in result.stage_events_kept:
        if event.stage == "R":
            return event.onset_sec, "R"
    if result.stage_events_kept:
        event = result.stage_events_kept[0]
        return event.onset_sec, event.stage
    return 0.0, "unscored"


def _plot_pair(axes, modality: PreprocessedModality, onset: float, segment_sec: float, title: str) -> None:
    index = _preferred_channel(modality)
    name = modality.original.ch_names[index]
    for axis, raw, label in (
        (axes[0], modality.original, "original"),
        (axes[1], modality.processed, "processed"),
    ):
        times, values = _snippet(raw, index, onset, segment_sec)
        axis.plot(times, values, color="#1f4e79", linewidth=0.6)
        axis.set_ylabel("µV")
        axis.set_title(f"{title}: {name} ({label})")
        axis.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time from recording start (s)")


def _preferred_channel(modality: PreprocessedModality) -> int:
    allowed = set(modality.analysis_channels) or set(modality.original.ch_names)
    if modality.role == "eeg":
        for token in ("C4-A", "C3-A", "C4-P", "C3-P", "C4", "C3"):
            for index, name in enumerate(modality.original.ch_names):
                if token in name and name in allowed:
                    return index
    for index, name in enumerate(modality.original.ch_names):
        if name in allowed:
            return index
    return 0


def _snippet(
    raw: mne.io.BaseRaw,
    index: int,
    onset: float,
    segment_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    sfreq = float(raw.info["sfreq"])
    start = min(int(round(onset * sfreq)), raw.n_times)
    stop = min(start + int(round(segment_sec * sfreq)), raw.n_times)
    values = raw.get_data(picks=[raw.ch_names[index]])[0, start:stop] * 1e6
    times = (np.arange(start, stop)) / sfreq
    return times, values
