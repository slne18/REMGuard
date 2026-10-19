from datetime import datetime, time
from pathlib import Path

import mne
import numpy as np
import pytest

from remguard.data.annotations import parse_scoring_text
from remguard.data.edf_header import EdfHeader, EdfSignalHeader, read_edf_header
from remguard.preprocessing import (
    ChannelAssignment,
    _assign_header_channels,
    _choose_chin_index,
    acquisition_band,
    assess_channel,
    classify_label,
    eeg_channel_action,
    emg_channel_action,
    emg_h_freq,
    filter_modality,
    find_gaps,
    make_fixed_epochs,
    preprocess_recording,
)

DATA = Path("data/raw")


def _sine(sfreq: float, seconds: float, freq: float, amplitude_uv: float) -> np.ndarray:
    times = np.arange(0, seconds, 1.0 / sfreq)
    return (amplitude_uv * 1e-6) * np.sin(2 * np.pi * freq * times)


def _raw(name: str, kind: str, samples: np.ndarray, sfreq: float) -> mne.io.RawArray:
    info = mne.create_info([name], sfreq, ch_types=[kind])
    return mne.io.RawArray(samples[np.newaxis, :], info, verbose="ERROR")


def test_channel_roles_cover_the_proof_of_concept_names():
    expected = {
        "LOC / A2": "eog",
        "ROC / A1": "eog",
        "CHIN-0": "emg_chin",
        "CHIN-1": "emg_chin",
        "C3-A2": "eeg",
        "EKG": "ecg",
        "EMG-EMG": "emg_chin",
        "ROC-LOC": "eog",
        "EMG1-EMG2": "emg_chin",
        "DX1-DX2": "emg_limb",
        "Fp2-F4": "eeg",
        "P4-O2": "eeg",
        "F3A2": "eeg",
        "C4A1": "eeg",
        "O1A2": "eeg",
        "O2A1": "eeg",
        "SAO2": "misc",
        "SpO2": "misc",
        "THE": "misc",
        "milo": "emg_chin",
        "EMG": "emg_chin",
    }
    for label, role in expected.items():
        assert classify_label(label).role == role
    assert classify_label("milo").ambiguous_label
    assert classify_label("EMG").assumed
    assert classify_label("EMG1-EMG2").assumed is False


def _header(labels: list[str]) -> EdfHeader:
    signals = [
        EdfSignalHeader(
            index=index,
            label=label,
            physical_dimension="uV",
            physical_min=-500,
            physical_max=500,
            digital_min=-32768,
            digital_max=32767,
            prefiltering="",
            samples_per_record=200,
            sampling_frequency_hz=200,
        )
        for index, label in enumerate(labels)
    ]
    return EdfHeader(
        path="night.edf",
        start_date="01.01.00",
        start_time=time(0, 0, 0),
        start_datetime_naive=datetime(2000, 1, 1),
        n_records=1,
        record_duration_sec=1.0,
        duration_sec=1.0,
        reserved="",
        signals=tuple(signals),
    )


def test_unlabeled_chin_is_the_broadband_channel_with_the_higher_floor():
    sfreq = 200.0
    windows = 24
    samples = int(8 * sfreq)
    rng = np.random.default_rng(0)
    chin = rng.normal(0, 1.0, size=windows * samples)
    leg = rng.normal(0, 0.05, size=windows * samples)
    leg[: 2 * samples] = rng.normal(0, 8.0, size=2 * samples)
    tone = np.sin(2 * np.pi * 40 * np.arange(windows * samples) / sfreq)
    heart = np.zeros(windows * samples)
    heart[:: int(sfreq)] = 20.0
    chosen = _choose_chin_index([tone, heart, leg, chin], sfreq)
    assert chosen == 3
    assert _choose_chin_index([tone], sfreq) is None


def test_a_named_chin_is_kept_and_a_bare_x_label_is_not_renamed():
    named = {item.edf_label: item.role for item in _assign_header_channels(_header(["CHIN", "X1", "C3-A2"]))}
    assert named["CHIN"] == "emg_chin"
    assert named["X1"] == "misc"


ISRUC = Path("data/test_REMGuard/healthy/subgroupIII_1/1.edf")


@pytest.mark.skipif(not ISRUC.exists(), reason="ISRUC sample night is not on this machine")
def test_downloaded_isruc_night_assigns_x1_to_chin():
    roles = {item.edf_label: item.role for item in _assign_header_channels(read_edf_header(ISRUC))}
    assert roles["X1"] == "emg_chin"
    assert roles["F3-A2"] == "eeg"
    assert roles["LOC-A2"] == "eog"


def _assignment(label: str, prefiltering: str, sfreq: float = 512.0) -> ChannelAssignment:
    role = classify_label(label)
    return ChannelAssignment(
        edf_label=label,
        mne_name=label,
        canonical_name=label,
        role=role.role,
        native_sfreq_hz=sfreq,
        original_unit="uV",
        physical_min_volts=-100e-6,
        physical_max_volts=100e-6,
        assumed=False,
        ambiguous_label=False,
        prefiltering=prefiltering,
    )


def test_acquisition_filter_is_kept_when_it_already_matches():
    assert acquisition_band("LP:30.00Hz HP:0.30Hz NOTCH:50") == (0.3, 30.0, 50.0)
    assert acquisition_band("BP: 0.1HZ -100HZ") == (0.1, 100.0, None)
    central = _assignment("C4-A1", "LP:30.00Hz HP:0.30Hz NOTCH:50")
    temporal = _assignment("F8-T4", "LP:100.00Hz HP:10.00Hz NOTCH:50")
    chin = _assignment("EMG1-EMG2", "LP:100.00Hz HP:10.00Hz NOTCH:50")
    wide = _assignment("C3-A2", "BP: 0.1HZ -100HZ", sfreq=100)
    assert eeg_channel_action(central) == "retain"
    assert eeg_channel_action(temporal) == "exclude"
    assert eeg_channel_action(wide) == "filter"
    assert emg_channel_action(chin, 512) == "retain"
    assert emg_channel_action(wide, 100) == "filter"


def test_emg_lowpass_follows_the_native_sampling_rate():
    assert emg_h_freq(100) == pytest.approx(40.0)
    assert emg_h_freq(128) == pytest.approx(54.0)
    assert emg_h_freq(512) == pytest.approx(100.0)


def test_eeg_and_emg_filters_differ_and_keep_in_band_amplitude():
    sfreq = 100.0
    eeg = _raw(
        "EEG C3-A2",
        "eeg",
        _sine(sfreq, 120, 10, 20) + _sine(sfreq, 120, 0.05, 100),
        sfreq,
    )
    emg = _raw("EMG chin", "emg", _sine(sfreq, 120, 25, 50) + _sine(sfreq, 120, 1, 80), sfreq)
    eeg_before = eeg.get_data().copy()
    emg_before = emg.get_data().copy()

    eeg_filtered, eeg_filter = filter_modality(eeg, "eeg")
    emg_filtered, emg_filter = filter_modality(emg, "emg_chin")

    assert np.allclose(eeg.get_data(), eeg_before)
    assert np.allclose(emg.get_data(), emg_before)
    assert eeg_filter.l_freq_hz == pytest.approx(0.3)
    assert eeg_filter.h_freq_hz == pytest.approx(35)
    assert emg_filter.l_freq_hz == pytest.approx(10)
    assert emg_filter.h_freq_hz == pytest.approx(40)
    assert eeg_filter.phase == "zero"
    assert emg_filter.notch_hz is None

    trim = int(20 * sfreq)
    original_tone = _sine(sfreq, 120, 25, 50)[trim:-trim]
    filtered_tone = emg_filtered.get_data()[0, trim:-trim]
    scale = np.dot(filtered_tone, original_tone) / np.dot(original_tone, original_tone)
    assert scale == pytest.approx(1.0, abs=0.2)

    def _bin_power(values: np.ndarray, freq: float) -> float:
        spectrum = np.fft.rfft(values * np.hanning(values.size))
        freqs = np.fft.rfftfreq(values.size, 1.0 / sfreq)
        return float(np.abs(spectrum[np.argmin(np.abs(freqs - freq))]) ** 2)

    slow = eeg.get_data()[0, trim:-trim]
    slow_filtered = eeg_filtered.get_data()[0, trim:-trim]
    assert _bin_power(slow_filtered, 0.05) < 0.05 * _bin_power(slow, 0.05)


def test_missing_emg_does_not_crash_and_flat_channel_is_flagged():
    info = mne.create_info(["EEG only"], 100, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 1000)), info, verbose="ERROR")
    processed, applied = filter_modality(raw, "eeg")
    assert processed.n_times == raw.n_times
    assert applied.role == "eeg"

    assignment = classify_label("C3-A2")
    from remguard.preprocessing import ChannelAssignment

    channel = ChannelAssignment(
        edf_label="C3-A2",
        mne_name="C3-A2",
        canonical_name="EEG C3-A2",
        role=assignment.role,
        native_sfreq_hz=100,
        original_unit="uV",
        physical_min_volts=-500e-6,
        physical_max_volts=500e-6,
        assumed=False,
        ambiguous_label=False,
    )
    quality = assess_channel(np.zeros(1000), channel)
    assert quality.flat
    rails = np.full(1000, 500e-6)
    saturated = assess_channel(rails, channel)
    assert saturated.saturated
    reversed_header = ChannelAssignment(
        edf_label="C3-A2",
        mne_name="C3-A2",
        canonical_name="EEG C3-A2",
        role="eeg",
        native_sfreq_hz=100,
        original_unit="uV",
        physical_min_volts=-50e-6,
        physical_max_volts=50e-6,
        assumed=False,
        ambiguous_label=False,
        digital_min=410,
        digital_max=-410,
    )
    outside = assess_channel(np.full(1000, 200e-6), reversed_header)
    assert outside.saturated is False
    assert "digital_range_reversed" in outside.notes


def test_epochs_follow_annotation_onsets_without_baseline_subtraction():
    sfreq = 100.0
    samples = np.arange(0, 100 * sfreq) / sfreq
    raw = _raw("EEG C3-A2", "eeg", samples, sfreq)
    epochs = make_fixed_epochs(
        raw,
        onsets_sec=np.array([10.0]),
        codes=np.array([1]),
        epoch_sec=30.0,
        event_id={"W": 1},
    )
    expected = raw.get_data()[0, 1000:4000]
    assert epochs.get_data()[0, 0].shape == expected.shape
    assert np.allclose(epochs.get_data()[0, 0], expected)


def test_gaps_include_leading_unscored_time():
    text = """
Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\t22:38:39\tSLEEP-S0\t30\tROC-A2
"""
    scoring = parse_scoring_text(text, time(22, 33, 9))
    gaps = find_gaps(scoring.events, duration_sec=1000)
    assert gaps[0]["kind"] == "leading"
    assert gaps[0]["end_sec"] == pytest.approx(330)


@pytest.mark.skipif(not (DATA / "n12.edf").exists(), reason="n12 recording is not downloaded")
def test_n12_header_has_no_embedded_scores_and_duplicate_chin():
    header = read_edf_header(DATA / "n12.edf")
    assert header.has_embedded_annotations is False
    labels = [signal.label for signal in header.signals]
    assert labels.count("CHIN") == 2
    assert header.start_time == time(15, 14, 22)
    assert header.duration_sec == pytest.approx(29700)


@pytest.mark.skipif(
    not (DATA / "n12.edf").exists() or not (DATA / "n12.txt").exists(),
    reason="n12 recording is not downloaded",
)
def test_n12_pipeline_aligns_epochs_and_keeps_native_rate():
    result = preprocess_recording(DATA / "n12.edf", DATA / "n12.txt")
    assert "missing_eeg" not in result.issues
    assert "missing_emg_chin" not in result.issues
    assert result.modality("eeg").sfreq_hz == pytest.approx(100)
    assert result.modality("emg_chin").sfreq_hz == pytest.approx(100)
    assert [channel.canonical_name for channel in result.modality("emg_chin").channels] == [
        "EMG chin-1",
        "EMG chin-2",
    ]
    assert result.stage_epochs
    assert result.stage_events_kept[0].onset_sec == pytest.approx(0)
    assert any(event.stage == "R" for event in result.stage_events_kept)
    assert result.rem_emg_epochs is not None
    assert any(event.stage_column == "1S3" for event in result.scoring.events)
    assert all(event.stage_column != "1S3" for event in result.stage_events_kept)
    assert any(reason == "ambiguous_or_unscored" for _, _, reason in result.stage_events_skipped)
    eeg_filter = result.modality("eeg").applied_filter
    emg_filter = result.modality("emg_chin").applied_filter
    assert eeg_filter.l_freq_hz != emg_filter.l_freq_hz
    assert "annotation_calendar_date_disagrees_with_edf" in result.issues
