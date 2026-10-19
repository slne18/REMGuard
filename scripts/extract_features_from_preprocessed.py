"""Build REM-detector features from the compact preprocessed nights.

EEG and EOG are already filtered. EOG is not filtered again. Chin EMG is not
a feature. Labels come from the expert score file saved beside each night.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mne
import numpy as np

from remguard.data.annotations import parse_scoring_file
from remguard.data.epochs import build_epoch_timeline
from remguard.features import FEATURE_NAMES, epoch_feature_vector

mne.set_log_level("ERROR")

PREPROCESS = ROOT / "data" / "preprocess"
OUT = ROOT / "data" / "features" / "nights"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted(path for path in PREPROCESS.iterdir() if (path / "preprocessing.json").is_dir() is False and (path / "preprocessing.json").exists())
    print(f"{len(folders)} preprocessed nights", flush=True)
    rows = []
    for index, folder in enumerate(folders, start=1):
        target = OUT / f"{folder.name}.npz"
        if target.exists():
            print(f"[{index}/{len(folders)}] {folder.name} already extracted", flush=True)
            rows.append(_row_from_saved(target))
            continue
        print(f"[{index}/{len(folders)}] {folder.name}", flush=True)
        reason, table = _features(folder)
        np.savez_compressed(
            target,
            recording_id=np.asarray(table["recording_id"], dtype=str),
            onset_sec=np.asarray(table["onset_sec"], dtype=np.float64),
            rem=np.asarray(table["rem"], dtype=np.int8),
            values=np.asarray(table["values"], dtype=np.float32),
            feature_names=np.asarray(FEATURE_NAMES),
            reason=np.asarray(reason),
        )
        print(f"  {len(table['rem'])} epochs ({reason})", flush=True)
        rows.append({"recording_id": folder.name, "epochs": len(table["rem"]), "reason": reason})
    status = ROOT / "data" / "features" / "status.json"
    status.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Status: {status}", flush=True)


def _features(folder: Path) -> tuple[str, dict]:
    payload = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    eeg_items = [item for item in payload["modalities"] if item["role"] == "eeg"]
    if not eeg_items:
        return "no_eeg", _empty(folder.name)
    eeg_raws = [(item, mne.io.read_raw_fif(folder / item["file"], preload=True, verbose="ERROR")) for item in eeg_items]
    start = eeg_raws[0][1].info["meas_date"]
    if start is None:
        return "no_start_time", _empty(folder.name)
    text = folder / f"{folder.name}.txt"
    if not text.exists():
        return "no_scores", _empty(folder.name)
    scoring = parse_scoring_file(text, start.time())
    if not scoring.events:
        return "no_scores", _empty(folder.name)
    duration = max(raw.n_times / raw.info["sfreq"] for _, raw in eeg_raws)
    timeline = build_epoch_timeline(scoring, float(duration))
    labeled = [epoch for epoch in timeline.epochs if epoch.rem is not None]
    eog_raws = []
    for item in payload["modalities"]:
        if item["role"] == "eog":
            eog_raws.append(mne.io.read_raw_fif(folder / item["file"], preload=True, verbose="ERROR"))
    onsets = []
    rem = []
    rows = []
    for epoch in labeled:
        eeg_parts = []
        for item, raw in eeg_raws:
            piece = _slice(raw, item["analysis_channels"] or raw.ch_names, epoch.onset_sec)
            if piece is None:
                eeg_parts = []
                break
            eeg_parts.append(piece)
        if not eeg_parts:
            continue
        eeg = np.vstack(eeg_parts)
        eog = None
        if eog_raws:
            eog_parts = [_slice(raw, list(raw.ch_names), epoch.onset_sec) for raw in eog_raws]
            if any(part is None for part in eog_parts):
                eog = None
            else:
                eog = np.vstack(eog_parts)
        rows.append(epoch_feature_vector(eeg, float(eeg_raws[0][1].info["sfreq"]), eog))
        onsets.append(epoch.onset_sec)
        rem.append(int(epoch.rem))
    if not rows:
        return "no_labeled_epochs", _empty(folder.name)
    return "ok", {
        "recording_id": np.array([folder.name] * len(rows), dtype=object),
        "onset_sec": np.asarray(onsets, dtype=float),
        "rem": np.asarray(rem, dtype=int),
        "values": np.vstack(rows),
    }


def _slice(raw, picks, onset_sec: float) -> np.ndarray | None:
    sfreq = float(raw.info["sfreq"])
    start = int(np.round(onset_sec * sfreq))
    stop = start + int(np.round(30.0 * sfreq))
    if start < 0 or stop > raw.n_times:
        return None
    return raw.get_data(picks=list(picks), start=start, stop=stop)


def _empty(recording_id: str) -> dict:
    return {
        "recording_id": np.array([], dtype=object),
        "onset_sec": np.array([], dtype=float),
        "rem": np.array([], dtype=int),
        "values": np.zeros((0, len(FEATURE_NAMES))),
    }


def _row_from_saved(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {"recording_id": path.stem, "epochs": int(data["rem"].shape[0]), "reason": str(data["reason"])}


if __name__ == "__main__":
    main()
