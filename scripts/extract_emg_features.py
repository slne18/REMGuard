"""Chin activity during expert REM and during held-out XGBoost REM.

Chin EMG is not a detector input. Each night is scored by the model fold that
held that night out. The quiet-sleep baseline is the expert S2 (else S3, else
S4) median, and the same baseline is used for both sets of REM intervals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mne
import numpy as np

from remguard.boosting import _fit_full, _select_params, inner_participant_split, participant_folds
from remguard.data.annotations import parse_scoring_file
from remguard.data.epochs import EPOCH_SEC, build_epoch_timeline
from remguard.emg_features import baseline_intervals_for, chin_features, expert_rem_intervals
from train_xgboost_cv import GRID, NIGHTS, load_nights

mne.set_log_level("ERROR")

PREPROCESS = ROOT / "data" / "preprocess"
REPORT = ROOT / "reports" / "chin_emg_during_rem.md"
RANK_REPORT = ROOT / "reports" / "chin_versus_healthy.md"
RANK_TABLE = ROOT / "reports" / "chin_versus_healthy.csv"
LOW_REPORT = ROOT / "reports" / "low_rbd_nights.md"
GROUPS = ("Healthy", "RBD", "NFLE", "PLM", "Insomnia", "Narcolepsy", "SDB", "Bruxism")
MIN_REM_SEC = 30.0


def main() -> None:
    table, _skipped = load_nights(NIGHTS)
    print(f"{table.n_epochs} labeled epochs", flush=True)
    predicted = _held_out_rem(table)
    rows = []
    skipped = []
    folders = sorted(path for path in PREPROCESS.iterdir() if (path / "preprocessing.json").is_file())
    for index, folder in enumerate(folders, start=1):
        print(f"[{index}/{len(folders)}] {folder.name}", flush=True)
        row, reason = _night(folder, predicted.get(folder.name, []))
        if row is None:
            skipped.append({"recording_id": folder.name, "reason": reason})
            print(f"  skipped ({reason})", flush=True)
            continue
        rows.append(row)
        print(
            f"  expert {row['expert_rem_min']:.0f} min, elevated {row['expert_elevated']:.0f}% | "
            f"detected {row['detected_rem_min']:.0f} min, elevated {row['detected_elevated']:.0f}%",
            flush=True,
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_report(rows, skipped), encoding="utf-8")
    ranked = _rank_against_healthy(rows)
    _attach_agreed_ranks(ranked)
    RANK_REPORT.write_text(_rank_report(ranked), encoding="utf-8")
    _write_rank_table(ranked)
    LOW_REPORT.write_text(_low_rbd_report(ranked), encoding="utf-8")
    print(f"Wrote {REPORT}", flush=True)
    print(f"Wrote {RANK_REPORT}", flush=True)
    print(f"Wrote {LOW_REPORT}", flush=True)


def _held_out_rem(table) -> dict[str, list[tuple[float, float]]]:
    participants = list(dict.fromkeys(table.recording_id.tolist()))
    by_night: dict[str, list[tuple[float, float]]] = {recording: [] for recording in participants}
    for fold_index, test_ids in enumerate(participant_folds(participants, n_splits=5, seed=0)):
        test_set = set(test_ids.tolist())
        train_ids = np.array([recording for recording in participants if recording not in test_set])
        fit_ids, val_ids = inner_participant_split(train_ids, seed=fold_index + 1)
        choice = _select_params(table, fit_ids, val_ids, GRID, n_estimators=400, early_stopping_rounds=30, n_jobs=4)
        model = _fit_full(table, train_ids, choice["params"], choice["best_iteration"], n_jobs=4)
        test_mask = np.array([recording in test_set for recording in table.recording_id])
        called = model.predict(table.values[test_mask]).astype(int)
        for recording, onset, label in zip(table.recording_id[test_mask], table.onset_sec[test_mask], called):
            if int(label) == 1:
                by_night[str(recording)].append((float(onset), float(onset) + EPOCH_SEC))
        print(f"  fold {fold_index + 1}: {int(np.sum(called))} detected REM epochs", flush=True)
    return by_night


def _night(folder: Path, predicted: list[tuple[float, float]]) -> tuple[dict | None, str]:
    payload = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    emg_items = [item for item in payload["modalities"] if item["role"] == "emg_chin"]
    if not emg_items:
        return None, "no chin EMG"
    channels = []
    start = None
    duration = 0.0
    for item in emg_items:
        raw = mne.io.read_raw_fif(folder / item["file"], preload=True, verbose="ERROR")
        if start is None:
            start = raw.info["meas_date"]
        duration = max(duration, raw.n_times / float(raw.info["sfreq"]))
        sfreq = float(raw.info["sfreq"])
        for name in raw.ch_names:
            channels.append({"name": name, "samples": raw.get_data(picks=[name])[0] * 1e6, "sfreq": sfreq})
    if start is None:
        return None, "no start time"
    scores = folder / f"{folder.name}.txt"
    if not scores.exists():
        return None, "no scores"
    timeline = build_epoch_timeline(parse_scoring_file(scores, start.time()), float(duration))
    baseline, stage = baseline_intervals_for(timeline)
    if stage is None:
        return None, "no quiet-sleep baseline"
    expert_intervals = expert_rem_intervals(timeline)
    expert = _average(channels, expert_intervals, baseline, None)
    locked = (expert["baseline_uv"], expert["threshold_uv"])
    detected = _average(channels, predicted, [], locked)
    agreed = _average(channels, _shared_intervals(expert_intervals, predicted), [], locked)
    if expert["baseline_uv"] != expert["baseline_uv"]:
        return None, "no quiet-sleep baseline"
    return {
        "recording_id": folder.name,
        "group": _cohort(folder.name),
        "baseline_stage": stage,
        "expert_rem_min": expert["rem_duration_sec"] / 60.0,
        "detected_rem_min": detected["rem_duration_sec"] / 60.0,
        "agreed_rem_min": agreed["rem_duration_sec"] / 60.0,
        "expert_elevated": expert["elevated_time_percent"],
        "detected_elevated": detected["elevated_time_percent"],
        "agreed_elevated": agreed["elevated_time_percent"],
        "expert_bursts": expert["bursts_per_hour"],
        "detected_bursts": detected["bursts_per_hour"],
        "expert_tonic": expert["tonic_like_epoch_percent"],
        "detected_tonic": detected["tonic_like_epoch_percent"],
        "expert_ratio": _ratio(expert["median_rectified_uv"], expert["baseline_uv"]),
        "expert_usable": _usable(expert["valid_emg_duration_sec"], expert["rem_duration_sec"]),
    }, "ok"


def _average(channels, intervals, baseline, locked) -> dict[str, float]:
    measured = []
    for channel in channels:
        features = chin_features(channel["samples"], channel["sfreq"], intervals, baseline, locked_baseline=locked)
        if "flat_channel" in features.notes:
            continue
        measured.append(features)
    if not measured:
        nan = float("nan")
        return {
            "rem_duration_sec": 0.0,
            "valid_emg_duration_sec": 0.0,
            "elevated_time_percent": nan,
            "bursts_per_hour": nan,
            "tonic_like_epoch_percent": nan,
            "median_rectified_uv": nan,
            "baseline_uv": nan if locked is None else locked[0],
            "threshold_uv": nan if locked is None else locked[1],
        }
    summary = {}
    for key in (
        "rem_duration_sec",
        "valid_emg_duration_sec",
        "elevated_time_percent",
        "bursts_per_hour",
        "tonic_like_epoch_percent",
        "median_rectified_uv",
        "baseline_uv",
        "threshold_uv",
    ):
        values = np.array([getattr(features, key) for features in measured], dtype=float)
        summary[key] = float(np.nanmean(values))
    return summary


def _cohort(recording_id: str) -> str:
    name = recording_id.lower()
    for prefix, label in (
        ("nfle", "NFLE"),
        ("narco", "Narcolepsy"),
        ("brux", "Bruxism"),
        ("plm", "PLM"),
        ("ins", "Insomnia"),
        ("rbd", "RBD"),
        ("sdb", "SDB"),
    ):
        if name.startswith(prefix):
            return label
    if name.startswith("n") and name[1:].isdigit():
        return "Healthy"
    return recording_id


def _report(rows: list[dict], skipped: list[dict]) -> str:
    lines = [
        "# Chin activity during REM",
        "",
        "Chin EMG was not used to detect REM. For each night, the REM epochs are the ones the XGBoost model called when that night was held out of training. The comparison is the same measurement inside the expert's REM epochs.",
        "",
        "A sample counts as elevated when it is above twice that night's own quiet-sleep level. Quiet sleep is expert S2, or S3, or S4 if S2 is shorter than 30 seconds. Wake is not the baseline. Because each night has its own level, the percentages below can be compared across nights. The raw microvolt values cannot.",
        "",
        "The table is the median across nights. Activity percentages use nights with at least 30 seconds of that kind of REM. REM minutes include nights where none was found.",
        "",
        "| Group | Nights | REM minutes, expert | REM minutes, detected | Elevated time, expert | Elevated time, detected | Bursts per hour, expert | Bursts per hour, detected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        members = [row for row in rows if row["group"] == group]
        if not members:
            continue
        lines.append(
            "| {group} | {n} | {expert_min} | {detected_min} | {expert_elevated} | {detected_elevated} | {expert_bursts} | {detected_bursts} |".format(
                group=group,
                n=len(members),
                expert_min=_fmt(_median(members, "expert_rem_min")),
                detected_min=_fmt(_median(members, "detected_rem_min")),
                expert_elevated=_fmt(_median(_with_rem(members, "expert_rem_min"), "expert_elevated"), "%"),
                detected_elevated=_fmt(_median(_with_rem(members, "detected_rem_min"), "detected_elevated"), "%"),
                expert_bursts=_fmt(_median(_with_rem(members, "expert_rem_min"), "expert_bursts")),
                detected_bursts=_fmt(_median(_with_rem(members, "detected_rem_min"), "detected_bursts")),
            )
        )
    healthy = [row for row in rows if row["group"] == "Healthy"]
    rbd = [row for row in rows if row["group"] == "RBD"]
    lines.extend(["", _reading(healthy, rbd), ""])
    if skipped:
        lines.append("No chin measurement for: " + ", ".join(f"{row['recording_id']} ({row['reason']})" for row in skipped) + ".")
        lines.append("")
    lines.append(
        "Elevated time and burst rate describe how much the chin stays active during REM in this collection. They are not a clinical RBD score and they are not a Parkinson's score."
    )
    lines.append("")
    return "\n".join(lines)


def _with_rem(rows: list[dict], key: str) -> list[dict]:
    return [row for row in rows if row[key] >= MIN_REM_SEC / 60.0]


def _median(rows: list[dict], key: str) -> float:
    values = np.array([row[key] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def _fmt(value: float, suffix: str = "") -> str:
    if value != value:
        return "n/a"
    return f"{value:.0f}{suffix}"


def _reading(healthy: list[dict], rbd: list[dict]) -> str:
    healthy_expert = _median(_with_rem(healthy, "expert_rem_min"), "expert_elevated")
    rbd_expert = _median(_with_rem(rbd, "expert_rem_min"), "expert_elevated")
    healthy_detected = _median(_with_rem(healthy, "detected_rem_min"), "detected_elevated")
    rbd_detected = _median(_with_rem(rbd, "detected_rem_min"), "detected_elevated")
    healthy_tonic = _median(_with_rem(healthy, "expert_rem_min"), "expert_tonic")
    rbd_tonic = _median(_with_rem(rbd, "expert_rem_min"), "expert_tonic")
    return (
        f"On expert REM, the median chin elevated time is {healthy_expert:.0f}% in healthy nights and {rbd_expert:.0f}% in RBD nights. "
        f"Tonic-like epochs, where at least half of the epoch is above the threshold, are {healthy_tonic:.0f}% and {rbd_tonic:.0f}%. "
        f"Using the detector's REM instead, elevated time is {healthy_detected:.0f}% in healthy nights and {rbd_detected:.0f}% in RBD nights. "
        "Detected REM lasts longer than expert REM, and the higher percentages come from epochs the expert did not call REM, where the chin is more active."
    )


def _rank_against_healthy(rows: list[dict]) -> list[dict]:
    healthy = [row for row in rows if row["group"] == "Healthy"]
    ranked = []
    for row in rows:
        expert_reference = _healthy_reference(healthy, row, "expert_rem_min", "expert_elevated")
        detected_reference = _healthy_reference(healthy, row, "detected_rem_min", "detected_elevated")
        ranked.append(
            {
                **row,
                "expert_rank": _percentile(row, "expert_rem_min", "expert_elevated", expert_reference),
                "detected_rank": _percentile(row, "detected_rem_min", "detected_elevated", detected_reference),
            }
        )
    return ranked


def _healthy_reference(healthy: list[dict], row: dict, duration_key: str, value_key: str) -> np.ndarray:
    others = [night for night in healthy if night["recording_id"] != row["recording_id"]]
    values = [
        night[value_key]
        for night in others
        if night[duration_key] >= MIN_REM_SEC / 60.0 and night[value_key] == night[value_key]
    ]
    return np.array(values, dtype=float)


def _percentile(row: dict, duration_key: str, value_key: str, reference: np.ndarray) -> float:
    value = row[value_key]
    if row[duration_key] < MIN_REM_SEC / 60.0 or value != value or reference.size == 0:
        return float("nan")
    return float(100.0 * np.mean(reference < value))


def _rank_report(rows: list[dict]) -> str:
    healthy = [row for row in rows if row["group"] == "Healthy"]
    expert_reference = _healthy_reference(healthy, {"recording_id": ""}, "expert_rem_min", "expert_elevated")
    detected_reference = _healthy_reference(healthy, {"recording_id": ""}, "detected_rem_min", "detected_elevated")
    lines = [
        "# Chin activity compared with healthy nights",
        "",
        "Each night is ranked by the share of REM in which the chin stays above that night's own quiet-sleep level. The rank is the percentage of healthy nights with a lower value. A healthy night is left out of its own comparison. A rank of 50 is the healthy midpoint. A rank of 90 means the night is higher than 90 percent of the healthy nights.",
        "",
        f"The healthy elevated time reaches {_fmt(float(np.percentile(expert_reference, 90)), '%')} at the 90th percentile during expert REM, and {_fmt(float(np.percentile(detected_reference, 90)), '%')} during detected REM.",
        "",
        "Nights with less than 30 seconds of that REM are left out of that column. The high count is nights at or above a rank of 90.",
        "",
        "| Group | Nights | Median rank, expert REM | High, expert REM | Median rank, detected REM | High, detected REM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        members = [row for row in rows if row["group"] == group]
        if not members:
            continue
        lines.append(
            "| {group} | {n} | {expert_rank} | {expert_high} | {detected_rank} | {detected_high} |".format(
                group=group,
                n=len(members),
                expert_rank=_fmt(_median(members, "expert_rank")),
                expert_high=_high_count(members, "expert_rank"),
                detected_rank=_fmt(_median(members, "detected_rank")),
                detected_high=_high_count(members, "detected_rank"),
            )
        )
    lines.extend(["", _rank_reading(rows), ""])
    lines.append(
        "A high rank means the chin was active during REM more often than in the healthy nights. It is a comparison inside this collection, not a clinical RBD score and not a Parkinson's score."
    )
    lines.append("")
    lines.append(f"One row per night: `{RANK_TABLE.name}`.")
    lines.append("")
    return "\n".join(lines)


def _high_count(rows: list[dict], key: str) -> str:
    scored = [row for row in rows if row[key] == row[key]]
    high = sum(row[key] >= 90.0 for row in scored)
    return f"{high}/{len(scored)}"


def _rank_reading(rows: list[dict]) -> str:
    healthy = [row for row in rows if row["group"] == "Healthy"]
    rbd = [row for row in rows if row["group"] == "RBD"]
    return (
        f"On expert REM, {_high_count(rbd, 'expert_rank')} RBD nights sit at or above 90 percent of the healthy nights, "
        f"against {_high_count(healthy, 'expert_rank')} healthy nights. "
        f"On detected REM, the counts are {_high_count(rbd, 'detected_rank')} for RBD and {_high_count(healthy, 'detected_rank')} for healthy nights."
    )


def _write_rank_table(rows: list[dict]) -> None:
    header = "recording_id,group,expert_elevated_percent,expert_rank,detected_elevated_percent,detected_rank"
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["recording_id"],
                    row["group"],
                    _csv_number(row["expert_elevated"]),
                    _csv_number(row["expert_rank"]),
                    _csv_number(row["detected_elevated"]),
                    _csv_number(row["detected_rank"]),
                ]
            )
        )
    RANK_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_number(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.2f}"


def _shared_intervals(expert: list[tuple[float, float]], predicted: list[tuple[float, float]]) -> list[tuple[float, float]]:
    predicted_onsets = {round(start, 3) for start, _end in predicted}
    return [(start, end) for start, end in expert if round(start, 3) in predicted_onsets]


def _ratio(level: float, baseline: float) -> float:
    if level != level or baseline != baseline or baseline <= 0:
        return float("nan")
    return float(level / baseline)


def _usable(valid_sec: float, rem_sec: float) -> float:
    if rem_sec != rem_sec or rem_sec <= 0 or valid_sec != valid_sec:
        return float("nan")
    return float(valid_sec / rem_sec)


def _attach_agreed_ranks(rows: list[dict]) -> None:
    healthy = [row for row in rows if row["group"] == "Healthy"]
    for row in rows:
        reference = _healthy_reference(healthy, row, "agreed_rem_min", "agreed_elevated")
        row["agreed_rank"] = _percentile(row, "agreed_rem_min", "agreed_elevated", reference)


def _low_rbd_report(rows: list[dict]) -> str:
    rbd = sorted((row for row in rows if row["group"] == "RBD"), key=lambda row: row["expert_rank"])
    low = [row for row in rbd if row["expert_rank"] < 60]
    middle = [row for row in rbd if 60 <= row["expert_rank"] < 80]
    healthy = [row for row in rows if row["group"] == "Healthy"]
    lines = [
        "# RBD nights the cutoff missed",
        "",
        f"On expert REM, {len(low)} RBD nights have a rank of 53 or lower. A rank of 87 is just under the cutoff of 90. These nights are the ones that sit in the healthy range.",
        "",
        "The typical level is the median chin size during expert REM divided by that night's own quiet-sleep level. The threshold is twice quiet sleep, so a typical level below 2 means most of the epoch stays under the line. Usable EMG is the share of the REM interval that is not a flat dropout. Microvolts are not compared across nights.",
        "",
        "| Night | Rank | REM minutes | Elevated time | Typical level | Usable EMG |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in low:
        lines.append(_low_row(row))
    lines.extend(["", "What each night shows:", ""])
    for row in low:
        lines.append(f"- `{row['recording_id']}`: {_reason(row)}.")
    if middle:
        names = ", ".join(f"`{row['recording_id']}` ({row['expert_rank']:.0f})" for row in middle)
        lines.extend(["", f"{len(middle)} more RBD nights sit between the healthy range and the rank-87 cluster: {names}.", ""])
    lines.extend(
        [
            "## Epochs both the expert and the model call REM",
            "",
            "Detected REM by itself includes epochs the expert did not call REM. This section keeps only the overlap. A night with less than 30 seconds of overlap is left out of the rank.",
            "",
            "| Night | Expert rank | Overlap minutes | Overlap elevated time | Overlap rank |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in low:
        lines.append(
            f"| {row['recording_id']} | {_fmt(row['expert_rank'])} | {_fmt(row['agreed_rem_min'])} | "
            f"{_fmt(row['agreed_elevated'], '%')} | {_fmt(row['agreed_rank'])} |"
        )
    lines.extend(
        [
            "",
            _overlap_reading(low, rbd, healthy),
            "",
            "This is a description of these nights. It is not a clinical RBD score and it is not a Parkinson's score.",
            "",
        ]
    )
    return "\n".join(lines)


def _low_row(row: dict) -> str:
    ratio = "n/a" if row["expert_ratio"] != row["expert_ratio"] else f"{row['expert_ratio']:.1f}×"
    return (
        f"| {row['recording_id']} | {_fmt(row['expert_rank'])} | {_fmt(row['expert_rem_min'])} | "
        f"{_fmt(row['expert_elevated'], '%')} | {ratio} | {_fmt(100.0 * row['expert_usable'], '%')} |"
    )


def _overlap_reading(low: list[dict], rbd: list[dict], healthy: list[dict]) -> str:
    stayed = [
        row["recording_id"]
        for row in low
        if row["agreed_rank"] == row["agreed_rank"] and row["agreed_rank"] < 60
    ]
    rose = [
        f"`{row['recording_id']}` ({row['agreed_rank']:.0f})"
        for row in low
        if row["agreed_rank"] == row["agreed_rank"] and row["agreed_rank"] >= 60
    ]
    missing = [row["recording_id"] for row in low if row["agreed_rank"] != row["agreed_rank"]]
    parts = [
        f"On the overlap, the median rank is {_fmt(_median(healthy, 'agreed_rank'))} for healthy nights and "
        f"{_fmt(_median(rbd, 'agreed_rank'))} for RBD nights. "
        f"The cutoff of 90 flags {_high_count(rbd, 'agreed_rank')} RBD nights and {_high_count(healthy, 'agreed_rank')} healthy nights."
    ]
    if stayed and not rose and not missing:
        parts.append(
            "The low nights stay at rank 53 or lower on the overlap, so the quiet chin is in the REM both sources agree on."
        )
    else:
        if stayed:
            parts.append(f"{', '.join(stayed)} stay at rank 53 or lower on the overlap.")
        if rose:
            parts.append(f"{', '.join(rose)} move above that range once only the agreed epochs are kept.")
        if missing:
            parts.append(f"{', '.join(missing)} have less than 30 seconds of overlap, so they are not ranked there.")
    return " ".join(parts)


def _reason(row: dict) -> str:
    reasons = []
    if row["expert_rem_min"] < 30:
        reasons.append(f"expert REM lasts {row['expert_rem_min']:.0f} min, shorter than the RBD median of about 70 min")
    if row["expert_usable"] == row["expert_usable"] and row["expert_usable"] < 0.8:
        reasons.append(f"only {100.0 * row['expert_usable']:.0f}% of that REM has usable EMG")
    if row["expert_ratio"] == row["expert_ratio"] and row["expert_ratio"] < 1.5:
        reasons.append(
            f"the typical sample is {row['expert_ratio']:.1f}× its own quiet sleep, so most of the epoch never reaches the 2× threshold"
        )
    elif row["expert_ratio"] == row["expert_ratio"]:
        reasons.append(
            f"the typical sample is {row['expert_ratio']:.1f}× its own quiet sleep and {row['expert_elevated']:.0f}% of the epoch crosses the 2× threshold"
        )
    if not reasons:
        reasons.append("the chin stays near this night's quiet-sleep level through the expert REM that is available")
    return "; ".join(reasons)


if __name__ == "__main__":
    main()
