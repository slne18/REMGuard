"""Parse CAP REMlogic .txt score files into recording-relative times.

Expert sleep stages live in these text files. The EDF reserved field of the
proof-of-concept recordings is empty, so the scores are not read from the EDF.
CAP microstructure rows (MCAP-A1/A2/A3) are counted and then left out of the
sleep-stage list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import time
from pathlib import Path

STAGE_FROM_EVENT = {
    "SLEEP-S0": "W",
    "SLEEP-S1": "S1",
    "SLEEP-S2": "S2",
    "SLEEP-S3": "S3",
    "SLEEP-S4": "S4",
    "SLEEP-REM": "R",
    "SLEEP-MT": "MT",
    "SLEEP-UNSCORED": "UNSCORED",
}

KNOWN_STAGES = {"W", "S1", "S2", "S3", "S4", "R", "MT", "UNSCORED"}
_CLOCK = re.compile(r"^(\d{1,2})[:.](\d{2})[:.](\d{2})$")


@dataclass(frozen=True)
class SleepEvent:
    stage: str
    onset_sec: float
    duration_sec: float
    clock_time: str
    event_code: str
    stage_column: str
    issues: tuple[str, ...] = ()

    @property
    def end_sec(self) -> float:
        return self.onset_sec + self.duration_sec

    @property
    def is_rem(self) -> bool:
        return self.stage == "R" and "ambiguous_stage" not in self.issues


@dataclass
class ScoringFile:
    recording_date: str | None
    events: list[SleepEvent]
    n_cap_events: int = 0
    n_ignored_rows: int = 0
    issues: list[str] = field(default_factory=list)

    def rem_intervals(self) -> list[tuple[float, float]]:
        return [(event.onset_sec, event.end_sec) for event in self.events if event.is_rem]


def parse_scoring_file(path: str | Path, recording_start: time) -> ScoringFile:
    text = Path(path).read_text(encoding="latin1")
    return parse_scoring_text(text, recording_start)


def parse_scoring_text(text: str, recording_start: time) -> ScoringFile:
    lines = text.splitlines()
    recording_date = _recording_date(lines)
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Sleep Stage")),
        None,
    )
    if header_index is None:
        return ScoringFile(
            recording_date=recording_date,
            events=[],
            issues=["missing_header"],
        )

    columns = _column_index(lines[header_index].split("\t"))
    required = ("stage", "time", "event", "duration")
    missing = [name for name in required if name not in columns]
    if missing:
        return ScoringFile(
            recording_date=recording_date,
            events=[],
            issues=[f"missing_columns:{','.join(missing)}"],
        )

    events: list[SleepEvent] = []
    n_cap = 0
    n_ignored = 0
    file_issues: list[str] = []
    for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) <= max(columns.values()):
            n_ignored += 1
            file_issues.append(f"short_row:{line_number}")
            continue
        event_code = parts[columns["event"]].strip()
        if event_code.startswith("MCAP-"):
            n_cap += 1
            continue
        if not event_code.startswith("SLEEP-"):
            n_ignored += 1
            file_issues.append(f"ignored_event:{line_number}:{event_code}")
            continue
        events.append(
            _sleep_event(
                parts=parts,
                columns=columns,
                event_code=event_code,
                recording_start=recording_start,
                line_number=line_number,
            )
        )

    events.sort(key=lambda item: (item.onset_sec, item.event_code))
    events = _mark_overlaps(events)
    return ScoringFile(
        recording_date=recording_date,
        events=events,
        n_cap_events=n_cap,
        n_ignored_rows=n_ignored,
        issues=file_issues,
    )


def _sleep_event(
    parts: list[str],
    columns: dict[str, int],
    event_code: str,
    recording_start: time,
    line_number: int,
) -> SleepEvent:
    issues: list[str] = []
    stage_column = parts[columns["stage"]].strip()
    clock_text = parts[columns["time"]].strip()
    duration_text = parts[columns["duration"]].strip()
    try:
        duration = float(duration_text)
    except ValueError:
        duration = float("nan")
        issues.append(f"bad_duration:{line_number}")
    if duration <= 0 or duration != duration:
        issues.append(f"nonpositive_duration:{line_number}")

    onset = _clock_to_onset(clock_text, recording_start)
    if onset is None:
        onset = float("nan")
        issues.append(f"bad_clock:{line_number}:{clock_text}")

    event_stage = STAGE_FROM_EVENT.get(event_code)
    if event_stage is None:
        event_stage = "AMBIGUOUS"
        issues.append("ambiguous_stage")
        issues.append(f"unknown_event:{event_code}")
    column_stage = _normalize_stage(stage_column)
    if column_stage is None:
        issues.append("ambiguous_stage")
        issues.append(f"unparsed_stage_column:{stage_column}")
    elif column_stage != event_stage:
        issues.append(f"stage_event_mismatch:{stage_column}!={event_stage}")

    if abs(duration - 30.0) > 1e-6 and duration == duration:
        issues.append("nonstandard_duration")

    return SleepEvent(
        stage=event_stage,
        onset_sec=onset,
        duration_sec=duration,
        clock_time=clock_text,
        event_code=event_code,
        stage_column=stage_column,
        issues=tuple(issues),
    )


def _normalize_stage(stage_column: str) -> str | None:
    token = stage_column.strip().upper()
    token = {"REM": "R", "WAKE": "W"}.get(token, token)
    if token in KNOWN_STAGES:
        return token
    return None


def _clock_to_onset(clock_text: str, recording_start: time) -> float | None:
    match = _CLOCK.match(clock_text)
    if match is None:
        return None
    hour, minute, second = (int(value) for value in match.groups())
    if hour > 23 or minute > 59 or second > 59:
        return None
    event_sec = hour * 3600 + minute * 60 + second
    start_sec = recording_start.hour * 3600 + recording_start.minute * 60 + recording_start.second
    # Overnight recordings cross midnight. A clock time more than one minute
    # before the EDF start is treated as the next day.
    if event_sec < start_sec - 60:
        event_sec += 24 * 3600
    return float(event_sec - start_sec)


def _mark_overlaps(events: list[SleepEvent]) -> list[SleepEvent]:
    flagged: set[int] = set()
    for index in range(1, len(events)):
        previous = events[index - 1]
        current = events[index]
        if current.onset_sec != current.onset_sec or previous.end_sec != previous.end_sec:
            continue
        if current.onset_sec < previous.end_sec - 1e-3:
            flagged.add(index - 1)
            flagged.add(index)
    if not flagged:
        return events
    updated: list[SleepEvent] = []
    for index, event in enumerate(events):
        if index in flagged and "overlap" not in event.issues:
            updated.append(replace(event, issues=event.issues + ("overlap",)))
        else:
            updated.append(event)
    return updated


def _column_index(header: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for index, name in enumerate(header):
        key = re.sub(r"[^a-z]", "", name.lower())
        if "stage" in key:
            found["stage"] = index
        elif key.startswith("time"):
            found["time"] = index
        elif key == "event":
            found["event"] = index
        elif key.startswith("duration"):
            found["duration"] = index
        elif "position" in key:
            found["position"] = index
        elif "location" in key:
            found["location"] = index
    return found


def _recording_date(lines: list[str]) -> str | None:
    for line in lines:
        if line.lower().startswith("recording date:"):
            return line.split(":", 1)[1].strip() or None
    return None
