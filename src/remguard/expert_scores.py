"""Optional expert stage files, compared with detected REM.

ISRUC writes one stage number per 30-second epoch, starting at the beginning
of the recording: 0 wake, 1 N1, 2 N2, 3 N3, 5 REM. A CAP REMlogic file is
read with the recording's clock start, the same way as the training scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time

import numpy as np

from remguard.data.annotations import parse_scoring_text
from remguard.features import EPOCH_SEC

_ISRUC_LINE = re.compile(r"[0-5]")
_ISRUC_REM = 5


@dataclass(frozen=True)
class ExpertRem:
    format: str
    rem_blocks: list[tuple[float, float]]
    covered: list[tuple[float, float]]


def parse_expert_scores(text: str, recording_start: time) -> ExpertRem:
    if "Sleep Stage" in text:
        return _from_remlogic(text, recording_start)
    if _is_isruc(text):
        return _from_isruc(text)
    raise ValueError(
        "This score file is not a list of 30-second stages or a REMlogic score. "
        "A stage list has one number per line: 0 wake, 1 N1, 2 N2, 3 N3, 5 REM."
    )


def expert_comparison(
    expert: ExpertRem,
    detected_blocks: list[tuple[float, float]],
    n_epochs: int,
    duration_sec: float,
) -> dict:
    """Compare expert REM with the epochs the model scored."""
    rem = _clip(_merge(expert.rem_blocks), duration_sec)
    covered = _clip(_merge(expert.covered), duration_sec)
    detected = _epoch_mask(detected_blocks, n_epochs)
    expert_rem = _epoch_mask(rem, n_epochs)
    scored = _epoch_mask(covered, n_epochs)
    agreed = detected & expert_rem
    detected_where_scored = detected & scored

    def minutes(mask: np.ndarray) -> float:
        return round(float(mask.sum()) * EPOCH_SEC / 60.0, 1)

    return {
        "format": expert.format,
        "rem_minutes": round(sum(end - start for start, end in rem) / 60.0, 1),
        "rem_blocks": [[round(start, 1), round(end, 1)] for start, end in rem],
        "agreed_minutes": minutes(agreed),
        "expert_only_minutes": minutes(expert_rem & ~detected),
        "detected_only_minutes": minutes(detected_where_scored & ~expert_rem),
        "share_of_expert_also_detected": _share(agreed.sum(), expert_rem.sum()),
        "share_of_detected_also_expert": _share(agreed.sum(), detected_where_scored.sum()),
    }


def _from_isruc(text: str) -> ExpertRem:
    codes = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        token = line.strip()
        if not token:
            continue
        if _ISRUC_LINE.fullmatch(token) is None:
            raise ValueError(
                f"Line {line_number} is {token!r}. Expected one number per line: "
                "0 wake, 1 N1, 2 N2, 3 N3, 5 REM."
            )
        codes.append(int(token))
    if not codes:
        raise ValueError("The expert score file is empty.")
    rem = [(index * EPOCH_SEC, (index + 1) * EPOCH_SEC) for index, code in enumerate(codes) if code == _ISRUC_REM]
    covered = [(0.0, len(codes) * EPOCH_SEC)]
    return ExpertRem("isruc", _merge(rem), covered)


def _from_remlogic(text: str, recording_start: time) -> ExpertRem:
    scoring = parse_scoring_text(text, recording_start)
    events = [event for event in scoring.events if np.isfinite(event.onset_sec) and event.duration_sec > 0]
    if not events:
        raise ValueError("The score file has a header but no sleep stages.")
    rem = [(event.onset_sec, event.end_sec) for event in events if event.is_rem]
    covered = [(event.onset_sec, event.end_sec) for event in events]
    return ExpertRem("remlogic", _merge(rem), _merge(covered))


def _is_isruc(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or _ISRUC_LINE.fullmatch(lines[0]) is None:
        return False
    good = sum(_ISRUC_LINE.fullmatch(line) is not None for line in lines)
    return good / len(lines) >= 0.5


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 0.5:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _clip(intervals: list[tuple[float, float]], duration_sec: float) -> list[tuple[float, float]]:
    clipped = []
    for start, end in intervals:
        onset = max(0.0, start)
        stop = min(float(duration_sec), end)
        if stop > onset:
            clipped.append((onset, stop))
    return clipped


def _epoch_mask(intervals: list[tuple[float, float]], n_epochs: int) -> np.ndarray:
    mask = np.zeros(n_epochs, dtype=bool)
    if n_epochs <= 0:
        return mask
    for start, end in intervals:
        first = max(0, int(np.floor(start / EPOCH_SEC)))
        last = min(n_epochs - 1, int(np.ceil(end / EPOCH_SEC) - 1e-9))
        for index in range(first, last + 1):
            onset = index * EPOCH_SEC
            overlap = min(onset + EPOCH_SEC, end) - max(onset, start)
            if overlap >= EPOCH_SEC / 2.0:
                mask[index] = True
    return mask


def _share(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 1)
