"""Time-indexed expert labels for 30-second epochs.

REM is taken only from expert ``SLEEP-REM`` rows. Group names and chin EMG
are not used. Movement time (MT) stays in the hypnogram and is left out of
the binary REM versus non-REM/wake target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from remguard.data.annotations import ScoringFile, SleepEvent

EPOCH_SEC = 30.0
BINARY_NEGATIVE = {"W", "S1", "S2", "S3", "S4"}
SCORED_STAGES = BINARY_NEGATIVE | {"R", "MT"}
_BLOCKING = {"ambiguous_stage", "overlap", "nonstandard_duration"}


@dataclass(frozen=True)
class EpochLabel:
    onset_sec: float
    duration_sec: float
    stage: str
    rem: bool | None
    fully_annotated: bool
    source: str
    issues: tuple[str, ...] = ()

    @property
    def end_sec(self) -> float:
        return self.onset_sec + self.duration_sec


@dataclass
class EpochTimeline:
    epoch_sec: float
    duration_sec: float
    epochs: list[EpochLabel] = field(default_factory=list)

    def expert_epochs(self) -> list[EpochLabel]:
        return [epoch for epoch in self.epochs if epoch.source == "expert"]

    def counts(self) -> dict[str, int]:
        rem = sum(epoch.rem is True for epoch in self.epochs)
        non_rem = sum(epoch.rem is False for epoch in self.epochs)
        excluded = sum(epoch.rem is None for epoch in self.epochs)
        full = sum(epoch.fully_annotated for epoch in self.epochs)
        gaps = sum(epoch.source == "gap" for epoch in self.epochs)
        flagged = sum(bool(epoch.issues) and epoch.source == "expert" for epoch in self.epochs)
        return {
            "epochs": len(self.epochs),
            "fully_annotated": full,
            "rem": rem,
            "non_rem": non_rem,
            "excluded_from_binary": excluded,
            "gap_epochs": gaps,
            "flagged_expert": flagged,
        }


def build_epoch_timeline(
    scoring: ScoringFile,
    duration_sec: float,
    epoch_sec: float = EPOCH_SEC,
) -> EpochTimeline:
    """Place expert rows on a recording-time line and fill the holes."""
    epochs: list[EpochLabel] = []
    cursor = 0.0
    for event in scoring.events:
        if event.onset_sec == event.onset_sec and event.onset_sec > cursor + 0.5:
            epochs.extend(_gap_epochs(cursor, event.onset_sec, epoch_sec))
        epochs.append(_from_event(event, duration_sec, epoch_sec))
        if event.end_sec == event.end_sec:
            cursor = max(cursor, event.end_sec)
    if duration_sec == duration_sec and duration_sec > cursor + 0.5:
        epochs.extend(_gap_epochs(cursor, duration_sec, epoch_sec))
    return EpochTimeline(epoch_sec=epoch_sec, duration_sec=duration_sec, epochs=epochs)


def label_discrepancies(timeline: EpochTimeline, scoring: ScoringFile) -> list[str]:
    """Return mismatches between expert-sourced epochs and the scoring file."""
    experts = timeline.expert_epochs()
    problems: list[str] = []
    if len(experts) != len(scoring.events):
        problems.append(f"expert_count:{len(experts)}!={len(scoring.events)}")
    for index, (label, event) in enumerate(zip(experts, scoring.events)):
        if label.stage != event.stage:
            problems.append(f"stage:{index}:{label.stage}!={event.stage}")
        if abs(label.onset_sec - event.onset_sec) > 1e-6:
            problems.append(f"onset:{index}:{label.onset_sec}!={event.onset_sec}")
        if label.rem is True and not event.is_rem:
            problems.append(f"rem_without_expert_rem:{index}")
        if event.is_rem and label.fully_annotated and label.rem is not True:
            problems.append(f"expert_rem_missing_binary_label:{index}")
    return problems


def _from_event(event: SleepEvent, duration_sec: float, epoch_sec: float) -> EpochLabel:
    issues = list(event.issues)
    if event.onset_sec == event.onset_sec and event.onset_sec < -1e-6:
        issues.append("starts_before_recording")
    if (
        duration_sec == duration_sec
        and event.onset_sec == event.onset_sec
        and event.duration_sec == event.duration_sec
        and event.onset_sec + event.duration_sec > duration_sec + 1e-3
    ):
        issues.append("extends_past_recording")
    fully = _fully_annotated(event, issues, duration_sec, epoch_sec)
    return EpochLabel(
        onset_sec=event.onset_sec,
        duration_sec=event.duration_sec,
        stage=event.stage,
        rem=_rem_label(event.stage, fully),
        fully_annotated=fully,
        source="expert",
        issues=tuple(issues),
    )


def _fully_annotated(
    event: SleepEvent,
    issues: list[str],
    duration_sec: float,
    epoch_sec: float,
) -> bool:
    if event.stage not in SCORED_STAGES:
        return False
    if abs(event.duration_sec - epoch_sec) > 1e-3:
        return False
    if event.onset_sec != event.onset_sec or event.onset_sec < -1e-6:
        return False
    if duration_sec == duration_sec and event.onset_sec + epoch_sec > duration_sec + 1e-3:
        return False
    for issue in issues:
        if issue in _BLOCKING or issue.startswith(
            ("stage_event_mismatch", "bad_", "nonpositive", "unknown_event", "unparsed_stage")
        ):
            return False
    return True


def _rem_label(stage: str, fully_annotated: bool) -> bool | None:
    if not fully_annotated:
        return None
    if stage == "R":
        return True
    if stage in BINARY_NEGATIVE:
        return False
    return None


def _gap_epochs(start: float, end: float, epoch_sec: float) -> list[EpochLabel]:
    span = end - start
    if span <= 0.5:
        return []
    steps = span / epoch_sec
    nearest = round(steps)
    if abs(steps - nearest) < 1e-3 and nearest >= 1:
        return [
            EpochLabel(
                onset_sec=start + index * epoch_sec,
                duration_sec=epoch_sec,
                stage="UNSCORED",
                rem=None,
                fully_annotated=False,
                source="gap",
                issues=("missing_annotation",),
            )
            for index in range(int(nearest))
        ]
    return [
        EpochLabel(
            onset_sec=start,
            duration_sec=span,
            stage="UNSCORED",
            rem=None,
            fully_annotated=False,
            source="gap",
            issues=("missing_annotation", "unaligned_gap"),
        )
    ]


def plot_hypnogram(
    timeline: EpochTimeline,
    scoring: ScoringFile,
    out_path: str,
    title: str,
) -> str:
    """Save a hypnogram of the epoch timeline above the original scoring rows."""
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, sharey=True)
    _draw_epochs(axes[0], timeline.epochs, "Epoch labels")
    raw = [
        EpochLabel(
            onset_sec=event.onset_sec,
            duration_sec=event.duration_sec,
            stage=event.stage,
            rem=None,
            fully_annotated=not event.issues,
            source="expert",
            issues=event.issues,
        )
        for event in scoring.events
    ]
    _draw_epochs(axes[1], raw, "Scoring file")
    axes[1].set_xlabel("Hours from EDF start")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    plt.close(figure)
    return str(out)


_LEVELS = {
    "W": 7,
    "R": 6,
    "S1": 5,
    "S2": 4,
    "S3": 3,
    "S4": 2,
    "MT": 1,
    "UNSCORED": 0,
    "AMBIGUOUS": -1,
}
_TICKS = ("W", "R", "S1", "S2", "S3", "S4", "MT", "UNSCORED")


def _draw_epochs(axis, epochs: list[EpochLabel], panel_title: str) -> None:
    show_ambiguous = any(epoch.stage == "AMBIGUOUS" for epoch in epochs)
    for epoch in epochs:
        if epoch.onset_sec != epoch.onset_sec or epoch.duration_sec != epoch.duration_sec:
            continue
        level = _LEVELS.get(epoch.stage, _LEVELS["AMBIGUOUS"])
        if epoch.source == "gap":
            color = "#b0b0b0"
        elif epoch.issues:
            color = "#d9782d"
        else:
            color = "#1f4e79"
        axis.hlines(level, epoch.onset_sec / 3600.0, epoch.end_sec / 3600.0, colors=color, linewidth=2)
    names = list(_TICKS)
    if show_ambiguous:
        names.append("AMBIGUOUS")
    axis.set_yticks([_LEVELS[name] for name in names])
    axis.set_yticklabels(names)
    axis.set_ylim(-1.6 if show_ambiguous else -0.6, 7.6)
    axis.set_title(panel_title)
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
