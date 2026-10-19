from datetime import time
from pathlib import Path

import pytest

from remguard.data.annotations import parse_scoring_text
from remguard.data.edf_header import read_edf_header
from remguard.data.epochs import build_epoch_timeline, label_discrepancies

N6_START = time(22, 33, 9)
N12_START = time(15, 14, 22)
DATA = Path("data/raw")

HEADER = "Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation"


def _timeline(rows: str, start: time, duration: float):
    scoring = parse_scoring_text(f"{HEADER}\n{rows}", start)
    return scoring, build_epoch_timeline(scoring, duration)


def test_gap_becomes_an_unscored_epoch_and_rem_stays_expert_only():
    scoring, timeline = _timeline(
        "W\t22:33:09\tSLEEP-S0\t30\tROC-A2\n"
        "R\t22:34:09\tSLEEP-REM\t30\tROC-A2\n",
        N6_START,
        90,
    )
    assert [epoch.stage for epoch in timeline.epochs] == ["W", "UNSCORED", "R"]
    gap = timeline.epochs[1]
    assert gap.source == "gap"
    assert gap.rem is None
    assert gap.fully_annotated is False
    assert gap.issues == ("missing_annotation",)
    assert timeline.epochs[0].rem is False
    assert timeline.epochs[2].rem is True
    assert label_discrepancies(timeline, scoring) == []


def test_overlap_midnight_offset_and_edges():
    scoring, timeline = _timeline(
        "S2\t22:33:00\tSLEEP-S2\t30\tROC-A2\n"
        "S2\t22:33:20\tSLEEP-S2\t30\tROC-A2\n"
        "R\t07:18:09\tSLEEP-REM\t30\tROC-A2\n",
        N6_START,
        32000,
    )
    early, overlapped, rem = timeline.expert_epochs()
    assert early.onset_sec == -9
    assert early.fully_annotated is False
    assert "starts_before_recording" in early.issues
    assert "overlap" in early.issues and "overlap" in overlapped.issues
    assert early.rem is None and overlapped.rem is None
    expected = (7 * 3600 + 18 * 60 + 9 + 24 * 3600) - (22 * 3600 + 33 * 60 + 9)
    assert rem.onset_sec == expected
    assert rem.rem is True
    assert rem.fully_annotated is True
    assert label_discrepancies(timeline, scoring) == []


def test_epoch_ending_at_the_recording_edge_is_kept():
    _scoring, inside = _timeline("S2\t15:14:52\tSLEEP-S2\t30\tLOC-A2\n", N12_START, 60)
    # 15:14:52 is 30 s after 15:14:22, so the epoch occupies 30–60 s.
    epoch = inside.expert_epochs()[0]
    assert epoch.onset_sec == 30
    assert epoch.fully_annotated is True
    assert epoch.rem is False

    _scoring, past = _timeline("S2\t15:14:52\tSLEEP-S2\t30\tLOC-A2\n", N12_START, 59)
    epoch = past.expert_epochs()[0]
    assert epoch.fully_annotated is False
    assert "extends_past_recording" in epoch.issues
    assert epoch.rem is None


def test_unaligned_gap_and_ambiguous_stage_are_excluded_from_the_binary_target():
    scoring, timeline = _timeline(
        "W\t15:14:22\tSLEEP-S0\t30\tLOC-A2\n"
        "1S3\t15:15:37\tSLEEP-S3\t30\tLOC-A2\n"
        "MT\t15:16:07\tSLEEP-MT\t30\tLOC-A2\n",
        N12_START,
        135,
    )
    gap, ambiguous, movement = timeline.epochs[1:]
    assert gap.issues == ("missing_annotation", "unaligned_gap")
    assert abs(gap.duration_sec - 45) < 1e-6
    assert ambiguous.stage == "S3"
    assert ambiguous.fully_annotated is False
    assert ambiguous.rem is None
    assert movement.stage == "MT"
    assert movement.fully_annotated is True
    assert movement.rem is None
    assert label_discrepancies(timeline, scoring) == []


def test_stage_and_event_mismatch_is_not_a_clean_rem_label():
    scoring, timeline = _timeline(
        "R\t15:14:22\tSLEEP-S2\t30\tLOC-A2\n",
        N12_START,
        30,
    )
    epoch = timeline.epochs[0]
    assert epoch.stage == "S2"
    assert epoch.rem is None
    assert epoch.fully_annotated is False
    assert any(issue.startswith("stage_event_mismatch") for issue in epoch.issues)
    assert label_discrepancies(timeline, scoring) == []


@pytest.mark.skipif(not (DATA / "n12.txt").exists(), reason="proof-of-concept scores are not downloaded")
@pytest.mark.parametrize("recording_id", ["n12", "n6", "rbd1"])
def test_proof_of_concept_scores_match_the_epoch_timeline(recording_id: str):
    header = read_edf_header(DATA / f"{recording_id}.edf")
    text = (DATA / f"{recording_id}.txt").read_text(encoding="latin1")
    scoring = parse_scoring_text(text, header.start_time)
    timeline = build_epoch_timeline(scoring, header.duration_sec)
    assert label_discrepancies(timeline, scoring) == []
    assert all(epoch.rem is not True or epoch.stage == "R" for epoch in timeline.epochs)
    assert timeline.counts()["rem"] == sum(event.is_rem and event.duration_sec == 30 for event in scoring.events)
