from datetime import time

from remguard.data.annotations import parse_scoring_text


N12_START = time(15, 14, 22)
N6_START = time(22, 33, 9)


def test_position_column_is_optional_and_cap_rows_are_separated():
    text = """
Recording Date:\t01/01/1999
Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\t22:38:39\tSLEEP-S0\t30\tROC-A2
S2\t22:39:09\tSLEEP-S2\t30\tROC-A2
S2\t22:39:21\tMCAP-A1\t8\tC3-A2
R\t07:18:09\tSLEEP-REM\t30\tROC-A2
"""
    scoring = parse_scoring_text(text, N6_START)
    assert scoring.n_cap_events == 1
    assert [event.stage for event in scoring.events] == ["W", "S2", "R"]
    assert scoring.events[0].onset_sec == 330
    # 07:18:09 is after midnight, so it is on the next day relative to 22:33:09.
    expected = (7 * 3600 + 18 * 60 + 9 + 24 * 3600) - (22 * 3600 + 33 * 60 + 9)
    assert scoring.events[2].onset_sec == expected
    assert scoring.events[2].is_rem


def test_unparsed_stage_column_is_flagged_and_not_called_rem():
    text = """
Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
1S3\tUnknown Position\t19:14:52\tSLEEP-S3\t30\tLOC-A2
"""
    scoring = parse_scoring_text(text, N12_START)
    event = scoring.events[0]
    assert event.stage == "S3"
    assert "ambiguous_stage" in event.issues
    assert event.is_rem is False


def test_overlap_and_short_rows_are_explicit():
    text = """
Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
S2\tUnknown Position\t15:14:22\tSLEEP-S2\t30\tLOC-A2
S2\tUnknown Position\t15:14:32\tSLEEP-S2\t30\tLOC-A2
too-short
"""
    scoring = parse_scoring_text(text, N12_START)
    assert scoring.n_ignored_rows == 1
    assert any(issue.startswith("short_row") for issue in scoring.issues)
    assert all("overlap" in event.issues for event in scoring.events)


def test_dotted_clocks_are_recording_relative():
    text = """
Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\tUnknown Position\t22.18.17\tSLEEP-S0\t30\tEEG-Fp2-F4
R\tUnknown Position\t22.18.47\tSLEEP-REM\t30\tEEG-Fp2-F4
"""
    scoring = parse_scoring_text(text, time(22, 18, 17))
    assert [event.onset_sec for event in scoring.events] == [0, 30]
    assert scoring.events[1].is_rem

    rem_column = """
Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
REM\tUnknown Position\t22.18.47\tSLEEP-REM\t30\tEEG-Fp2-F4
"""
    scored = parse_scoring_text(rem_column, time(22, 18, 17))
    assert scored.events[0].stage == "R"
    assert scored.events[0].is_rem
    assert "ambiguous_stage" not in scored.events[0].issues


def test_clock_before_start_stays_negative_within_one_minute():
    text = """
Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\t22:33:00\tSLEEP-S0\t30\tROC-A2
"""
    scoring = parse_scoring_text(text, N6_START)
    assert scoring.events[0].onset_sec == -9
