from datetime import time
from pathlib import Path

import pytest

from remguard.expert_scores import expert_comparison, parse_expert_scores

REMLOGIC = """
Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\t22:00:00\tSLEEP-S0\t30\tROC-A2
R\t22:00:30\tSLEEP-REM\t30\tROC-A2
R\t22:01:00\tSLEEP-REM\t30\tROC-A2
S2\t22:01:30\tSLEEP-S2\t30\tROC-A2
"""


def test_isruc_rem_is_code_five_and_agrees_epoch_by_epoch():
    expert = parse_expert_scores("0\n5\n5\n2\n", time(0, 0, 0))
    assert expert.format == "isruc"
    assert expert.rem_blocks == [(30.0, 90.0)]
    compared = expert_comparison(expert, [(30.0, 60.0)], n_epochs=4, duration_sec=120.0)
    assert compared["rem_minutes"] == 1.0
    assert compared["agreed_minutes"] == 0.5
    assert compared["expert_only_minutes"] == 0.5
    assert compared["share_of_expert_also_detected"] == 50.0
    assert compared["share_of_detected_also_expert"] == 100.0


def test_stage_four_is_not_rem_and_a_bad_line_is_rejected():
    expert = parse_expert_scores("4\n5\n", time(0, 0, 0))
    assert expert.rem_blocks == [(30.0, 60.0)]
    with pytest.raises(ValueError, match="Line 2"):
        parse_expert_scores("0\nREM\n", time(0, 0, 0))


def test_remlogic_uses_the_recording_clock_and_ignores_unscored_time():
    expert = parse_expert_scores(REMLOGIC, time(22, 0, 0))
    assert expert.format == "remlogic"
    assert expert.rem_blocks == [(30.0, 90.0)]
    # The model calls REM in an epoch the expert never scored.
    compared = expert_comparison(expert, [(30.0, 90.0), (120.0, 150.0)], n_epochs=5, duration_sec=150.0)
    assert compared["agreed_minutes"] == 1.0
    assert compared["detected_only_minutes"] == 0.0
    assert compared["share_of_detected_also_expert"] == 100.0


def test_downloaded_isruc_score_has_the_expected_rem():
    path = Path("data/test_REMGuard/non_healthy/subgroupI_1/1_1.txt")
    if not path.is_file():
        pytest.skip("ISRUC score file is not on this machine")
    expert = parse_expert_scores(path.read_text(encoding="latin1"), time(0, 0, 0))
    assert expert.format == "isruc"
    compared = expert_comparison(expert, expert.rem_blocks, n_epochs=880, duration_sec=880 * 30)
    assert compared["rem_minutes"] == 59.0
    assert compared["agreed_minutes"] == 59.0
    assert compared["share_of_expert_also_detected"] == 100.0
