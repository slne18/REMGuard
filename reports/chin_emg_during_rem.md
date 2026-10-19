# Chin activity during REM

Chin EMG was not used to detect REM. For each night, the REM epochs are the ones the XGBoost model called when that night was held out of training. The comparison is the same measurement inside the expert's REM epochs.

A sample counts as elevated when it is above twice that night's own quiet-sleep level. Quiet sleep is expert S2, or S3, or S4 if S2 is shorter than 30 seconds. Wake is not the baseline. Because each night has its own level, the percentages below can be compared across nights. The raw microvolt values cannot.

The table is the median across nights. Activity percentages use nights with at least 30 seconds of that kind of REM. REM minutes include nights where none was found.

| Group | Nights | REM minutes, expert | REM minutes, detected | Elevated time, expert | Elevated time, detected | Bursts per hour, expert | Bursts per hour, detected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Healthy | 15 | 108 | 122 | 6% | 15% | 70 | 102 |
| RBD | 22 | 71 | 126 | 19% | 26% | 56 | 54 |
| NFLE | 38 | 94 | 117 | 6% | 14% | 42 | 66 |
| PLM | 10 | 73 | 95 | 10% | 21% | 115 | 269 |
| Insomnia | 9 | 66 | 96 | 5% | 13% | 47 | 175 |
| Narcolepsy | 5 | 111 | 170 | 10% | 19% | 80 | 213 |
| SDB | 4 | 25 | 98 | 12% | 30% | 99 | 568 |
| Bruxism | 2 | 68 | 73 | 5% | 13% | 42 | 727 |

On expert REM, the median chin elevated time is 6% in healthy nights and 19% in RBD nights. Tonic-like epochs, where at least half of the epoch is above the threshold, are 0% and 3%. Using the detector's REM instead, elevated time is 15% in healthy nights and 26% in RBD nights. Detected REM lasts longer than expert REM, and the higher percentages come from epochs the expert did not call REM, where the chin is more active.

No chin measurement for: n16 (no chin EMG), nfle25 (no chin EMG), nfle33 (no chin EMG).

Elevated time and burst rate describe how much the chin stays active during REM in this collection. They are not a clinical RBD score and they are not a Parkinson's score.
