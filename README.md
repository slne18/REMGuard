# REMGuard

REMGuard reads one night of sleep from an EDF file with brain (EEG), eye (EOG), and chin (EMG) signals. It finds REM, measures how long the chin stays active during that REM, and compares that amount with healthy nights.

## How REM is detected

The detector was trained on the [PhysioNet CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/). Experts had already marked each 30 seconds of those nights as REM or not. The model learns from the brain and eye signals only. The chin is not used to decide what is REM.

Each 30 seconds is turned into a few numbers: how much of the brain signal sits in the usual sleep frequency bands, how mixed that spectrum is, and how large the brain and eye signals are. An XGBoost model learns which of those patterns match expert REM. Whole nights stay together, so a night used for testing was not also used for training. The page uses one model fitted on all of those labeled nights.

## The chin score

Once REM is found, the chin is compared with its own quiet level on that same night. The quiet level is the typical chin size during the calmer half of the 30-second pieces the model did not call REM. A sample counts as active when it is larger than twice that level.

The score is the share of detected REM in which the chin is active. The same rule was run on the 15 healthy control nights in the CAP database. A new night is flagged when its share is higher than in 90 percent of those healthy nights.

## The page

The page shows that score as a large percentage, with a flag when the night is above the healthy line. Three pictures sit under it:

- Where this night sits among the 15 healthy nights. Each tick is one of those nights. The mark is the upload. The line is their 90th percentile.
- The night as a strip. Dark blocks are the 30-second pieces called REM. An optional expert score file can be added, and its REM is drawn in a second color, with the minutes the two agree on.
- A short stretch of the chin signal during one REM period, with the night's own threshold drawn across it.
