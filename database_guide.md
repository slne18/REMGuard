# CAP Sleep Database --- Quick File Guide

Official dataset: [PhysioNet CAP Sleep
Database](https://physionet.org/content/capslpdb/1.0.0/)

## 1. Main files for each participant

Most recordings follow this naming pattern:

`<group><number>.<extension>`

Example: `rbd3.edf`, `rbd3.txt`, `rbd3.edf.st` belong to the same
recording.

  -----------------------------------------------------------------------
  Extension / file        What it contains        Use in this project
  ----------------------- ----------------------- -----------------------
  `.edf`                  Raw polysomnography     **Main signal file.**
                          (PSG) signals: EEG,     Load with MNE-Python
                          EOG, chin EMG, leg EMG, and select EEG/EOG for
                          and physiological       REM detection; use chin
                          channels such as        EMG later for
                          breathing, oxygen       muscle-activity
                          saturation and ECG      analysis.
                          (channels vary by       
                          recording).             

  `.txt`                  Expert scoring exported **Human reference
                          in REMlogic report      labels.** Use to
                          format. Includes sleep  identify REM (`R`),
                          stage, body position    wake (`W`), NREM stages
                          when available, time,   (`S1`--`S4`) and
                          event, duration and     movement time (`MT`).
                          signal location.        

  `.edf.st`               PhysioBank-compatible   Alternative
                          annotations containing  machine-readable source
                          equivalent scoring      of expert annotations;
                          information in          `.txt` may be easier to
                          annotation format.      inspect initially.

  `gender-age.xlsx`       Participant age and     Optional demographic
                          gender information.     metadata; handle
                                                  carefully and avoid
                                                  using it as a
                                                  predictive input unless
                                                  justified.
  -----------------------------------------------------------------------

Not every recording should be assumed to contain identical channels or
metadata. Inspect each EDF header before processing.

## 2. Other dataset-level files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `RECORDS`                           List of record names in the
                                      database. Useful for building a
                                      file inventory.

  `ANNOTATORS`                        Information about the annotators.

  `ScoringReader.m`                   MATLAB helper for reading `.txt`
                                      scoring files and extracting
                                      sleep-stage/CAP annotations.

  `CAP.m`                             MATLAB function for computing
                                      CAP-related measures from the
                                      scoring data. Not required for a
                                      basic REM detector.

  `CAP-sequence.jpg`                  Illustration of a Cyclic
                                      Alternating Pattern (CAP) sequence.

  `SHA256SUMS.txt`                    Checksums for checking file
                                      integrity.
  -----------------------------------------------------------------------

## 3. Participant-name prefixes

  Prefix      Group                               Number of recordings
  ----------- --------------------------------- ----------------------
  `n`         Healthy controls                                      16
  `rbd`       REM sleep behavior disorder                           22
  `nfle`      Nocturnal frontal lobe epilepsy                       40
  `plm`       Periodic limb movements                               10
  `ins`       Insomnia                                               9
  `narco`     Narcolepsy                                             5
  `sdb`       Sleep-disordered breathing                             4
  `brux`      Bruxism                                                2
  **Total**                                                    **108**

These are the database's diagnostic group labels; they are not
sleep-stage labels.

## 4. Sleep-stage labels in the `.txt` scoring files

-   `W` --- Wake
-   `S1`, `S2`, `S3`, `S4` --- Non-REM sleep stages under the
    Rechtschaffen & Kales scoring system
-   `R` --- REM sleep
-   `MT` --- Movement time

The dataset's MATLAB helper describes a numeric hypnogram encoding of
`0` for wake, `1`--`4` for sleep stages, and `5` for REM. If you parse
the text annotations yourself, check the event names and timing
conventions rather than assuming all files encode labels identically.

Source: [Official PhysioNet documentation and file
listing](https://physionet.org/content/capslpdb/1.0.0/).
