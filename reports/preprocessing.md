# Preprocessing

All 108 CAP nights were preprocessed. Each raw EDF was filtered, reduced to the signals this project uses, written to `data/preprocess/<recording>/`, and then deleted. The saved set is 17 GB. No night failed.

Each folder holds the expert REMlogic score file and three float32 FIF files when the recording had those channels: filtered EEG, filtered EOG, and filtered chin EMG. Limb EMG, ECG, oxygen, and respiratory channels were not stored.

## What was filtered

Filters are zero-phase FIR, so they use samples on both sides of each time point. That is appropriate for a finished recording and is not a real-time filter. Nothing was z-scored. Values stay in volts inside the FIF files.

EEG was kept when the EDF prefilter was already a sleep band: 0.3–30 Hz on 92 files, 0.5–30 Hz on 2, and 0.1–30 Hz on 1. Twelve nights had a wider or blank prefilter and were band-passed at 0.3–35 Hz: `n4`, `n6`–`n9`, `n12`–`n16`, `nfle25`, and `nfle33`. `nfle20` and `rbd21` mixed the two: some EEG channels already matched the analysis band and were kept, and the others were filtered. `nfle20` is the only night with EEG stored from two native rates (128 Hz and 256 Hz).

EOG was band-passed at 0.3–10 Hz before it was saved. Chin EMG was high-passed at 10 Hz. The lowpass was 100 Hz when the sampling rate could support it, and lower otherwise (54 Hz on the 128 Hz chin channels, 40 Hz on one slower channel, 90 Hz on five). A channel whose acquisition filter already matched that band was not filtered again. On 83 nights the chin recording already included a 10–100 Hz band and a 50 Hz notch, and that acquisition filter was kept. No new 50 Hz notch was added.

EEG stored with a highpass of 5 Hz or more was left out of the channels used for sleep staging. Flat channels are those with a standard deviation below 1 µV. They were flagged, not deleted.

## What was stored

After filtering, EEG and EOG were saved at 100 Hz. Chin EMG was saved at 256 Hz when the native rate was higher, and at the native rate otherwise (128 Hz on 9 nights, 200 Hz on 5, 100 Hz on 1). Native EEG rates were 512 Hz (81 nights), 256 Hz (9), 200 Hz (3), 128 Hz (12), and 100 Hz (4). The lower storage rate still covers the filtered bands, which is why the raw EDFs did not have to be kept.

`n16` has no EOG and no chin EMG. `nfle25` and `nfle33` have no chin EMG. Every night has EEG.

Expert scores are not inside the EDFs. All 108 annotation files were copied next to the signals. `n13` and `n14` have an unusable EDF record count in the header; duration was taken from the loaded signal instead. The calendar date in the score file disagrees with the EDF header for `n12` and `nfle27`. Alignment uses the clock time, not that calendar date.
