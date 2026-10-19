# RBD nights the cutoff missed

On expert REM, 4 RBD nights have a rank of 53 or lower. A rank of 87 is just under the cutoff of 90. These nights are the ones that sit in the healthy range.

The typical level is the median chin size during expert REM divided by that night's own quiet-sleep level. The threshold is twice quiet sleep, so a typical level below 2 means most of the epoch stays under the line. Usable EMG is the share of the REM interval that is not a flat dropout. Microvolts are not compared across nights.

| Night | Rank | REM minutes | Elevated time | Typical level | Usable EMG |
| --- | ---: | ---: | ---: | ---: | ---: |
| rbd10 | 0 | 40 | 4% | 0.4× | 100% |
| rbd11 | 0 | 40 | 4% | 0.4× | 100% |
| rbd17 | 27 | 70 | 5% | 0.6× | 100% |
| rbd6 | 53 | 64 | 6% | 0.4× | 100% |

What each night shows:

- `rbd10`: the typical sample is 0.4× its own quiet sleep, so most of the epoch never reaches the 2× threshold.
- `rbd11`: the typical sample is 0.4× its own quiet sleep, so most of the epoch never reaches the 2× threshold.
- `rbd17`: the typical sample is 0.6× its own quiet sleep, so most of the epoch never reaches the 2× threshold.
- `rbd6`: the typical sample is 0.4× its own quiet sleep, so most of the epoch never reaches the 2× threshold.

4 more RBD nights sit between the healthy range and the rank-87 cluster: `rbd1` (73), `rbd4` (73), `rbd5` (73), `rbd9` (73).

## Epochs both the expert and the model call REM

Detected REM by itself includes epochs the expert did not call REM. This section keeps only the overlap. A night with less than 30 seconds of overlap is left out of the rank.

| Night | Expert rank | Overlap minutes | Overlap elevated time | Overlap rank |
| --- | ---: | ---: | ---: | ---: |
| rbd10 | 0 | 34 | 3% | 0 |
| rbd11 | 0 | 34 | 3% | 7 |
| rbd17 | 27 | 68 | 4% | 33 |
| rbd6 | 53 | 34 | 4% | 33 |

On the overlap, the median rank is 50 for healthy nights and 87 for RBD nights. The cutoff of 90 flags 5/22 RBD nights and 2/15 healthy nights. The low nights stay at rank 53 or lower on the overlap, so the quiet chin is in the REM both sources agree on.

This is a description of these nights. It is not a clinical RBD score and it is not a Parkinson's score.
