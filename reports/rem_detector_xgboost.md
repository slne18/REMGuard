# REM detector, all extracted nights

The label is expert REM versus expert wake and non-REM. Chin EMG is not an input. Diagnosis is not an input. Each recording id is treated as one participant, which is how the CAP files are named.

Cross-validation is five folds. Each fold holds out about 20 percent of the participants and trains on the other 80 percent. No participant appears in more than one test fold. Inside the training participants, a further 20 percent is used only to compare the parameter grid and to stop adding trees. The winning number of trees is then refit on all training participants. The test participants are not used for either choice.

Grid: learning rate 0.05 or 0.1, max depth 3 or 6, min child weight 1 or 5, subsample 0.8 or 1. Early stopping watches log loss, with at most 400 trees and a patience of 30.

Participants with labeled epochs: 108. Epochs: 107975. Expert REM share: 0.175.
Recordings with no usable epochs: 0.

| Metric | Mean | Standard deviation | Variance |
| --- | ---: | ---: | ---: |
| Classification error | 0.153 | 0.026 | 0.0007 |
| Log loss | 0.333 | 0.038 | 0.0014 |
| Brier score | 0.106 | 0.015 | 0.0002 |
| REM recall | 0.811 | 0.047 | 0.0022 |
| Specificity | 0.855 | 0.038 | 0.0015 |
| Precision | 0.548 | 0.060 | 0.0036 |
| F1 | 0.651 | 0.036 | 0.0013 |
| Balanced accuracy | 0.833 | 0.016 | 0.0003 |
| Majority-class error | 0.175 | 0.011 | 0.0001 |

Learning curves while the tree count is chosen: `rem_detector_learning_curves.png`.
Learning curves of the refit reported model: `rem_detector_refit_curves.png`.

## Folds

### Fold 1

- Test participants (22): ins4, ins7, ins8, ins9, n1, n11, n14, n3, n6, narco1, nfle11, nfle14, nfle16, nfle19, nfle26, nfle30, nfle5, rbd15, rbd16, rbd17, rbd5, rbd8.
- Train epochs 85406, test epochs 22569.
- Parameters: depth 6, learning rate 0.1, min child weight 1, subsample 0.8, trees 400.
- Error 0.146, log loss 0.326, Brier 0.102.
- REM recall 0.771, specificity 0.872, precision 0.574, F1 0.658, balanced accuracy 0.822.
- Confusion matrix: true non-REM 16085, false REM 2362, missed REM 942, true REM 3180.
- Majority-class error on this fold: 0.183.

### Fold 2

- Test participants (22): ins1, ins3, n13, n2, n5, n8, narco4, nfle13, nfle20, nfle40, nfle9, plm1, plm9, rbd1, rbd10, rbd11, rbd13, rbd14, rbd19, rbd2, rbd7, sdb2.
- Train epochs 86044, test epochs 21931.
- Parameters: depth 6, learning rate 0.05, min child weight 5, subsample 0.8, trees 289.
- Error 0.199, log loss 0.398, Brier 0.133.
- REM recall 0.861, specificity 0.789, precision 0.450, F1 0.591, balanced accuracy 0.825.
- Confusion matrix: true non-REM 14419, false REM 3853, missed REM 507, true REM 3152.
- Majority-class error on this fold: 0.167.

### Fold 3

- Test participants (22): brux2, ins2, ins5, n15, n16, n4, n7, narco2, nfle12, nfle2, nfle28, nfle29, nfle32, nfle39, nfle4, nfle6, nfle8, plm2, plm3, plm8, rbd20, rbd4.
- Train epochs 84697, test epochs 23278.
- Parameters: depth 6, learning rate 0.1, min child weight 5, subsample 0.8, trees 400.
- Error 0.135, log loss 0.299, Brier 0.095.
- REM recall 0.791, specificity 0.882, precision 0.608, F1 0.688, balanced accuracy 0.837.
- Confusion matrix: true non-REM 16671, false REM 2234, missed REM 912, true REM 3461.
- Majority-class error on this fold: 0.188.

### Fold 4

- Test participants (21): brux1, n10, n12, n9, nfle1, nfle17, nfle21, nfle22, nfle23, nfle25, nfle27, nfle33, nfle35, nfle36, nfle37, rbd12, rbd18, rbd6, rbd9, sdb1, sdb3.
- Train epochs 88714, test epochs 19261.
- Parameters: depth 6, learning rate 0.1, min child weight 1, subsample 0.8, trees 399.
- Error 0.142, log loss 0.317, Brier 0.100.
- REM recall 0.769, specificity 0.877, precision 0.571, F1 0.655, balanced accuracy 0.823.
- Confusion matrix: true non-REM 13909, false REM 1959, missed REM 785, true REM 2608.
- Majority-class error on this fold: 0.176.

### Fold 5

- Test participants (21): ins6, narco3, narco5, nfle10, nfle15, nfle18, nfle24, nfle3, nfle31, nfle34, nfle38, nfle7, plm10, plm4, plm5, plm6, plm7, rbd21, rbd22, rbd3, sdb4.
- Train epochs 87039, test epochs 20936.
- Parameters: depth 6, learning rate 0.1, min child weight 1, subsample 0.8, trees 398.
- Error 0.142, log loss 0.323, Brier 0.102.
- REM recall 0.863, specificity 0.857, precision 0.538, F1 0.663, balanced accuracy 0.860.
- Confusion matrix: true non-REM 15040, false REM 2508, missed REM 465, true REM 2923.
- Majority-class error on this fold: 0.162.

These figures describe this CAP collection under this split. They do not estimate performance on other recording systems, and they are not a Parkinson's score.
