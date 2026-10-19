# REM detector, random forest

The label is expert REM versus expert wake and non-REM. Chin EMG is not an input. Diagnosis is not an input. Each recording id is treated as one participant. The five test groups are the same nights as in the XGBoost run.

Each fold holds out about 20 percent of the nights. Inside the training nights, a further 20 percent is the validation set. The forest is grown to 400 trees. Validation log loss is read every 10 trees, and growth stops after 30 trees with no improvement. The winning depth, leaf size, and tree count are refit on all training nights. The test nights are not used for either choice.

Grid: max depth 8 or 16, minimum samples per leaf 1 or 5. Each tree considers sqrt(10) features. Class imbalance uses balanced class weights. A median imputer and a standard scaler are fit on the nights being trained, so missing EOG stays inside the training fold.

Participants with labeled epochs: 108. Epochs: 107975. Expert REM share: 0.175.
Recordings with no usable epochs: 0.

| Metric | Mean | Standard deviation | Variance |
| --- | ---: | ---: | ---: |
| Classification error | 0.145 | 0.024 | 0.0006 |
| Log loss | 0.316 | 0.040 | 0.0016 |
| Brier score | 0.100 | 0.015 | 0.0002 |
| REM recall | 0.705 | 0.075 | 0.0057 |
| Specificity | 0.888 | 0.039 | 0.0015 |
| Precision | 0.579 | 0.076 | 0.0057 |
| F1 | 0.631 | 0.040 | 0.0016 |
| Balanced accuracy | 0.796 | 0.028 | 0.0008 |
| Majority-class error | 0.175 | 0.011 | 0.0001 |

Learning curves while the tree count is chosen: `rem_detector_random_forest_learning_curves.png`.
Learning curves of the refit reported model: `rem_detector_random_forest_refit_curves.png`.

## Folds

### Fold 1

- Test participants (22): ins4, ins7, ins8, ins9, n1, n11, n14, n3, n6, narco1, nfle11, nfle14, nfle16, nfle19, nfle26, nfle30, nfle5, rbd15, rbd16, rbd17, rbd5, rbd8.
- Train epochs 85406, test epochs 22569.
- Parameters: depth 16, min samples per leaf 1, trees 130.
- At that tree count, before the refit: train log loss 0.152, validation 0.258, test 0.332. Train error 0.049, validation 0.113, test 0.147.
- Refit train log loss 0.160, test log loss 0.334. Refit train error 0.056, test error 0.150.
- Test error 0.150, log loss 0.334, Brier 0.102.
- REM recall 0.636, specificity 0.898, precision 0.582, F1 0.608, balanced accuracy 0.767.
- Confusion matrix: true non-REM 16566, false REM 1881, missed REM 1501, true REM 2621.
- Majority-class error on this fold: 0.183.

### Fold 2

- Test participants (22): ins1, ins3, n13, n2, n5, n8, narco4, nfle13, nfle20, nfle40, nfle9, plm1, plm9, rbd1, rbd10, rbd11, rbd13, rbd14, rbd19, rbd2, rbd7, sdb2.
- Train epochs 86044, test epochs 21931.
- Parameters: depth 16, min samples per leaf 5, trees 210.
- At that tree count, before the refit: train log loss 0.191, validation 0.358, test 0.386. Train error 0.069, validation 0.166, test 0.185.
- Refit train log loss 0.195, test log loss 0.375. Refit train error 0.073, test error 0.182.
- Test error 0.182, log loss 0.375, Brier 0.123.
- REM recall 0.791, specificity 0.824, precision 0.473, F1 0.592, balanced accuracy 0.807.
- Confusion matrix: true non-REM 15050, false REM 3222, missed REM 766, true REM 2893.
- Majority-class error on this fold: 0.167.

### Fold 3

- Test participants (22): brux2, ins2, ins5, n15, n16, n4, n7, narco2, nfle12, nfle2, nfle28, nfle29, nfle32, nfle39, nfle4, nfle6, nfle8, plm2, plm3, plm8, rbd20, rbd4.
- Train epochs 84697, test epochs 23278.
- Parameters: depth 16, min samples per leaf 1, trees 200.
- At that tree count, before the refit: train log loss 0.157, validation 0.344, test 0.276. Train error 0.054, validation 0.163, test 0.121.
- Refit train log loss 0.170, test log loss 0.272. Refit train error 0.062, test error 0.118.
- Test error 0.118, log loss 0.272, Brier 0.085.
- REM recall 0.678, specificity 0.929, precision 0.687, F1 0.683, balanced accuracy 0.803.
- Confusion matrix: true non-REM 17556, false REM 1349, missed REM 1408, true REM 2965.
- Majority-class error on this fold: 0.188.

### Fold 4

- Test participants (21): brux1, n10, n12, n9, nfle1, nfle17, nfle21, nfle22, nfle23, nfle25, nfle27, nfle33, nfle35, nfle36, nfle37, rbd12, rbd18, rbd6, rbd9, sdb1, sdb3.
- Train epochs 88714, test epochs 19261.
- Parameters: depth 16, min samples per leaf 1, trees 120.
- At that tree count, before the refit: train log loss 0.157, validation 0.322, test 0.313. Train error 0.055, validation 0.142, test 0.146.
- Refit train log loss 0.162, test log loss 0.308. Refit train error 0.055, test error 0.146.
- Test error 0.146, log loss 0.308, Brier 0.098.
- REM recall 0.640, specificity 0.900, precision 0.578, F1 0.607, balanced accuracy 0.770.
- Confusion matrix: true non-REM 14279, false REM 1589, missed REM 1221, true REM 2172.
- Majority-class error on this fold: 0.176.

### Fold 5

- Test participants (21): ins6, narco3, narco5, nfle10, nfle15, nfle18, nfle24, nfle3, nfle31, nfle34, nfle38, nfle7, plm10, plm4, plm5, plm6, plm7, rbd21, rbd22, rbd3, sdb4.
- Train epochs 87039, test epochs 20936.
- Parameters: depth 16, min samples per leaf 1, trees 130.
- At that tree count, before the refit: train log loss 0.146, validation 0.326, test 0.287. Train error 0.049, validation 0.166, test 0.125.
- Refit train log loss 0.166, test log loss 0.289. Refit train error 0.059, test error 0.128.
- Test error 0.128, log loss 0.289, Brier 0.091.
- REM recall 0.780, specificity 0.889, precision 0.576, F1 0.663, balanced accuracy 0.835.
- Confusion matrix: true non-REM 15607, false REM 1941, missed REM 746, true REM 2642.
- Majority-class error on this fold: 0.162.

These figures describe this CAP collection under this split. They do not estimate performance on other recording systems, and they are not a Parkinson's score.
