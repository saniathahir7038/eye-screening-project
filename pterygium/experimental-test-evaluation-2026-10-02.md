# Previous vs experimental model: fixed-threshold test counts (2026-10-02)

The model and threshold were fixed before this evaluation. The 61 SLID test
images were excluded from training and threshold tuning for this experimental
head, although prior project model iterations had already evaluated this same
test split. This is slit-lamp data, **not** an independent smartphone test.

| Outcome on the same 61 images | Previous APK model | New experimental APK model |
| --- | ---: | ---: |
| Pterygium annotation correctly flagged | 25/25 | 24/25 |
| No pterygium annotation correctly cleared | 33/36 | 36/36 |
| False positives | 3 | 0 |
| False negatives | 0 | 1 |
| Overall correct | 58/61 (95.1%) | 60/61 (98.4%) |

Three no-pterygium-annotation images (`6.png`, `12.png`, `48.png`) changed
from falsely flagged to correctly cleared. One pterygium-annotation image
(`1664.png`) changed from correctly flagged to missed. Thus the new model
improves the total by two correct decisions on this split, while losing one
pterygium detection. These are model classifications against dataset labels,
not clinical diagnoses. The models have different score distributions; raw
scores should not be compared as calibrated probabilities. The previous and
new decision thresholds were 0.1810068861 and 0.1684806645, respectively.

The Android TFLite file and Streamlit Keras model agreed on all 61 decisions;
maximum absolute score difference was 0.00000340. The original photos all
passed the app's technical quality checks, no auto-framing occurred, and the
Python implementation of the app preprocessing path gave the same 61 decisions
as the saved processed files. This was not an on-device replay of all 61 files.

On the separate 33-image web-collected external partition, the previous model
correctly flagged **14/14 pterygium-labeled** but cleared **0/19 normal-labeled**
images (19 false positives). The new model correctly flagged **13/14** and
cleared **11/19** (1 missed positive, 8 false positives). This partition was
used during model development and inspected during threshold iteration, so it
is not a pristine final test.
Furthermore, the app quality gate would reject 21 of those 33 low-resolution
images; among the 12 it would screen, it would correctly flag 5/5 labeled
pterygium and correctly clear 4/7 labeled normal images. These counts are too
small and modality-mismatched to estimate smartphone performance.

The exact per-image SLID scores and checks are in
`artifacts/domain_adapt_v1/slid_test_evaluation.json` (local generated file).
Reproduce with `python evaluate_experimental_test.py`. Do not tune the model or
threshold against these results and then call this same split an unseen test.
