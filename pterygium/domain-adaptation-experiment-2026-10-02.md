# Exploratory domain-adaptation experiment (2026-10-02)

This is an engineering experiment, **not** clinical validation or a deployable
accuracy claim. At the time of this experiment, the accepted Streamlit model
and Android TFLite asset were not changed. The user later requested an
experimental deployment; see [experimental-test-evaluation-2026-10-02.md](experimental-test-evaluation-2026-10-02.md)
for fixed-threshold test counts and current limitations.

## Method

- Started from the accepted frozen MobileNetV2 feature extractor. Replaced only
  its logistic output head; did not retrain the image backbone.
- Combined original SLID training images with a separate web-collected external
  eye-image dataset. After exact-pixel deduplication, 168 external images
  remained, arranged into 167 near-duplicate groups.
- Split external groups by label into 101 training, 34 validation, and 33 test
  images. No grouped duplicate crossed those splits.
- Searched 9 regularization/external-weight combinations. Selected the decision
  threshold using external and SLID validation data, requiring at least 90%
  sensitivity on both before maximizing the lower specificity.
- Tested the selected candidate on the reserved 33-image external test partition.

| Set | Model | Detected positives | Missed positives | Correct normals | False positives |
| --- | --- | ---: | ---: | ---: | ---: |
| External validation | Candidate | 14/15 | 1 | 14/19 | 5 |
| SLID validation | Candidate | 23/24 | 1 | 37/37 | 0 |
| External test | Candidate | 13/14 | 1 | 11/19 | 8 |
| External test | Current accepted model | 14/14 | 0 | 0/19 | 19 |
| SLID validation | Current accepted model | 23/24 | 1 | 37/37 | 0 |

The candidate threshold is 0.1684806645, **specific to the candidate's new
output head**, not a recalibration of the old 0.181 threshold. The integrated
Keras head matched its fitted classifier on the test images within 3e-7.

## Interpretation and decision

The candidate reduces false positives on this small external set, while adding
one false negative in the external test. An earlier threshold search on this
same partition was inspected before the sensitivity-first rule was run, so the
external test is now part of an **exploratory** iteration, not an untouched
final test. Most images in this external set also fail the app's minimum
resolution gate, so these raw-model counts are **not** APK end-to-end metrics.
The dataset consists of web-collected images, not a matched, clinician-labeled
set from the target smartphone capture workflow.

These numbers alone did not warrant replacing the model or APK; the later
0.3.0-experimental replacement was made at the user's explicit request. The next
decision-quality test needs a new, independently labeled smartphone-eye dataset
with both normal and pterygium cases, captured across people, lighting, and
devices. Keep each person's images in only one data split. Prespecify a
minimum acceptable sensitivity and false-positive rate before evaluating the
final untouched test set; also measure quality-gate rejection and per-device
performance.

Reproduce the exploratory experiment with `domain_adapt_experiment.py`. The
candidate and machine-readable results are stored outside the repository at
`%LOCALAPPDATA%\Temp\PterygiumDomainAdaptSensitivity\`.
