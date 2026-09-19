# Pterygium module

Initial SLID dataset preparation for the Eye Screening Project. Work proceeds one stage at a time, with user approval before the next stage. All module code, data and outputs stay in this folder.

## Step 1: dataset setup and audit

Use Python 3.11+ and Git LFS. From this folder in PowerShell:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements-audit.txt
& .venv/Scripts/python.exe download.py
& .venv/Scripts/python.exe audit.py
& .venv/Scripts/python.exe -m unittest discover -s tests -v
```

The downloader pins the official SLID revision, verifies the archive SHA-256, extracts the images and records source provenance. The audit checks image decoding, missing/ambiguous filenames, file sizes, annotation row counts and IDs, pterygium rectangle bounds, and exact duplicates by decoded RGB pixels.

If Git LFS stalls, download the archive from the official repository's download link and run `download.py --archive data/SLID-images.zip`. The same pinned SHA-256 check applies. The downloader rechecks extracted file sizes and CRCs instead of trusting an old completion marker.

Outputs (local and excluded from Git):

- `data/provenance.json`: source revision and checksums.
- `artifacts/audit/image_manifest.csv`: one row per annotated image, including exclusions and issues.
- `artifacts/audit/binary_labels.csv`: eligible image-level labels, with any integrity/annotation problems withheld.
- `artifacts/audit/audit_summary.json`: counts and issues in machine-readable form.
- `artifacts/audit/audit-report.md`: human-readable results and pending work.

CSV image paths are relative to `data/images`, including any directory inside the archive. The audit can also accept `--annotations`, `--images` and `--output`. An audit with issues returns a nonzero exit status and still writes its reports. Malformed annotation input fails before generating new reports; previously generated reports are not refreshed by a failed parse.

## Label policy

- **1:** at least one annotation contains `Pterygium`, including images with additional diseases.
- **0:** every annotation has an empty lesion string. The CSV has no explicit `Normal` field; these are dataset-derived normal candidates.
- **Excluded:** other-disease-only images, or images with audit issues.

The specification expects 163 positives and 245 normal candidates. Counts are calculated from the source, never forced to match expectations. Multiple lesion annotations in one image produce only one classification row. The 92 pterygium-only images are a subset of the positives.

## Initial audit findings — 2026-09-14

The verified archive contains 2,617 eye images plus 2,617 macOS AppleDouble metadata files. Metadata files are recognized by both their location/name and signature, and excluded from the image inventory. There are 10,204 annotation data rows (10,205 CSV lines including the header).

The binary subset has 408 readable images: 163 pterygium positives and 245 normal candidates. No missing/corrupt images, file-size mismatches or invalid pterygium rectangle bounds were found. Eight exact duplicate pairs exist across the full dataset; one pair (`3.png`, `116.png`) is in the normal subset. These two images must remain in the same future split or be deduplicated during Step 2.

Two source images, `382.png` and `383.png`, declare four annotations each but contain three rows each. Both are intraocular-lens images already excluded from this binary experiment. The full audit intentionally returns `needs_review` (exit code 1) to preserve these source findings; the selected subset's integrity status is `passed`. The source annotations are retained unchanged.

Eleven automated tests cover the label rules, corrupt/missing/ambiguous images, incomplete annotations, invalid rectangles, duplicates, metadata recognition and archive validation. The complete generated report remains under `artifacts/audit/`.

## Step 2: preprocessing, duplicate review and split

Install the additional dependency and prepare the locked manifests:

```powershell
& .venv/Scripts/python.exe -m pip install -r requirements-phase2.txt
& .venv/Scripts/python.exe prepare_dataset.py
& .venv/Scripts/python.exe -m unittest discover -s tests -v
```

The process preserves the complete ocular field. It applies a neutral mask to the upper-left 20% × 20% region of every image, including the four images where the source information box cannot be reliably detected. This prevents the box from becoming a class shortcut. The full image is then converted to RGB and resized to 224 × 224 pixels. All 164 lesion boxes remain valid after coordinate mapping; none is clipped.

Perceptual hashes nominate pairs for review but never group them automatically. The reviewed decision file records 22 visually repeated-capture pairs and two rejected lookalike pairs; the exact duplicate `3.png`/`116.png` is also grouped. The resulting 23 multi-image groups are kept within a single split. Because SLID does not publish patient identifiers, visual grouping reduces leakage but cannot prove patient-independent separation.

The deterministic seed `20260916` produces:

| Split | Pterygium | Normal candidate | Total |
|---|---:|---:|---:|
| Training | 114 | 172 | 286 |
| Validation | 24 | 37 | 61 |
| Testing | 25 | 36 | 61 |

Outputs under `artifacts/phase2/` include processed images, locked split manifests, the duplicate candidate CSV, a machine-readable summary, a report, and crop/overlay/duplicate review sheets. Phase 2 passes with zero unresolved duplicate candidates and zero group-split violations. Seventeen automated tests now pass across the audit and preparation stages.

## Step 3: MobileNetV2 training

Install the full module requirements and run training:

```powershell
& .venv/Scripts/python.exe -m pip install -r requirements.txt
& .venv/Scripts/python.exe train.py --smoke-test
& .venv/Scripts/python.exe train.py
```

Training uses ImageNet-pretrained MobileNetV2 with global average pooling, 30% dropout and a sigmoid classification output. The first stage freezes the backbone. The second stage opens the final 30-layer backbone region; keeping BatchNormalization frozen leaves 19 backbone layers trainable. Balanced class weights compensate for the 172/114 training-class counts.

Only training images receive horizontal flips, small rotations, translations, zoom, brightness changes and contrast changes. Early stopping and model selection use the 61-image validation split. The 61-image test split is loaded only for integrity checks and remains unevaluated until Step 4.

The deterministic seed `20260918` produced these development results:

| Result | Value |
|---|---:|
| Selected checkpoint | Fine-tuned MobileNetV2 |
| Validation ROC-AUC | 0.9989 |
| Validation threshold | 0.055293 |
| Validation sensitivity | 1.0000 |
| Validation specificity | 0.9730 |
| Validation balanced accuracy | 0.9865 |
| Validation confusion | TP 24, TN 36, FP 1, FN 0 |

The cutoff maximizes validation balanced accuracy; ties prefer sensitivity, then specificity. Its low numeric value shows that the sigmoid score is not a calibrated probability. Treat it as a model score and keep the cutoff locked for the Phase 4 test evaluation.

Outputs under `artifacts/phase3/` include the selected Keras model, a TensorFlow SavedModel export, both stage checkpoints, the full configuration, training history, validation predictions, threshold curve, plots and the Phase 3 report. Reloading the selected model reproduced all validation scores exactly. Twenty-one automated tests now pass.

## Step 4: locked test evaluation and Grad-CAM review

Run a validation-only Grad-CAM check before the locked evaluation:

```powershell
& .venv/Scripts/python.exe evaluate.py --smoke-test
& .venv/Scripts/python.exe evaluate.py
```

The first successful test run writes `artifacts/phase4/evaluation_lock.json` with the model, threshold and test-data fingerprints. Later commands reuse the locked results and do not run test inference again.

The fixed Phase 3 model and threshold produced:

| Test result | Value |
|---|---:|
| Images | 61 |
| Accuracy | 0.9016 |
| Sensitivity | 1.0000 |
| Specificity | 0.8333 |
| Precision | 0.8065 |
| F1-score | 0.8929 |
| Balanced accuracy | 0.9167 |
| ROC-AUC | 1.0000 |
| Confusion | TP 25, TN 30, FP 6, FN 0 |

The model ranked all positive test images above all normal candidates, producing ROC-AUC 1.0. The validation-selected threshold remained locked and caused six false positives; it was not changed using test results.

Grad-CAM overlays were generated for all 61 test images. The maximum-attention point fell inside an annotation box for 20 of 25 positive images, and lesion boxes received 1.55 times the attention expected from their area on average. Visual review found that the six false positives emphasized eyelids, borders or peripheral ocular tissue. A few images also showed attention on the standardized upper-left corner. This partially passes lesion localization but identifies a shortcut risk.

Outputs under `artifacts/phase4/` include the locked prediction CSV, metrics summary, evaluation report, ROC curve, confusion matrix, score distribution, all Grad-CAM overlays and focused review sheets.

## Phase 4A corrective iteration

Run the reproducible corrective training and validation review with:

```powershell
& .venv/Scripts/python.exe corrective_train.py
```

The correction starts from the Phase 3 model, freezes the visual feature extractor and retrains only the classifier head. Every training image is presented unchanged and with a neutral 45 x 45 patch in each of the three corners not standardized during Phase 2. The locked test set is never passed to this pipeline.

The selected checkpoint passed all preset validation gates. ROC-AUC was 0.9977, sensitivity was 0.9583, specificity was 1.0000 and accuracy was 0.9836 at the newly selected validation threshold of 0.181007. Mean positive box-attention lift improved from 2.151 to 2.165, the maximum-attention point remained inside a lesion box for 18 of 24 positive validation images, and mean normal-image corner attention fell from 0.0262 to 0.0255.

Outputs under `artifacts/phase4a/` include the corrected Keras model, SavedModel export, validation predictions, threshold curve, Grad-CAM overlays, comparison plot, review sheets and machine-readable summary. Twenty-eight automated tests pass. A new independent test dataset is required for an unbiased estimate of corrected-model performance.

## Approval stages

1. Dataset setup and audit ? complete.
2. Near-duplicate review, preprocessing and reproducible group splits ? complete.
3. MobileNetV2 training and validation-based threshold selection ? complete.
4. Locked test evaluation and Grad-CAM review ? complete with review findings.
5. Corrective train/validation iteration for border and corner reliance - complete.
6. Python upload interface - awaiting approval.
7. TensorFlow Lite conversion and Android integration.

Public patient identifiers are unavailable, so patient-independent separation cannot be proven. Accepted duplicate and repeated-capture groups remain inside one split. Bounding-box checks validate coordinates, not clinical correctness. SLID uses slit-lamp images; smartphone performance requires separate data and validation. Model outputs are screening results, not confirmed diagnoses.

## Sources

- [Module specification](https://app.notion.com/p/3cbc16ecc9ae80fd8d8ceb713104598e)
- [Official SLID repository](https://github.com/xumingyu-hub/SLID)
- [SLID dataset paper](https://doi.org/10.3389/fdgth.2025.1716501)

Keep downloaded data and derivatives local. The repository does not include a standalone dataset licence; confirm redistribution terms with the authors before sharing data.
