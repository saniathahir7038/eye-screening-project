# Pterygium development plan

Captured from the chat supplied by Kaushiik on 2026-09-14. Requirements source: [Pterygium Module](https://app.notion.com/p/Pterygium-Module-3cbc16ecc9ae80fd8d8ceb713104598e).

Work exclusively in the existing pterygium folder. The current user instruction is to develop one step at a time with approval before each subsequent step. Dataset setup and audit are the current stage; preprocessing, splitting and model training await the next approval. Earlier planning notes do not authorize completing all stages automatically.

## Scope

Build a binary academic screening prototype with MobileNetV2 transfer learning. Outputs: No visible pterygium pattern; Suspected pterygium; a model score; Grad-CAM; and a retake message for images that fail quality checks. Suspected results recommend professional eye examination. The model does not establish a diagnosis or grade severity.

## Development order

1. Download the official SLID repository and its Git LFS image ZIP.
2. Extract safely and inspect images.
3. Audit Annotations.csv before deriving labels.
4. Identify any-pterygium positives and explicitly normal negatives; exclude other-disease-only images from the first experiment.
5. Produce one row per image with provenance and exclusion reasons.
6. Check missing/corrupt files, exact duplicates and perceptual similarity.
7. Crop the useful image field and remove black information regions using the same preprocessing at training and inference.
8. Make reproducible approximately 70/15/15 stratified group splits. Similarity groups are not verified patient IDs.
9. Train ImageNet MobileNetV2: frozen backbone followed by optional low-rate fine-tuning; augmentation only in training.
10. Evaluate sensitivity, specificity, precision, F1, ROC-AUC and confusion matrix. Select threshold on validation only.
11. Inspect Grad-CAM and annotation overlays for shortcuts and cropping errors.
12. Build and test a local Python image-upload interface.
13. Export TensorFlow Lite and verify numerical agreement before mobile integration.
14. Fine-tune and independently test on smartphone images when access is granted.

## Expected source facts to audit

The supplied chat reports a roughly 687 MB Original_Slit-lamp_Images.zip and 10,205 annotation rows. The Notion page expects 2,617 source images, 163 pterygium-positive images and 245 normal images. These are source expectations, not implementation results; actual counts belong in the audit report.

## Limitations and distribution

SLID consists of slit-lamp images. Its test results cannot be presented as validated smartphone screening accuracy. Normal-only controls do not establish specificity against other ocular conditions. Confidence is an uncalibrated model score unless calibration is separately demonstrated. A heuristic quality gate cannot establish that the full eye is visible or detect every unsuitable image.

No standalone licence was visible in the official repository at inspection. Public download and a paper citation do not establish blanket permission for every use. Keep data and generated image derivatives local and excluded from Git; confirm applicable terms with the authors before redistribution. No author-contact messages are sent by this module.

## Sources

- https://github.com/xumingyu-hub/SLID
- https://doi.org/10.3389/fdgth.2025.1716501
- notion-source.md contains the retrieved requirements snapshot.
