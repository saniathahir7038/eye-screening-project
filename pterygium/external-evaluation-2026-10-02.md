# External pterygium evaluation — 2026-10-02

## Source and protocol

The external images came from TaeKeun Yoo's [Mendeley Data release, version 2](https://data.mendeley.com/datasets/t75wjsw6bw/2), DOI `10.17632/t75wjsw6bw.2`, licensed CC BY-NC 3.0. The release describes web-collected ocular-surface photographs classified by two board-certified ophthalmologists. These are not confirmed to be camera-native smartphone photographs. The archive contains 75 images in `pterygium/` and 96 in `normal/`; the [associated paper](https://pubmed.ncbi.nlm.nih.gov/33862570/) describes 75 and 94, respectively, so the release count differs from the publication.

The images were downloaded to a temporary directory outside the repository. `external_eval.py` ran the current image decoder, conservative auto-framing, technical quality gate, Phase 2 mask/resize, and locked threshold `0.18100688606500626` against both the accepted Keras model and the embedded Android float16 TFLite model. No weights, preprocessing thresholds, or decision threshold were fitted to this dataset. Folder names supplied the labels; labels were not independently re-adjudicated here.

The 408-image SLID manifest used for development had no exact RGB-pixel match with these external files. The nearest processed-image pHash distance was 10; none fell within the prior pHash review cutoff of 8. This cannot prove patient or source independence, but the dataset was not part of training or threshold selection in this repository.

## Results

| Outcome with the Android TFLite model | Pterygium-labeled | Normal-labeled |
| --- | ---: | ---: |
| Rejected by technical quality gate | 50 | 79 |
| Screened, flagged suspected | 25 | 17 |
| Screened, no visible pattern | 0 | 0 |

The app screened only **42/171 images (24.6%)**. All 129 rejections failed the minimum 224 × 224 resolution check; 20 also failed sharpness and 9 also failed aspect ratio. One image was auto-framed; no decision became inconclusive. Among the 42 screened files, the model flagged **all 42 positive**, including **17/17 normal-labeled photos**. Raw-file accuracy was 25/42 (59.5%), sensitivity 25/25, and specificity 0/17. Three exact duplicate pairs occur within the external pterygium folder, two of which were screened. After deduplicating screened files, the counts are **23/23 positive-labeled flagged and 0/17 normal-labeled cleared** (23/40 correct, 57.5%).

The Keras and Android TFLite models agreed on every one of the 42 decisions. Their largest absolute score difference was 0.0114. Among screened normal-labeled images, the TFLite score ranged from 0.460 to approximately 1.000 (median 0.999), far above the locked threshold.

## Interpretation and limits

This is a strong external warning that the current model over-flags normal ocular-surface photographs and is not ready for reliable smartphone screening. It does **not** estimate population-level diagnostic performance: the dataset is small, web-collected, has a release/publication count discrepancy and internal duplicates, and the quality gate rejects most images. The result does not establish the clinical diagnosis of any individual photographed eye. A separate clinician-labeled, camera-native smartphone cohort with subject/device separation is still required.

To reproduce, download version 2 of the linked archive, extract only `normal/` and `pterygium/` into a directory outside the repository, then run:

```powershell
python external_eval.py PATH_TO_EXTRACTED_DIRECTORY --backend tflite --output PATH_TO_REPORT.json
```
