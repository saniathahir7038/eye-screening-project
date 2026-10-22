# SmartKC bundled-dataset internal test - 2026-10-10

## Purpose

This test checks whether the existing SmartKC analysis pipeline and the four sample images bundled in the repository can serve as the technical reference for a Keratoconus module inside the existing Pterygium Android application. It does not test the phone camera or Placido hardware.

## Configuration

- Inputs: `kc_left.jpg`, `kc_right.jpg`, `nokc_left.jpg`, `nokc_right.jpg`
- Centre selection: automatic
- Mire segmentation: PyTorch U-Net (`dl`)
- Mire localization: graph clustering
- Mires: 22
- Nominal working distance: 75 mm
- Camera parameters: 4.8 x 6.4 mm sensor, 4.755 mm focal length
- Python: 3.10.21
- OpenCV: 4.7.0
- PyTorch: 1.13.1 CPU

Command:

```powershell
../shared/environments/smartkc/Scripts/python.exe main.py `
  --n_mires 22 `
  --working_distance 75.0 `
  --camera_params "4.8 6.4 4.755" `
  --model_file ./data/ring_distribution.txt `
  --base_dir data/sample_images `
  --center_selection auto `
  --mire_seg_method dl `
  --mire_loc_method graph_cluster `
  --verbose `
  --output_dir artifacts/instant_result_phase0_2026-10-10
```

## Result

All four images completed without a pipeline exception. Each image generated a segmentation mask, cleaned mire overlay, reconstructed corneal surface, axial map, tangential map, masked maps, and numeric outputs.

| Image | Filename label | Sim-K1 | Sim-K2 | Difference | Axis | Effective distance | aCLMI | PPK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nokc_right` | No KC | 39.44 D | 38.28 D | 1.16 D | 102 deg | 71.47 mm | 2.52 | 0.25 |
| `nokc_left` | No KC | 41.21 D | 40.79 D | 0.42 D | 112 deg | 72.56 mm | 1.49 | 0.04 |
| `kc_right` | KC | 50.40 D | 45.28 D | 5.12 D | 122 deg | 73.88 mm | 4.30 | 0.94 |
| `kc_left` | KC | 50.15 D | 45.30 D | 4.85 D | 63 deg | 74.71 mm | 6.03 | 1.00 |

The KC-labelled pair has higher corneal power, Sim-K difference, aCLMI, and PPK than the no-KC-labelled pair in this four-image smoke set. This is a useful end-to-end technical signal. Four filename-labelled examples cannot establish a screening threshold, sensitivity, specificity, or clinical validity.

## Historical-output comparison

The newly generated 500 x 500 maps are visually consistent with the historical sample maps committed in `data/sample_images_results`. Structural similarity (SSIM) ranges from 0.817 to 0.955 for axial maps and 0.733 to 0.901 for tangential maps.

The files are not byte-identical. The repository does not record the exact environment and command that produced the historical maps, so they cannot be used as strict numerical golden files. The outputs from this run, together with its recorded environment and hashes, become the reproducible baseline for the Android port.

## Performance finding

The complete CPU run took 429.2 seconds:

| Image | Approximate processing time |
|---|---:|
| `nokc_right` | 161 seconds |
| `nokc_left` | 113 seconds |
| `kc_right` | 85 seconds |
| `kc_left` | 70 seconds |

The current Python implementation cannot meet the proposed immediate-result target by being embedded unchanged. The Android module must optimize or port segmentation, Arc-Step reconstruction, Zernike smoothing, and map generation. Stage-level profiling is required before selecting Kotlin, native C++, OpenCV, and TFLite boundaries.

## Integration conclusion

Adding Keratoconus as a module inside the Pterygium application is technically feasible. The two modules can share the application shell, navigation, permissions, offline storage, records, export, and camera infrastructure. They must keep separate capture guidance, preprocessing, models/algorithms, calibration, results, and validation evidence.

Recommended module entry points:

- `Pterygium screening`
- `Keratoconus / SmartKC screening`
- `Records`
- `Research dataset test`

The Keratoconus module should initially support dataset replay and display maps and measurements. A binary screening message should remain disabled until a documented rule is locked and evaluated on an adequate clinician-labelled, subject-separated dataset captured with the intended hardware.

## Evidence

Generated files and their SHA-256 hashes are recorded locally in:

- `artifacts/instant_result_phase0_2026-10-10/baseline_manifest.json`
- `artifacts/instant_result_phase0_2026-10-10/dl_graph_cluster_simk.csv`
- per-image folders under `artifacts/instant_result_phase0_2026-10-10/`

The generated artifact directory is excluded from Git because it contains reproducible binary and numeric outputs. This report and the implementation plan remain reviewable source documentation.

## Phase status

The four-image internal smoke baseline passes. Phase 0 remains partially open until stage-level timings and numerical tolerances are defined for Android parity. No Android application source was changed during this test.
