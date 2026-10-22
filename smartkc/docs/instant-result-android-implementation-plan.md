# SmartKC Instant-Result Module Implementation Plan

## 1. Objective

Add a Keratoconus/SmartKC module to the existing Pterygium Android APK. The module will work with the SmartKC Placido attachment and produce an offline screening result immediately after a successful eye capture. The combined application will share its shell and infrastructure while keeping the Pterygium and Keratoconus algorithms, calibration, results, and validation evidence isolated. The original SmartKC research application will remain unchanged as a reference.

The result workflow must produce, for each eye:

- an accepted source image and quality report;
- detected mire centre and Placido rings;
- axial and tangential corneal maps;
- Sim-K, mean-K, aCLMI, PPK, KISA, and relevant processing-quality indicators;
- one of `No keratoconus pattern detected`, `Suspected keratoconus - professional examination recommended`, or `Inconclusive - retake required` only after a decision rule has been specified and validated;
- a locally stored, auditable record containing the inputs, calibration profile, algorithm version, outputs, and failure reasons.

The target is an offline result within 10 seconds after the final accepted capture on the connected OnePlus 9R, with a preferred target of 5 seconds. Performance targets are engineering gates, not claims of clinical accuracy.

## 2. Scope boundaries

### Included

- A new Keratoconus module inside the existing Pterygium APK and application ID.
- Live CameraX capture and dataset replay modes.
- Right-eye and left-eye workflows.
- Automatic quality checks, short-burst capture, and best-frame selection.
- Automatic centre detection with manual correction when required.
- On-device SmartKC processing and map generation.
- Local Room records with explicit draft, completed, inconclusive, and failed states.
- Export of images, measurements, maps, and a machine-readable result manifest.
- Optional compatibility with the existing upload infrastructure after local processing is stable.

### Excluded from the first release

- A diagnostic or severity claim.
- Cloud dependence for producing a result.
- Automatic deletion of patient records.
- Reusing the existing folder list as the record database.
- Enabling a binary screening message before its decision rule and validation evidence are approved.

## 3. Reuse and attribution

The new project will retain the SmartKC code attribution required by `LICENSE-CODE` (MIT) and the documentation attribution required by `LICENSE` (CC BY 4.0). Third-party notices will cover CameraX, OpenCV, Room, numerical libraries, and any converted segmentation model.

The following existing components will be used as references or adapted:

- CameraX capture and camera-parameter collection.
- Centre, exposure, and sharpness checks from the Android application.
- Manual centre marking and metadata conventions.
- Mire preprocessing, segmentation, and localization.
- Camera/Placido geometry and working-distance calculations.
- Arc-Step surface reconstruction.
- Zernike smoothing.
- Axial and tangential map generation.
- Sim-K, aCLMI, PPK, and KISA calculations.

## 4. Proposed architecture

### Android application

- Existing `com.eyescreening.pterygium` Android project, expanded into an Eye Screening application shell with separate Pterygium and Keratoconus routes.
- Shared navigation, permissions, offline records, export, and camera infrastructure with module-specific processing contracts.
- CameraX for preview, focus, exposure, burst capture, and camera metadata.
- Room for structured tests, eyes, captures, processing runs, measurements, and exported files.
- WorkManager only for optional upload/export work; result generation will run as an immediate foreground task.
- A versioned calibration profile containing sensor size, focal length, working distance, Placido model, image orientation, crop, and algorithm parameters.

### Processing engine

- OpenCV Android/C++ for image conversion, cropping, quality checks, ring preprocessing, and map rendering.
- A native C++ numerical module for Arc-Step reconstruction and metric calculations where Kotlin would be too slow.
- A TFLite mire-segmentation model if the existing approximately 29 MB PyTorch U-Net can be converted with acceptable parity and latency.
- A classical OpenCV segmentation fallback if model conversion fails or the model output is unreliable.
- Explicit result and error objects between stages; no stage will silently continue with invalid geometry.

### Product screens

1. Home: Pterygium Screening, Keratoconus/SmartKC Screening, Records, Settings.
2. Keratoconus module: New Live Test or Research Dataset Test.
3. Test details: generated test ID, optional research identifier, age range/sex only if required by the study.
4. Hardware and calibration check.
5. Guided right-eye capture and review.
6. Guided left-eye capture and review.
7. Processing progress with named stages.
8. Per-eye maps, measurements, screening state, and quality limitations.
9. Record details and export.

## 5. Data model

Each test will have a generated immutable UUID. Patient-entered text will not be used as a filesystem key.

- `TestSession`: UUID, timestamps, status, app/algorithm versions, device and calibration profile.
- `EyeCapture`: side, original URI, accepted-frame URI, quality measurements, detected/manual centre.
- `ProcessingRun`: stage status, duration, warnings, parameters, and reproducibility hashes.
- `TopographyResult`: axial/tangential maps and numeric measurements.
- `ScreeningResult`: decision rule version, state, explanation, and referral wording.
- `GroundTruth`: optional clinical values stored separately from the generated result.

Records will distinguish `draft`, `capture_complete`, `processing`, `completed`, `inconclusive`, and `failed`. Skipping optional ground truth will not alter the completed processing result.

## 6. Phased implementation

### Phase 0 — Freeze the reference and define acceptance data

Deliverables:

- Run the existing Python pipeline on `kc_left.jpg`, `kc_right.jpg`, `nokc_left.jpg`, and `nokc_right.jpg`.
- Save centre coordinates, segmented rings, localized mire coordinates, reconstructed surfaces, maps, metrics, processing time, and errors as golden artifacts.
- Add more representative images before using the set for accuracy assessment; the four samples are sufficient only for initial technical parity.
- Record the exact reference configuration: segmentation method, localization method, 22-mire model, working distance, camera parameters, skip angles, crop, and dependency versions.
- Define tolerance for every intermediate output and final metric.

Acceptance gate:

- All golden outputs are reproducible from one documented command.
- Expected failures are represented explicitly.
- No Android port begins until inputs and output tolerances are frozen.

### Phase 1 — Combined APK foundation

Deliverables:

- Extend the existing Pterygium Android project; the original SmartKC application remains untouched.
- Add an Eye Screening home screen that routes to the existing Pterygium workflow or the new Keratoconus module.
- Add Keratoconus Live Test, Dataset Test, Records, Settings, and attribution screens.
- Room schema and session-state model.
- Dataset assets or a controlled file picker for replaying the four sample images.
- Build, unit-test, install, and cold-launch verification on the OnePlus 9R.

Acceptance gate:

- Existing Pterygium screening behavior and model output remain unchanged after the shared-shell migration.
- The combined APK updates and launches on the connected phone.
- The legacy SmartKC package can remain installed separately during comparison testing.
- A dataset test can be created and reopened without becoming a false completed record.
- Process death does not corrupt the session.

### Phase 2 — Live capture and quality control

Deliverables:

- Guided hardware alignment, rear-camera selection, tap focus, exposure control, and working-distance guidance.
- Short burst of candidate frames with centre, exposure, glare, and sharpness measurements.
- Automatic best-frame selection plus review/retake.
- Right/left-eye state machine.
- Dataset and live images feed the same post-capture interface.

Acceptance gate:

- Camera permission, rotation, interruption, and retake paths work on the physical phone.
- Every rejection includes a visible reason.
- The stored source image retains required resolution and camera metadata.

### Phase 3 — Mobile preprocessing and mire localization

Deliverables:

- Image normalization, orientation, crop, centre detection, and manual correction.
- First mobile segmentation implementation.
- Ring localization and support-angle exclusion.
- Debug overlay showing detected rings and rejected points.
- Stage-level comparison utility against Python golden artifacts.

Acceptance gate:

- Centre and mire-coordinate errors remain within the Phase 0 tolerances.
- Failure cases return `Inconclusive` instead of fabricated maps.
- Dataset replay is deterministic across repeated runs.

### Phase 4 — Corneal reconstruction port

Deliverables:

- Camera/Placido mapping and working-distance calculation.
- Arc-Step reconstruction.
- Zernike smoothing or an approved numerically equivalent implementation.
- Native memory, bounds, finite-number, and convergence checks.

Acceptance gate:

- Intermediate surfaces and radii match the Python reference within defined tolerances.
- Invalid calibration, insufficient rings, NaN/Infinity, or unstable fitting stops processing safely.
- Processing meets the memory and thermal gates on the OnePlus 9R.

### Phase 5 — Maps and measurements

Deliverables:

- Axial and tangential maps using fixed legends and units.
- Sim-K, diff, mean-K, aCLMI, PPK, KISA, and quality values.
- Exportable result manifest with algorithm and calibration versions.
- Per-stage processing timing.

Acceptance gate:

- Metric and map parity passes against every golden image.
- Legends, units, orientation, laterality, and colour mapping are verified.
- A failed metric cannot be displayed as zero or a normal result.

### Phase 6 — Screening decision layer

Deliverables:

- A versioned decision-rule interface independent of map generation.
- Three-state result: no detected pattern, suspected pattern, or inconclusive.
- Referral and research-prototype wording.
- Separate generated output and clinician-entered ground truth.

Acceptance gate:

- The rule is supported by an approved source or derived and locked using a suitable labelled development cohort.
- Evaluation uses subject-separated data that was not used to choose thresholds.
- Sensitivity, specificity, confidence intervals, failure rate, and device/hardware scope are documented.
- Until this gate passes, the APK displays maps and measurements without a binary clinical interpretation.

### Phase 7 — Immediate-result integration

Deliverables:

- Automatic processing after the final accepted eye capture.
- Named progress stages and cancellation/retry behavior.
- Results saved transactionally so partial processing cannot appear complete.
- Preferred processing time of 5 seconds and maximum target of 10 seconds on the OnePlus 9R.

Acceptance gate:

- Repeated end-to-end dataset and live runs do not crash or leak memory.
- Offline mode works with airplane mode enabled.
- Results remain reproducible after app restart.

### Phase 8 — Physical-device and hardware validation

Deliverables:

- LED-ring, ATtiny85, OTG, Placido attachment, focus-distance, and alignment tests.
- Device-specific calibration for the OnePlus 9R.
- Tests under controlled illumination, repeated captures, different operators, left/right eyes, and expected failure conditions.
- Comparison of APK outputs with the Python pipeline for the same original captures.

Acceptance gate:

- Hardware remains stable and safe during repeated use.
- Same-image Android/Python parity passes.
- Capture repeatability and operator variation are quantified.

### Phase 9 — Records, export, and optional infrastructure integration

Deliverables:

- Searchable records with explicit state and processing version.
- Export package containing original images, metadata, maps, metrics, and manifest.
- Optional upload adapter compatible with the authorized backend.
- Retry, duplicate prevention, encryption, retention, and deletion behavior.

Acceptance gate:

- Upload is never required for an immediate result.
- No internal folder or crash-log directory appears as a patient record.
- Export/import round trips preserve hashes and metadata.

### Phase 10 — Release hardening

Deliverables:

- Accessibility, privacy, threat, failure-recovery, and dependency reviews.
- Signed research APK, release notes, attribution, operator guide, and reproducible build instructions.
- Automated unit, native, instrumentation, migration, and golden-parity tests.

Acceptance gate:

- All required tests pass on a clean build and fresh installation.
- Known limitations and supported hardware/calibration profiles are visible in the app.
- Clinical or field use remains disabled until separate validation and governance requirements are satisfied.

## 7. Verification strategy

### Numerical parity

Compare Python and Android at stage boundaries rather than checking only the final screenshot:

- decoded and cropped pixel arrays;
- centre coordinates;
- segmentation masks;
- mire coordinates and rejected points;
- reconstructed radius/elevation arrays;
- smoothed surfaces;
- axial/tangential numeric maps;
- final metrics.

### Android testing

- Kotlin unit tests for state, validation, records, result wording, and failure propagation.
- C++ tests for geometry, Arc-Step, smoothing, metrics, and invalid numeric inputs.
- Instrumentation tests for camera permission, rotation, interruption, dataset selection, process death, and export.
- Physical tests for latency, memory, battery, heat, focus, and capture repeatability.

### Clinical evaluation boundary

The four sample images and existing technical pipeline outputs cannot establish screening accuracy. Any suspected/not-suspected claim requires an adequately sized, clinician-labelled, subject-separated cohort captured using the intended hardware and devices.

## 8. Principal risks and controls

- **Device calibration:** sensor size, focal length, zoom, crop, and minimum focus distance affect geometry. Use locked, versioned calibration profiles and reject unsupported configurations.
- **Python-to-mobile parity:** SciPy, scikit-image, graph clustering, and Zernike behavior may differ. Freeze intermediate outputs and port stage by stage.
- **Segmentation model conversion:** the existing PyTorch U-Net may not convert cleanly or run fast enough. Benchmark TFLite and classical OpenCV paths during Phase 3.
- **Hardware variation:** working distance and Placido geometry must be physically controlled. Add a pre-test hardware/calibration check.
- **False certainty:** processing failure must never become a normal result. Use the explicit inconclusive state throughout the pipeline.
- **Insufficient validation:** technical parity does not prove clinical validity. Keep decision logic disabled until Phase 6 passes.

## 9. Approval sequence

Development will proceed one phase at a time. Each phase ends with an installable or reviewable artifact, its test evidence, known limitations, and a request for approval before the next phase begins.

The first requested approval will be for **Phase 0 only: freeze the Python reference configuration and generate golden outputs**. No new Android application code is required during Phase 0.
