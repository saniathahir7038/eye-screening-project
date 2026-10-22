# Eye Screening Hardware Test APK

## Install

1. Copy the APK to an ARM64 Android phone.
2. Allow installation from the app used to open the APK.
3. Install and open **Eye Screening**.
4. Choose one of the three modules.

The app processes images and videos on the phone. It does not need an internet connection.

## Pterygium

### Live test

1. Open **Pterygium screening**.
2. Tap **Capture photo**.
3. Fill the frame with one eye while keeping the full iris and visible white of the eye in view.
4. Use bright, even light and avoid glare.
5. Take the photo and wait for the immediate result.
6. Retake the photo when the app reports poor quality or an unstable framing result.

### Sample test

1. Select one of the four labelled samples.
2. Tap **Run sample test**.
3. Confirm that pterygium samples return a suspected result and no-pterygium samples return a normal result.
4. The model score and quality measurements appear under the result.

## Keratoconus / SmartKC

### Live 3D Placido test

1. Attach the 3D Placido model and switch on its light.
2. Open **Keratoconus screening**.
3. Tap **Capture Placido photo**.
4. Centre the reflected rings over the cornea and make every visible ring as sharp as possible.
5. Take one photo. Analysis starts automatically.
6. Review the detected-ring count, centre and experimental irregularity value.
7. Retake when the app reports **Inconclusive**.
8. Save the same original photo for comparison with the desktop SmartKC pipeline.

The live binary message uses an experimental ring-irregularity rule. It is a hardware research result and is not yet equivalent to the full desktop SmartKC PPK result.

### Sample test

1. Select one of the four included SmartKC images.
2. Tap **Run sample test**.
3. Wait while the mobile Placido model processes the image.
4. Review the saved reference PPK, aCLMI, Sim-K and maps.
5. Confirm that the **Mobile model check** reports its detected rings and irregularity.

## Dry Eye / DEDector

### Live 3D Placido test

1. Attach the 3D Placido model and switch on its light.
2. Open **Dry Eye screening** and select the eye.
3. Tap **Record Placido video**.
4. Centre the reflected rings and hold the phone steady.
5. Ask the person to blink once, then comfortably keep the eye open.
6. Record for 12 to 15 seconds. Stop immediately if the person is uncomfortable.
7. End the recording. Analysis starts automatically.
8. Review the experimental NIBUT result or follow the retake reason.
9. Use **Share test video** to preserve the original video for comparison with the desktop DEDector pipeline.

A measured NIBUT below 10 seconds is shown as possible tear-film instability. An unreadable capture returns **Inconclusive**.

### Sample test

1. Tap **Run synthetic sample**.
2. The included synthetic video tests playback, decoding and failure handling.
3. It has no patient label and must not be treated as a Dry Eye result.
4. A rejection or inconclusive result is expected.

## Result meanings

- **No pattern detected:** the experimental screening rule did not cross its threshold.
- **Possible or suspected pattern:** arrange a professional eye examination.
- **Inconclusive:** the image or video could not support a reliable screening output; retake it.
- All results are research screening outputs. They are not diagnoses.

## Hardware feedback to return

For every SmartKC and Dry Eye hardware test, keep:

1. Phone model and Android version.
2. 3D attachment version.
3. Original image or video.
4. Screenshot of the app result.
5. Whether capture was easy to centre and focus.
6. Any crash, delay or repeated inconclusive result.
