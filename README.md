# Eye Screening Project

Research workspace for three smartphone-assisted eye-screening modules:

1. **SmartKC** — keratoconus screening from a Placido-ring eye image.
2. **DEDector** — dry-eye screening and NIBUT estimation from a Placido-ring eye video.
3. **Pterygium** — a MobileNetV2 research model and local screening interface for external-eye photographs.

## Repository structure

```text
eye-screening-project/
|-- smartkc/
|-- dedector/
|-- pterygium/
|-- test-data/
|-- results/
|-- shared/
`-- docs/
```

## Current status

- One combined Android hardware-validation APK now provides live and bundled-sample paths for Pterygium, Keratoconus/SmartKC, and Dry Eye/DEDector.
- SmartKC now runs the shared Placido U-Net on-device and reports capture validity plus an experimental ring-irregularity screening result.
- Dry Eye now analyses a captured Placido video immediately and reports experimental NIBUT or a clear retake reason.
- See [the Android quick-use guide](pterygium/android_app/USAGE.md) for the current workflow and capability of each module.
- The official SmartKC computer pipeline was run successfully on all four bundled sample images.
- An offline SmartKC Android build mode was added because Microsoft's private Firebase configuration is not public.
- The offline SmartKC debug APK builds successfully, but live decisions still require physical phone, desktop-parity and Placido-hardware validation.
- The DEDector CLI and trained segmentation model load successfully on CPU.
- DEDector invalid-video handling was hardened and covered by three passing tests.
- Complete DEDector/mobile parity validation still requires a genuine mire-pattern eye video.
- The Pterygium SLID audit, preprocessing, model training, locked evaluation, corrective validation iteration, Python interface, TensorFlow Lite export and standalone Android integration are complete. Physical-device and independent smartphone validation remain pending.

## Upstream projects

- [Microsoft SmartKC](https://github.com/microsoft/SmartKC-A-Smartphone-based-Corneal-Topographer)
- [Microsoft DEDector](https://github.com/microsoft/DEDector)

The upstream licences and attribution files are retained inside their respective module directories.

## Local-only content

Python environments, Android SDK files, generated builds, test recordings, APKs, and result outputs are intentionally excluded from Git. Recreate them locally using the module requirements and build configuration.

## Safety

This is an academic research prototype, not a medical diagnostic system. Human testing and patient-image collection require appropriate consent, privacy protection, clinical supervision, and institutional ethics approval.
