# Eye Screening Project

Research workspace for three smartphone-assisted eye-screening modules:

1. **SmartKC** — keratoconus screening from a Placido-ring eye image.
2. **DEDector** — dry-eye screening and NIBUT estimation from a Placido-ring eye video.
3. **Pterygium** — a planned custom model for screening external-eye photographs.

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

- The official SmartKC computer pipeline was run successfully on all four bundled sample images.
- An offline SmartKC Android build mode was added because Microsoft's private Firebase configuration is not public.
- The offline SmartKC debug APK builds successfully, but physical phone and Placido-hardware testing is still pending.
- The DEDector CLI and trained segmentation model load successfully on CPU.
- DEDector invalid-video handling was hardened and covered by three passing tests.
- Complete DEDector validation still requires a genuine mire-pattern eye video.
- Pterygium dataset selection and model implementation have not started in this repository.

## Upstream projects

- [Microsoft SmartKC](https://github.com/microsoft/SmartKC-A-Smartphone-based-Corneal-Topographer)
- [Microsoft DEDector](https://github.com/microsoft/DEDector)

The upstream licences and attribution files are retained inside their respective module directories.

## Local-only content

Python environments, Android SDK files, generated builds, test recordings, APKs, and result outputs are intentionally excluded from Git. Recreate them locally using the module requirements and build configuration.

## Safety

This is an academic research prototype, not a medical diagnostic system. Human testing and patient-image collection require appropriate consent, privacy protection, clinical supervision, and institutional ethics approval.
