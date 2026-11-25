# HandEncoder.mlpackage — model card

| | |
|---|---|
| Task | Embed one 7oda Auditor check pattern so that patterns from the same hand are close (cosine) and different hands are far apart |
| Architecture | Depthwise-separable 1D CNN student (run S1), distilled from a 10.7 M-parameter 1D ResNet teacher (run T2) |
| Parameters | 219,747 (8-bit palettized weights, 18 tensors) |
| File | `HandEncoder.mlpackage`, ML Program, minimum deployment iOS 17, 323 KB |
| Input | `x`: float32 `[1, 10, 256]` — SPEC §14.1 (accel HP / 0.02, gyro HP / 0.05, gravity, lag-0 template; pattern starts at index 25) |
| Outputs | `embedding`: float32 `[1, 128]`, L2-normalized; `hand`: float32 `[1, 3]` logits (right, left, twoHanded) |
| Training data | Simulated only: 10,000 synthetic people from `research/sim` (no real recordings yet) |
| Use on the phone | Cosine prototypes per (person, hand, accessory) and the SPEC §14.3 decision; the hand comes from a pooled LDA on enrollment embeddings, not from `hand` |
| Simulated result | At ≤ 2 % strangers accepted: right name in 65 % (aliasing IMU) / 84 % (filtered IMU) of cross-session checks; v0 on the same data: 6 % / 8 % (docs/V1_MODEL.md §6) |
| Limits | Not yet validated on real hands; thresholds tuned on the simulator; retrain/fine-tune on study data (docs/V1_MODEL.md §9) |

Files: `reference_io.npz` (16 inputs with the PyTorch outputs) and `export_summary.json`. Check the conversion on a Mac with `python research/v1/verify_coreml.py` (needs `coremltools` and `numpy`; CI runs it on every push).
