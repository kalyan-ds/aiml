# Models and Weights Directory

This directory stores trained model checkpoints (`*.pt`).

To build or restore the default prototype checkpoint `resnet50_fusion_v1.0.0.pt`:
```bash
python aiml/scripts/build_default_checkpoint.py
```

Checkpoint files are excluded from git tracking via `.gitignore` to prevent repository bloat (>100MB).
