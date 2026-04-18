# Adverse Weather Tracking Pipeline

This workspace now contains a modular Python implementation of the paper:

`Robust Multi-Stage Object Tracking in Adverse Weather Conditions using GAN-based Restoration, YOLOv8 Detection, and BoT-SORT Tracking`

The implementation is designed around the datasets already present in this directory:

- `D:\vistac\final_model\HAZY`
- `D:\vistac\final_model\RAIN`

## What is included

- A weather-aware restoration module with:
  - a conditional GAN generator/discriminator definition
  - a synthetic paired-data training script
  - a classical fallback restorer for immediate inference when no GAN checkpoint is available
- A YOLOv8 dual-path detector that runs on:
  - original frames
  - restored frames
  - fused detections after NMS
- A BoT-SORT tracker wrapper with lightweight temporal smoothing
- Dataset profiling and sequence discovery utilities
- Stage-wise proxy metrics and an expected performance matrix

## Repository layout

- `src/weather_track/`
- `scripts/train_restoration.py`
- `scripts/run_pipeline.py`
- `scripts/profile_dataset.py`
- `docs/performance_matrix.md`

## Important assumption

The local `HAZY` and `RAIN` folders contain frame sequences, but no ground-truth annotation files were found in the current directory. Because of that:

- the full pipeline is implemented and runnable
- restoration training uses synthetic haze/rain generation built from the local frames
- the pipeline reports proxy metrics locally
- the paper's benchmark values are documented as expected targets, not validated local scores

## Quick start

Profile the dataset:

```powershell
python scripts/profile_dataset.py
```

Run the pipeline on a short validation sequence:

```powershell
python scripts/run_pipeline.py --dataset HAZY --split val --sequence Bird2 --max-frames 10
```

Train the restoration GAN on synthetic pairs from the local frames:

```powershell
python scripts/train_restoration.py --epochs 5 --batch-size 4
```

Use a trained restoration checkpoint during inference:

```powershell
python scripts/run_pipeline.py --dataset RAIN --split val --sequence Boy --restoration-checkpoint weights/restoration_best.pt
```
