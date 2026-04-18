## FINANL MODEL IN final_model folder


# ExtremeTrack: Restoration-Assisted MixFormer Tracker
This project now uses a stronger tracking pipeline:

- MixFormer-style dual-stream tracker (`raw + restored`)
- Optional YOLO fallback for low-confidence re-detection
- Official QP validation (`IQA = 1 - BRISQUE/100`, positive if `IQA * CLE < 15`)

## Main files

- `D:/vistac/vistac_tracker/models.py`: MixFormer-style tracker (`mixformer_lite`, `mixformer_lite_large`)
- `D:/vistac/vistac_tracker/restoration.py`: frame restoration for haze/rain
- `D:/vistac/vistac_tracker/detector.py`: YOLO fallback detector
- `D:/vistac/vistac_tracker/engine.py`: training + tracking + official QP evaluation
- `D:/vistac/train.py`: training entrypoint
- `D:/vistac/infer.py`: validation/inference entrypoint

## High-throughput training

Use this command for full training with restoration + mixed precision:

```powershell
python D:/vistac/train.py --models mixformer_lite mixformer_lite_large --epochs 10 --batch-size 16 --samples-per-epoch 12000 --num-workers 6 --use-amp --channels-last --use-restoration
```

Enable YOLO fallback during validation:

```powershell
python D:/vistac/train.py --models mixformer_lite --epochs 10 --batch-size 16 --samples-per-epoch 12000 --num-workers 6 --use-amp --channels-last --use-restoration --yolo-fallback --yolo-model-name yolov8n.pt
```

## Output artifacts

- checkpoints: `D:/vistac/outputs/checkpoints`
- predictions: `D:/vistac/outputs/predictions`
- histories: `D:/vistac/outputs/experiments`
- leaderboard summary: `D:/vistac/outputs/leaderboard.json`

Checkpoint names include validation metrics:

`{model}_epoch{n}_iou{...}_qp{...}.pt`

## GPU utilization notes

Targeting sustained 80%+ utilization depends on batch size, model size, and dataloader throughput. If GPU usage is below target:

1. increase `--batch-size` until VRAM is near full
2. switch to `mixformer_lite_large`
3. increase `--num-workers` (6 to 10 range)
4. keep `--use-amp` and `--channels-last` enabled
5. increase `--samples-per-epoch` so training is compute-heavy
