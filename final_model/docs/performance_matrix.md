# Expected Performance Matrix

This file summarizes the expected stage-wise behavior of the implemented pipeline. Since no local annotation files were found in the current workspace, the values below are split into:

- expected paper-aligned ablation numbers
- locally reportable proxy metrics

## Pipeline stages

| Stage | Input | Output | Local metric(s) available now | Expected effect |
| --- | --- | --- | --- | --- |
| Restoration | Raw hazy/rainy frame | Restored frame | contrast gain, sharpness gain, entropy gain, optional synthetic PSNR/SSIM during GAN training | clearer object boundaries and improved visibility |
| Detection | Original + restored frame | fused YOLOv8 detections | detections per frame, mean confidence, fusion gain | higher recall and cleaner localization on degraded scenes |
| Tracking | Fused detections | tracked trajectories | unique tracks, average track length, longest track, mean track confidence, temporal smoothness proxy | fewer dropped tracks and more stable identities |

## Paper-reported expected ablation

These values come directly from the supplied PDF and should be treated as target expectations for the full system:

| Model configuration | Expected Avg QP |
| --- | --- |
| YOLOv8 only (baseline) | 0.015 |
| YOLOv8 + BoT-SORT | 0.021 |
| GAN + YOLOv8 | 0.024 |
| Full model (GAN + YOLOv8 + BoT-SORT) | 0.028 |

## Paper-reported expected final metrics

| Metric | Expected value |
| --- | --- |
| Average QP | 0.028 |
| Average IoU | 0.124 |
| Success Rate @0.5 | 0.031 |
| Success Rate @20 | 0.035 |
| Average OTE | 190 |

## How to interpret local runs

Without ground truth, the local runner cannot compute QP, IoU, OTE, or success-rate scores. Instead:

- restoration proxies should trend upward after the restoration stage
- fused detections should be at least as stable as single-path detections
- track length and mean confidence should improve when restoration and dual-path fusion are enabled

Once annotations are added, the same pipeline can be extended with benchmark evaluation scripts for true QP/IoU reporting.
