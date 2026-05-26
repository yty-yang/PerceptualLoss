# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research repository extending **NVRC** (Neural Video Representation Compression, NeurIPS 2024) — an INR-based neural video codec. The project's focus is on experimenting with perceptual loss functions (Wasserstein distortion, RankDVQA, saliency-guided WD) as alternatives to the default L1/MS-SSIM training objective.

## Environment

Use the `perceptual` conda environment:
```bash
conda activate perceptual
```

## Running Experiments

All training is run from the `NVRC/` directory using `accelerate launch`:

**Standard UVG overfitting (baseline):**
```bash
cd NVRC
bash scripts/train/overfitting_uvg_nvrc.sh <GPU_ID> <VID> <LAMB> <SCALE> <LR_S1> <LR_S2> <GRAD_ACCUM> <BATCH_SIZE>
# e.g.: bash scripts/train/overfitting_uvg_nvrc.sh 0 Beauty 1.0 s 2e-3 1e-4 1 144
```

**Perceptual loss experiments (wd / rankdvqa / wd-saliency):**
```bash
cd NVRC
bash scripts/train/nvrc_loss.sh --gpu_id 0 --vid Beauty --lamb 1.0 --scale s \
    --lr_s1 2e-3 --lr_s2 1e-4 --grad_accum 1 --batch_size 144 --loss_type wd-saliency
```

Both scripts run two sequential training stages (S1: 360 epochs, S2: 30 epochs fine-tuning). S2 resumes the model from S1 (`--resume-model-only`). Outputs go to `Outputs/NVRC/<DATASET>/<EXP_NAME>/`.

**Direct `accelerate launch` for custom runs:**
```bash
cd NVRC && accelerate launch --gpu_ids=0 --num_processes=1 --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config <yaml> --train-task-config <yaml> ...
```

## Analyzing Results

```bash
cd Analysis
python analyze.py               # summary table + RD plots
python analyze.py --no-plots    # table only
python analyze.py --outputs /path/to/Outputs  # custom path
```

Reads `results/all.txt` from each `s2`-stage experiment under `Outputs/`. Saves `results.csv` and plots to `Analysis/plots/`.

## Architecture

### Training pipeline (`main_nvrc.py` + `main_utils.py`)

The main loop iterates over **intra-period groups of frames**. For each group:
1. Creates dataset and task objects
2. Builds the codec model (`compress_model` wrapping an INR `model`)
3. Runs training + evaluation epochs
4. Compresses model weights to bitstream, decompresses, and evaluates decoded output

Config is entirely YAML-driven. `parse_args()` merges a hierarchy of YAML config files (experiment, data, task, compress model, INR model) with CLI overrides.

### Model hierarchy
- **`models/`**: INR backbone (HiNeRV-v2). `hinerv.py` is the main model; `encoding.py`, `layers.py`, `upsample.py` are building blocks.
- **`compress_models/`**: Wraps the INR model for entropy coding. `nvrc.py` implements compress/decompress.
- **`entropy_models/`**: Entropy coding layers (weight entropy model, grid entropy model).

### Loss system (`losses.py` + `losses_helpers.py`)
- `compute_loss(name, x, y)` dispatches to named loss functions. Available losses: `mse`, `l1`, `ms-ssim`, `ssim`, `wd`, `wd-saliency`, `rankdvqa`, and YUV-weighted variants.
- Perceptual model singletons (RankDVQA, WassersteinDistortion, EMLNETSaliency) live in `loss_models/` and are lazily loaded in `losses_helpers.py`.
- **`wd-saliency`** requires precomputed saliency maps. `OverfitTask.precompute_saliency()` runs EMLNETSaliency on the full video before training starts, caching `[T, 1, h_s, w_s]` tensors on CPU. During training, `set_saliency_context()` / `clear_saliency_context()` inject the precomputed maps into `wd_saliency()` via module-level globals in `losses_helpers.py`.

### Task (`tasks.py`)
`OverfitTask` manages the loss/metric configuration, per-frame I/O logging, and bridges `parse_batch` → `d_step` / `r_step` → loss computation. The distortion step (`d_step`) and rate step (`r_step`) are alternated during training when `--rate-steps` > 0.

### Data (`datasets.py`, `io_utils.py`)
Supports PNG frame sequences and raw YUV420p. Videos are divided into 3D patches `[T, H, W]`; the dataset returns `(idx, patch)` pairs where `idx` is a 3D patch coordinate.

## Key Config Files

All in `NVRC/scripts/configs/`:
- `nvrc/overfit/s1-360e.yaml`, `s2-30e.yaml` — stage training schedules
- `nvrc/compress_models/nvrc_s1.yaml`, `nvrc_s2.yaml` — compression model settings per stage
- `nvrc/models/uvg_hinerv-v2-{s,m,l}_1920x1080.yaml` — INR model sizes
- `tasks/overfit/{wd,rankdvqa,wd-saliency,l1_ms-ssim}.yaml` — loss/metric configs

## Output Structure

```
Outputs/NVRC/<VIDEO>/<EXP_NAME>/
├── bitstreams/        # compressed bitstreams per group
├── checkpoints/       # training checkpoints
├── outputs/           # decoded frames (when eval-enable-log=true)
├── results/           # CSV-format metrics (all.txt = video-level average)
└── rank_0/models/     # model architecture dumps
```
