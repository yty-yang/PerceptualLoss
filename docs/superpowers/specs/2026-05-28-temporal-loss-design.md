# Design: `wd-saliency-temp` Temporal Consistency Loss

**Date:** 2026-05-28  
**Branch:** conditional-flow  
**Status:** Approved

---

## Problem

When training with `wd-saliency`, large sigma values in non-salient regions give the codec freedom to hallucinate textures. These textures are individually plausible but differ frame-to-frame, causing visible flickering. WD cannot detect this problem because it measures per-frame distributional similarity, not inter-frame consistency.

---

## Solution

Add a temporal consistency loss term:

```
L_temp = λ_temp · mean_n( ||σ_t · w_t · (f_t^n - warp(f_{t-1}^n, flow_t))||² )
```

Where:
- `f_t^n` — model reconstruction of frame t (output of current forward pass)
- `f_{t-1}^n` — model reconstruction of frame t-1 (extra forward pass, detached)
- `flow_t` — precomputed optical flow from frame t-1 → t (computed on original frames)
- `σ_t` — spatial sigma map from `wd-saliency` for frame t; large in low-saliency (texture) regions
- `w_t = exp(-WD_raw(x_t, x_{t-1}) / τ)` — per-frame-pair scalar; near 1 for static/slow scenes, near 0 for large motion or scene cuts
- `λ_temp` — overall temporal loss weight (default: 0.1)
- `τ` — WD softening temperature (default: 1.0)

**Rationale for σ · w weighting:**
- σ (spatial): where WD gives the codec more freedom → enforce more temporal consistency
- w (temporal): where original frames are very different → relax consistency (motion blur, cuts are legitimate)

---

## Scope

- Applies **only** with loss type `wd-saliency-temp`
- Requires `wd-saliency` to already be active (reuses its saliency cache)
- Works only with PNGVideo (same constraint as `wd-saliency`)

---

## Architecture

### Precomputation (once before training)

Called from `main_nvrc.py` after `precompute_saliency()`:

```
precompute_temporal(dataset):
  1. Load original frames x_0 ... x_{T-1} from cache
  2. Run RAFT-small (torchvision `raft_small`, pretrained="C_T_V2") on consecutive pairs → flow_cache [T-1, 2, H, W] (CPU)
  3. Run VGG16WassersteinDistortion on (x_t, x_{t-1}) with constant log2_sigma=log2(8.0) (same default as `wd` loss) → wd_raw [T-1]
  4. Compute w_cache = exp(-wd_raw / τ)  [T-1] (CPU)
```

Stored as `self._flow_cache` and `self._w_cache` on `OverfitTask`.

RAFT is loaded as a singleton via `get_raft_model()` in `loss_utils.py` (frozen, eval mode, no grad).

### Training (per d_step)

After the existing `compute_d_loss` call in `d_step`, call `compute_temp_loss(model, output, inputs)`:

```
for each n in batch where idx[n, 0] > 0:
    t = idx[n, 0].item()

    # Previous frame forward pass
    inputs_prev = copy of inputs with idx[:,0] -= 1 (only for sample n)
    with torch.no_grad():
        f_prev = model(inputs_prev, compute_outputs=True, compute_rates=False)
    f_prev = f_prev[n].detach()   # [C, H, W]

    # Retrieve precomputed values
    flow_t   = flow_cache[t-1].to(device)   # [2, H, W]
    w_t      = w_cache[t-1].to(device)      # scalar
    sigma_t  = saliency_sigma_map(n, t)     # [1, H, W] — reuse _saliency_cache

    # Warp and loss
    warped   = warp(f_prev, flow_t)         # F.grid_sample, bilinear, border padding
    loss_n   = (sigma_t * w_t * (output[n] - warped)).pow(2).mean()

return λ_temp * mean(loss_n over valid n)
```

The extra forward pass reuses all existing `inputs` fields; only `idx` is modified.

### Warp implementation

Convert optical flow `[2, H, W]` (dx, dy in pixels) to a normalized sampling grid, then call `F.grid_sample(..., mode='bilinear', padding_mode='border', align_corners=True)`. Pixels that flow outside frame boundaries are clamped to the border (border padding).

---

## File Changes

| File | Change |
|---|---|
| `NVRC/loss_utils.py` | Add `get_raft_model()` singleton; add `set_temporal_context / get_temporal_context / clear_temporal_context` globals for `(flow_cache, w_cache)` |
| `NVRC/tasks.py` | Add `precompute_temporal(dataset)`; add `compute_temp_loss(model, output, inputs)`; extend `d_step` to call `compute_temp_loss` and add result to loss |
| `NVRC/losses.py` | Add `flow_warp(frame, flow)` helper; add `"wd-saliency-temp"` branch in `compute_loss` (delegates to `wd_saliency` for the main term; temporal term handled at d_step level) |
| `NVRC/main_nvrc.py` | Call `task.precompute_temporal(dataset)` after `task.precompute_saliency(dataset)` |
| `NVRC/scripts/configs/tasks/overfit/wd-saliency-temp.yaml` | New task config |
| `NVRC/scripts/train/nvrc_loss.sh` | Add `wd-saliency-temp` as valid `--loss_type` |
| `NVRC/main_utils.py` | Extend `OverfitTaskConfig` with `temp_weight: float = 0.1` and `temp_tau: float = 1.0` |

---

## New Config File

`scripts/configs/tasks/overfit/wd-saliency-temp.yaml`:
```yaml
loss: [1.0, wd-saliency-temp]
metric: [psnr, lpips, dists]
color_space: RGB
lamb: 1.0
temp_weight: 0.1
temp_tau: 1.0
```

---

## Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `temp_weight` (λ_temp) | 0.1 | Acts as regularizer; tune down if temporal loss dominates |
| `temp_tau` (τ) | 1.0 | Set relative to actual WD_raw distribution; log `w_cache` mean at startup to verify |

**τ calibration:** After first run, inspect logged `w_cache` mean. If most values are near 1 (τ too large → no discrimination) or near 0 (τ too small → all pairs treated as scene cuts), adjust τ accordingly.

---

## Edge Cases

- **First frame (t=0):** No temporal loss — skip.
- **Entire batch is t=0:** `compute_temp_loss` returns 0, no extra forward pass.
- **YUV format:** Temporal precompute checks for numpy cache (same guard as saliency); logs skip message if unsupported.
- **Intra-period boundaries:** Each intra-period group is trained independently with its own model. `idx[n, 0]` is the temporal patch index within the group (0-based). Frame with `idx[n, 0] == 0` has no predecessor within the group → skip. No cross-group temporal loss.
- **RAFT output resolution:** RAFT outputs flow at input resolution; no resizing needed if original frames are loaded at full resolution.
