# `wd-saliency-temp` Temporal Consistency Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a temporal consistency loss `L_temp = λ · mean(||σ_t · w_t · (f_t − warp(f_{t−1}, flow_t))||²)` that prevents flickering when training with `wd-saliency`, by penalising inter-frame inconsistency in regions where WD gives the codec the most freedom.

**Architecture:** Precompute optical flow (RAFT) and per-pair WD weights on original frames once before training — mirroring how saliency is precomputed. During `d_step`, run a second forward pass to get `f_{t−1}`, warp it with the cached flow, weight by σ (saliency-derived) and w (WD-based inter-frame similarity), and add to the distortion loss.

**Tech Stack:** PyTorch, torchvision `raft_small` (already verified available), existing `VGG16WassersteinDistortion` singleton.

---

## File Map

| File | Change |
|---|---|
| `NVRC/losses.py` | Add `flow_warp()` helper; add `"wd-saliency-temp"` branch in `compute_loss` |
| `NVRC/loss_utils.py` | Add `get_raft_model()` singleton; add temporal context globals + set/get/clear |
| `NVRC/main_utils.py` | Add `temp_weight` and `temp_tau` fields to `OverfitTaskConfig` |
| `NVRC/tasks.py` | Add `temp_weight`/`temp_tau` to `OverfitTask.__init__`; add `_flow_cache`/`_w_cache`; add `precompute_temporal()`; add `compute_temp_loss()`; extend `d_step` |
| `NVRC/main_nvrc.py` | Call `precompute_temporal()` after `precompute_saliency()` |
| `NVRC/scripts/configs/tasks/overfit/wd-saliency-temp.yaml` | New task config |
| `NVRC/scripts/train/nvrc_loss.sh` | Add `wd-saliency-temp` as valid loss type |
| `NVRC/tests/test_temporal_loss.py` | Unit tests for `flow_warp` and temporal context |

---

## Task 1: `flow_warp()` helper in `losses.py`

**Files:**
- Modify: `NVRC/losses.py` (after the `wd_saliency` function, before `lpips_metric`)
- Create: `NVRC/tests/test_temporal_loss.py`

- [ ] **Step 1: Write the failing test**

Create `NVRC/tests/__init__.py` (empty) and `NVRC/tests/test_temporal_loss.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
from losses import flow_warp


def test_flow_warp_zero_flow_is_identity():
    """Zero flow should return the original frame unchanged."""
    frame = torch.rand(2, 3, 64, 64)
    flow = torch.zeros(2, 2, 64, 64)
    warped = flow_warp(frame, flow)
    assert warped.shape == frame.shape
    assert torch.allclose(warped, frame, atol=1e-5), f"max diff: {(warped - frame).abs().max()}"


def test_flow_warp_shift_right_by_one():
    """Shifting every pixel one column right should give a rightward-shifted image."""
    H, W = 16, 16
    # Solid-colour columns so shifting is easy to verify
    frame = torch.zeros(1, 1, H, W)
    for x in range(W):
        frame[0, 0, :, x] = float(x)   # column value = x index

    # Forward flow (dx=1, dy=0): pixel at col x in prev is at col x+1 in curr.
    # Backward warp: output pixel at col x samples from col x-1 in prev.
    # So output[:, :, :, x] ≈ x - 1.
    flow = torch.zeros(1, 2, H, W)
    flow[0, 0] = 1.0   # dx = +1

    warped = flow_warp(frame, flow)
    # Interior columns 1..W-2 should equal (x - 1)
    for x in range(1, W - 1):
        expected = float(x - 1)
        got = warped[0, 0, H // 2, x].item()
        assert abs(got - expected) < 1e-4, f"col {x}: expected {expected}, got {got}"


def test_flow_warp_output_shape():
    for N, C, H, W in [(1, 3, 32, 32), (4, 1, 48, 64)]:
        frame = torch.rand(N, C, H, W)
        flow = torch.zeros(N, 2, H, W)
        assert flow_warp(frame, flow).shape == (N, C, H, W)


if __name__ == '__main__':
    test_flow_warp_zero_flow_is_identity()
    test_flow_warp_shift_right_by_one()
    test_flow_warp_output_shape()
    print('All flow_warp tests passed.')
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python tests/test_temporal_loss.py
```

Expected: `ImportError: cannot import name 'flow_warp' from 'losses'`

- [ ] **Step 3: Add `flow_warp()` to `losses.py`**

Insert after the `wd_saliency` function (after line 299), before `def lpips_metric`:

```python
def flow_warp(frame, flow):
    """Backward warp frame using forward optical flow.

    For each output pixel (x, y), samples frame at (x − dx, y − dy) where
    (dx, dy) = flow[:, :, y, x].  This is the standard backward-warp
    approximation used when only forward flow is available.

    Args:
        frame: [N, C, H, W] float tensor — frame to warp (f_{t−1})
        flow:  [N, 2, H, W] float tensor — forward flow in pixels;
               channel 0 = dx (horizontal), channel 1 = dy (vertical)
    Returns:
        [N, C, H, W] warped frame, border-clamped.
    """
    N, C, H, W = frame.shape

    # Build base grid of pixel coordinates [N, H, W, 2] (x, y order)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=frame.dtype, device=frame.device),
        torch.arange(W, dtype=frame.dtype, device=frame.device),
        indexing='ij',
    )
    # [H, W, 2] → [N, H, W, 2]
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

    # Subtract forward flow to get backward-warp sampling locations
    # flow: [N, 2, H, W] → [N, H, W, 2]
    grid = grid - flow.permute(0, 2, 3, 1)

    # Normalise to [-1, 1] for grid_sample
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0   # x
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0   # y

    return F.grid_sample(
        frame, grid,
        mode='bilinear', padding_mode='border', align_corners=True,
    )
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python tests/test_temporal_loss.py
```

Expected: `All flow_warp tests passed.`

- [ ] **Step 5: Commit**

```bash
git add NVRC/losses.py NVRC/tests/__init__.py NVRC/tests/test_temporal_loss.py
git commit -m "feat: add flow_warp helper and unit tests"
```

---

## Task 2: RAFT singleton in `loss_utils.py`

**Files:**
- Modify: `NVRC/loss_utils.py`
- Modify: `NVRC/tests/test_temporal_loss.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `NVRC/tests/test_temporal_loss.py`:

```python
import os, sys
# Add both NVRC/ and PerceptualLoss/ to path so NVRC package imports work
_NVRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_NVRC)
for p in (_NVRC, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from NVRC.loss_utils import get_raft_model


def test_raft_model_output_shape():
    """RAFT should return flow list whose last entry has shape [N, 2, H, W]."""
    from torchvision.models.optical_flow import Raft_Small_Weights
    device = 'cpu'
    raft = get_raft_model(device)
    transforms = Raft_Small_Weights.C_T_V2.transforms()
    img1 = torch.rand(1, 3, 128, 128)
    img2 = torch.rand(1, 3, 128, 128)
    img1_t, img2_t = transforms(img1, img2)
    with torch.no_grad():
        preds = raft(img1_t, img2_t)
    assert preds[-1].shape == (1, 2, 128, 128), f"Unexpected shape: {preds[-1].shape}"
    print('RAFT output shape OK.')
```

Run to confirm it fails:
```bash
cd /Users/yang/PerceptualLoss
conda run -n perceptual python -c "
import sys; sys.path.insert(0, 'NVRC'); sys.path.insert(0, '.')
from NVRC.loss_utils import get_raft_model
" 2>&1 | grep -i "importerror\|attributeerror" | head -5
```

Expected: `AttributeError` or `ImportError` on `get_raft_model`.

- [ ] **Step 2: Add `get_raft_model` singleton to `loss_utils.py`**

In `loss_utils.py`, add one line after the existing `_dists_model = None` global:

```python
_raft_model = None
```

After the existing `get_dists_model` function, add:

```python
def get_raft_model(device):
    global _raft_model
    if _raft_model is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        _raft_model = raft_small(weights=Raft_Small_Weights.C_T_V2).to(device).eval()
        for p in _raft_model.parameters():
            p.requires_grad_(False)
    return _raft_model
```

- [ ] **Step 3: Run the test**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys, os
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import tests.test_temporal_loss as t
t.test_raft_model_output_shape()
"
```

Expected: `RAFT output shape OK.`

- [ ] **Step 4: Commit**

```bash
git add NVRC/loss_utils.py NVRC/tests/test_temporal_loss.py
git commit -m "feat: add RAFT singleton and temporal context helpers"
```

---

## Task 3: Extend `OverfitTaskConfig` and `OverfitTask.__init__`

**Files:**
- Modify: `NVRC/main_utils.py` (lines ~124–130, `OverfitTaskConfig`)
- Modify: `NVRC/tasks.py` (lines ~7–36, `OverfitTask.__init__`; lines ~220–230, `create_overfit_task`)

- [ ] **Step 1: Add fields to `OverfitTaskConfig` in `main_utils.py`**

Find the `OverfitTaskConfig` dataclass (around line 124). Add two new fields after `enable_log`:

```python
@dataclass
class OverfitTaskConfig:
    loss: str = 'mse'
    metric: str = 'psnr'
    color_space: str = 'RGB'
    lamb: float = 1.0
    enable_log: bool = False
    temp_weight: float = 0.1
    temp_tau: float = 1.0
```

- [ ] **Step 2: Add parameters to `OverfitTask.__init__` in `tasks.py`**

The current signature is:
```python
def __init__(self, logger, video, loss_cfg, metric_cfg, lamb,
             channel_scale=None, channel_shift=None,
             enable_log=True, training=True, device=None):
```

Change to:
```python
def __init__(self, logger, video, loss_cfg, metric_cfg, lamb,
             channel_scale=None, channel_shift=None,
             enable_log=True, training=True, device=None,
             temp_weight=0.1, temp_tau=1.0):
```

In the body of `__init__`, after `self._saliency_cache: torch.Tensor | None = None`, add:

```python
        self.temp_weight = float(temp_weight)
        self.temp_tau = float(temp_tau)
        self._flow_cache: torch.Tensor | None = None   # [T-1, 2, H, W] CPU
        self._w_cache: torch.Tensor | None = None      # [T-1] CPU
```

- [ ] **Step 3: Pass `temp_weight`/`temp_tau` from `create_overfit_task` in `tasks.py`**

Find `create_overfit_task` (around line 220). Change the `OverfitTask(...)` call:

```python
    task = OverfitTask(logger, video, loss_cfg=config.loss, metric_cfg=config.metric,
                       lamb=config.lamb, channel_scale=channel_scale, channel_shift=channel_shift,
                       enable_log=config.enable_log, training=training, device=device,
                       temp_weight=getattr(config, 'temp_weight', 0.1),
                       temp_tau=getattr(config, 'temp_tau', 1.0))
```

- [ ] **Step 4: Verify smoke test**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
from main_utils import OverfitTaskConfig
cfg = OverfitTaskConfig()
print('temp_weight:', cfg.temp_weight, '  temp_tau:', cfg.temp_tau)
cfg2 = OverfitTaskConfig(temp_weight=0.05, temp_tau=2.0)
assert cfg2.temp_weight == 0.05 and cfg2.temp_tau == 2.0
print('OverfitTaskConfig fields OK.')
"
```

Expected:
```
temp_weight: 0.1   temp_tau: 1.0
OverfitTaskConfig fields OK.
```

- [ ] **Step 5: Commit**

```bash
git add NVRC/main_utils.py NVRC/tasks.py
git commit -m "feat: add temp_weight/temp_tau config and OverfitTask attributes"
```

---

## Task 4: `precompute_temporal()` in `tasks.py`

**Files:**
- Modify: `NVRC/tasks.py` (add method after `precompute_saliency`)

- [ ] **Step 1: Add the import at the top of `tasks.py`**

Find the existing import line:
```python
from NVRC.loss_utils import get_saliency_model, set_saliency_context, clear_saliency_context
```

Replace it with:
```python
from NVRC.loss_utils import (
    get_saliency_model, set_saliency_context, clear_saliency_context,
    get_raft_model, get_wloss,
)
```

- [ ] **Step 2: Add `precompute_temporal()` after `precompute_saliency()`**

Insert after the `precompute_saliency` method (after its closing line), before `def parse_batch`:

```python
    def precompute_temporal(self, dataset) -> None:
        """Precompute optical flow and WD-based weights for consecutive frame pairs.

        Stores:
            _flow_cache: [T-1, 2, H, W] CPU tensor — RAFT forward flow from t-1 → t
            _w_cache:    [T-1] CPU tensor — exp(-WD_raw(x_t, x_{t-1}) / tau)
        """
        if not any(loss_type == 'wd-saliency-temp' for loss_type in self.loss_cfg[1::2]):
            return

        video = dataset.video
        cache = getattr(video, 'cache', None)
        if cache is None or not (hasattr(cache, 'ndim') and cache.ndim == 4):
            self.logger.info('Temporal precomputation skipped: unsupported video format.')
            return

        T_total = dataset.get_num_frames()
        if T_total < 2:
            self.logger.info('Temporal precomputation skipped: need >= 2 frames.')
            return

        device = self.device
        from torchvision.models.optical_flow import Raft_Small_Weights

        raft = get_raft_model(device)
        raft_transforms = Raft_Small_Weights.C_T_V2.transforms()
        wloss = get_wloss(device)

        # Constant sigma map for WD_raw (log2(8) matches `wd` loss default sigma_const=8)
        H_orig, W_orig = cache.shape[1], cache.shape[2]
        log2_sigma = torch.full(
            (1, 1, H_orig, W_orig), math.log2(8.0),
            dtype=torch.float32, device=device,
        )

        scale = dataset.channel_scale.view(1, -1, 1, 1)
        shift = dataset.channel_shift.view(1, -1, 1, 1)

        flow_list: list[torch.Tensor] = []
        wd_list: list[float] = []
        prev_frame: torch.Tensor | None = None

        for t in range(T_total):
            frame_np = cache[t:t + 1]  # [1, H, W, C]
            frame = (
                torch.from_numpy(frame_np.astype(np.float32))
                .permute(0, 3, 1, 2)
            )
            frame = (frame * scale + shift).clamp(0.0, 1.0).to(device)  # [1, C, H, W]

            if prev_frame is not None:
                # Optical flow from prev frame to current frame
                img1_t, img2_t = raft_transforms(prev_frame, frame)
                with torch.no_grad():
                    flow_preds = raft(img1_t, img2_t)
                flow_list.append(flow_preds[-1].squeeze(0).cpu())  # [2, H, W]

                # WD between consecutive original frames (no grad needed)
                with torch.no_grad():
                    wd_val = wloss(prev_frame, frame, log2_sigma).mean().item()
                wd_list.append(wd_val)

            prev_frame = frame

        self._flow_cache = torch.stack(flow_list, dim=0)          # [T-1, 2, H, W]
        wd_tensor = torch.tensor(wd_list, dtype=torch.float32)
        self._w_cache = torch.exp(-wd_tensor / self.temp_tau).cpu()  # [T-1]

        self.logger.info(
            f'Precomputed temporal cache: flow={self._flow_cache.shape}, '
            f'w_mean={self._w_cache.mean():.3f}, w_min={self._w_cache.min():.3f}'
        )
```

- [ ] **Step 3: Verify shape smoke test** (requires a real video; skip if no video available, verify import only)

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
from tasks import OverfitTask
import inspect
src = inspect.getsource(OverfitTask.precompute_temporal)
assert 'flow_preds[-1]' in src
assert '_flow_cache' in src
assert '_w_cache' in src
print('precompute_temporal source OK.')
"
```

Expected: `precompute_temporal source OK.`

- [ ] **Step 4: Commit**

```bash
git add NVRC/tasks.py
git commit -m "feat: add precompute_temporal() to OverfitTask"
```

---

## Task 5: `compute_temp_loss()` in `tasks.py`

**Files:**
- Modify: `NVRC/tasks.py` (add method after `precompute_temporal`)

- [ ] **Step 1: Write the failing test**

Append to `NVRC/tests/test_temporal_loss.py`:

```python
def test_compute_temp_loss_returns_zero_when_no_cache():
    """compute_temp_loss should return 0 tensor when _flow_cache is None."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from unittest.mock import MagicMock
    import torch

    # Minimal OverfitTask mock
    task = MagicMock()
    task._flow_cache = None
    task._w_cache = None

    # Call the real method (unbound) with the mock as self
    from tasks import OverfitTask
    result = OverfitTask.compute_temp_loss(task, model=None, output=torch.zeros(2,3,1,8,8), inputs={'idx': torch.zeros(2,3,dtype=torch.long)})
    assert result.item() == 0.0
    print('compute_temp_loss no-cache returns 0 OK.')


def test_compute_temp_loss_skips_t0_samples():
    """All samples with t=0 should produce zero temporal loss."""
    from unittest.mock import MagicMock
    from tasks import OverfitTask

    task = MagicMock()
    task._flow_cache = torch.zeros(1, 2, 8, 8)  # non-None cache
    task._w_cache = torch.ones(1)
    task._saliency_cache = None
    task.temp_weight = 0.1
    task.parse_output = lambda x: x

    output = torch.rand(3, 3, 1, 8, 8, requires_grad=True)
    idx = torch.zeros(3, 3, dtype=torch.long)   # all t=0
    inputs = {'idx': idx, 'x': None, 'lamb': torch.tensor([1.0]),
              'vidx': torch.zeros(3, dtype=torch.int32),
              'vidx_max': 1, 'idx_max': (1, 1, 1),
              'rel_batch_size': 1.0,
              'video_size': (1, 8, 8), 'patch_size': (1, 8, 8), 'channels': 3}

    result = OverfitTask.compute_temp_loss(task, model=MagicMock(), output=output, inputs=inputs)
    assert result.item() == 0.0, f"Expected 0, got {result.item()}"
    print('compute_temp_loss t=0 returns 0 OK.')
```

Run:
```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
import tests.test_temporal_loss as t
t.test_compute_temp_loss_returns_zero_when_no_cache()
" 2>&1 | tail -5
```

Expected: AttributeError on `OverfitTask.compute_temp_loss` (method not yet added).

- [ ] **Step 2: Add `compute_temp_loss()` to `tasks.py`**

Insert after `precompute_temporal`, before `parse_batch`:

```python
    def compute_temp_loss(self, model, output, inputs):
        """Temporal consistency loss term for wd-saliency-temp.

        For each batch sample with temporal index t > 0:
          1. Run a second model forward pass at t-1 (detached).
          2. Warp f_{t-1} to frame t using precomputed RAFT flow.
          3. Weight the squared difference by σ_t (saliency sigma) · w_t (WD weight).

        Returns scalar tensor (0.0 when no valid pairs or cache absent).
        """
        if self._flow_cache is None or self._w_cache is None:
            return torch.tensor(0.0, device=output.device)

        idx = inputs['idx']      # [N, 3]
        t_vals = idx[:, 0]       # [N]
        valid_mask = t_vals > 0  # bool [N]

        if not valid_mask.any():
            return torch.tensor(0.0, device=output.device)

        device = output.device

        # Build a copy of inputs with temporal index decremented for valid samples.
        # Invalid samples (t=0) keep their index — their f_prev outputs are unused.
        idx_prev = idx.clone()
        idx_prev[valid_mask, 0] = t_vals[valid_mask] - 1
        inputs_prev = dict(inputs)
        inputs_prev['idx'] = idx_prev
        inputs_prev['x'] = None

        with torch.no_grad():
            f_prev_batch = model(inputs_prev, compute_outputs=True, compute_rates=False)
        f_prev_batch = self.parse_output(f_prev_batch).detach()  # [N, C, T_p, H_pad, W_pad]

        N, C, T_p, H_pad, W_pad = output.shape
        if T_p != 1:
            # Temporal loss is only implemented for T_patch=1 (standard config).
            return torch.tensor(0.0, device=device)

        temp_losses: list[torch.Tensor] = []

        for n in range(N):
            if not valid_mask[n]:
                continue

            t = t_vals[n].item()

            # --- Flow ---
            flow_orig = self._flow_cache[t - 1].to(device)  # [2, H_orig, W_orig]
            H_orig, W_orig = flow_orig.shape[-2], flow_orig.shape[-1]
            if (H_orig, W_orig) != (H_pad, W_pad):
                # Upsample flow and rescale magnitudes proportionally
                scale_h = H_pad / H_orig
                scale_w = W_pad / W_orig
                flow_up = F.interpolate(
                    flow_orig.unsqueeze(0), size=(H_pad, W_pad),
                    mode='bilinear', align_corners=True,
                ).squeeze(0)
                flow_up[0] *= scale_w   # dx
                flow_up[1] *= scale_h   # dy
            else:
                flow_up = flow_orig

            # --- w scalar ---
            w_t = self._w_cache[t - 1].to(device)  # scalar

            # --- sigma map (from saliency cache) ---
            if self._saliency_cache is not None:
                s = self._saliency_cache[t].to(device)  # [1, h_s, w_s]
                s = F.interpolate(
                    s.unsqueeze(0), size=(H_pad, W_pad),
                    mode='bilinear', antialias=False,
                ).squeeze(0)  # [1, H_pad, W_pad]
                s_mean = s.mean()
                p = 0.5 + (1 - 0.5) * s / (s_mean + 1e-8)
                sigma_t = 16.0 * 0.5 / p   # [1, H_pad, W_pad]
            else:
                sigma_t = torch.ones(1, H_pad, W_pad, device=device)

            # --- Warp and loss ---
            f_t = output[n, :, 0]         # [C, H_pad, W_pad]
            f_p = f_prev_batch[n, :, 0]   # [C, H_pad, W_pad]

            from losses import flow_warp
            warped = flow_warp(f_p.unsqueeze(0), flow_up.unsqueeze(0)).squeeze(0)  # [C, H_pad, W_pad]

            diff = sigma_t * w_t * (f_t - warped)   # [C, H_pad, W_pad]
            temp_losses.append(diff.pow(2).mean())

        if not temp_losses:
            return torch.tensor(0.0, device=device)

        return self.temp_weight * torch.stack(temp_losses).mean()
```

- [ ] **Step 3: Run the tests**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
import tests.test_temporal_loss as t
t.test_compute_temp_loss_returns_zero_when_no_cache()
t.test_compute_temp_loss_skips_t0_samples()
"
```

Expected:
```
compute_temp_loss no-cache returns 0 OK.
compute_temp_loss t=0 returns 0 OK.
```

- [ ] **Step 4: Commit**

```bash
git add NVRC/tasks.py NVRC/tests/test_temporal_loss.py
git commit -m "feat: add compute_temp_loss() to OverfitTask"
```

---

## Task 6: Extend `d_step` to call `compute_temp_loss`

**Files:**
- Modify: `NVRC/tasks.py` (`d_step` method, around line 152)

- [ ] **Step 1: Modify `d_step`**

Find the current `d_step` method:

```python
    def d_step(self, model, batch):
        inputs, target = self.parse_batch(batch)
        output = model(inputs, compute_outputs=True, compute_rates=False)
        output = self.parse_output(output)
        loss = self.compute_d_loss(
            output, target, inputs['lamb'],
            patch_coords=inputs['idx'],
            idx_max=inputs['idx_max'],
        )
        metrics = self.compute_metrics(output, target)
        return inputs, target, output, loss, metrics
```

Replace it with:

```python
    def d_step(self, model, batch):
        inputs, target = self.parse_batch(batch)
        output = model(inputs, compute_outputs=True, compute_rates=False)
        output = self.parse_output(output)
        loss = self.compute_d_loss(
            output, target, inputs['lamb'],
            patch_coords=inputs['idx'],
            idx_max=inputs['idx_max'],
        )
        if self._flow_cache is not None:
            loss = loss + self.compute_temp_loss(model, output, inputs)
        metrics = self.compute_metrics(output, target)
        return inputs, target, output, loss, metrics
```

- [ ] **Step 2: Verify the change compiles**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
from tasks import OverfitTask
import inspect
src = inspect.getsource(OverfitTask.d_step)
assert 'compute_temp_loss' in src
print('d_step wired OK.')
"
```

Expected: `d_step wired OK.`

- [ ] **Step 3: Commit**

```bash
git add NVRC/tasks.py
git commit -m "feat: wire compute_temp_loss into d_step"
```

---

## Task 7: Add `"wd-saliency-temp"` to `compute_loss` dispatcher in `losses.py`

**Files:**
- Modify: `NVRC/losses.py` (`compute_loss` function, around line 362)

The temporal penalty is computed in `d_step` directly. The `compute_loss` dispatcher only needs to handle the WD-saliency part for `"wd-saliency-temp"`.

- [ ] **Step 1: Add the branch**

Find in `compute_loss`:
```python
    elif name == "wd-saliency":
        loss = wd_saliency(x, y)
    else:
        raise ValueError
```

Replace with:
```python
    elif name == "wd-saliency":
        loss = wd_saliency(x, y)
    elif name == "wd-saliency-temp":
        loss = wd_saliency(x, y)   # temporal term added separately in d_step
    else:
        raise ValueError
```

- [ ] **Step 2: Verify**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
from losses import compute_loss
import torch
x = torch.rand(1, 3, 1, 32, 32)
y = torch.rand(1, 3, 1, 32, 32)
# Can't run full wd_saliency without saliency context, just test the dispatch reaches it
try:
    compute_loss('wd-saliency-temp', x, y)
except RuntimeError as e:
    if 'precomputed saliency context' in str(e):
        print('wd-saliency-temp dispatch OK (expected RuntimeError from missing context).')
    else:
        raise
"
```

Expected: `wd-saliency-temp dispatch OK (expected RuntimeError from missing context).`

- [ ] **Step 3: Commit**

```bash
git add NVRC/losses.py
git commit -m "feat: add wd-saliency-temp branch in compute_loss"
```

---

## Task 8: Call `precompute_temporal()` in `main_nvrc.py`

**Files:**
- Modify: `NVRC/main_nvrc.py` (after `precompute_saliency` call, around line 113)

- [ ] **Step 1: Add the call**

Find:
```python
        # Precompute saliency maps for the current group (no-op if loss != wd-saliency)
        train_task._saliency_cache = None
        train_task.precompute_saliency(train_dataset)
        eval_task._saliency_cache = train_task._saliency_cache  # share cache; same frames
```

Replace with:
```python
        # Precompute saliency maps for the current group (no-op if loss != wd-saliency*)
        train_task._saliency_cache = None
        train_task.precompute_saliency(train_dataset)
        eval_task._saliency_cache = train_task._saliency_cache  # share cache; same frames

        # Precompute optical flow + WD weights (no-op if loss != wd-saliency-temp)
        train_task._flow_cache = None
        train_task._w_cache = None
        train_task.precompute_temporal(train_dataset)
        eval_task._flow_cache = train_task._flow_cache   # share; same frames
        eval_task._w_cache = train_task._w_cache
```

- [ ] **Step 2: Verify the change compiles**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python -c "
import sys; sys.path.insert(0, '.')
import ast, pathlib
src = pathlib.Path('main_nvrc.py').read_text()
ast.parse(src)
assert 'precompute_temporal' in src
print('main_nvrc.py syntax OK, precompute_temporal present.')
"
```

Expected: `main_nvrc.py syntax OK, precompute_temporal present.`

- [ ] **Step 3: Commit**

```bash
git add NVRC/main_nvrc.py
git commit -m "feat: call precompute_temporal in main_nvrc.py training loop"
```

---

## Task 9: Config files and shell script

**Files:**
- Create: `NVRC/scripts/configs/tasks/overfit/wd-saliency-temp.yaml`
- Modify: `NVRC/scripts/train/nvrc_loss.sh`

- [ ] **Step 1: Create the task config YAML**

```bash
cat > /Users/yang/PerceptualLoss/NVRC/scripts/configs/tasks/overfit/wd-saliency-temp.yaml << 'EOF'
loss: [1.0, wd-saliency-temp]
metric: [psnr, lpips, dists]
color_space: RGB
lamb: 1.0
temp_weight: 0.1
temp_tau: 1.0
EOF
```

- [ ] **Step 2: Verify the YAML is valid**

```bash
conda run -n perceptual python -c "
import yaml
with open('/Users/yang/PerceptualLoss/NVRC/scripts/configs/tasks/overfit/wd-saliency-temp.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg)
assert cfg['loss'] == [1.0, 'wd-saliency-temp']
assert cfg['temp_weight'] == 0.1
print('YAML OK.')
"
```

- [ ] **Step 3: Add `wd-saliency-temp` to `nvrc_loss.sh`**

Find the `case "${LOSS_TYPE}"` block at line 96 of `NVRC/scripts/train/nvrc_loss.sh`:

```bash
case "${LOSS_TYPE}" in
    wd|rankdvqa|wd-saliency)
        TRAIN_TASK_CFG=scripts/configs/tasks/overfit/${LOSS_TYPE}.yaml
        EVAL_TASK_CFG=scripts/configs/tasks/overfit/${LOSS_TYPE}.yaml
        ;;
    l1_ms-ssim)
        TRAIN_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
        EVAL_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim.yaml
        ;;
    *)
        echo "Error: Unknown LOSS_TYPE '${LOSS_TYPE}'"
        echo "  Available: wd, rankdvqa, wd-saliency, l1_ms-ssim"
        exit 1
        ;;
esac
```

Replace with:

```bash
case "${LOSS_TYPE}" in
    wd|rankdvqa|wd-saliency|wd-saliency-temp)
        TRAIN_TASK_CFG=scripts/configs/tasks/overfit/${LOSS_TYPE}.yaml
        EVAL_TASK_CFG=scripts/configs/tasks/overfit/${LOSS_TYPE}.yaml
        ;;
    l1_ms-ssim)
        TRAIN_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
        EVAL_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim.yaml
        ;;
    *)
        echo "Error: Unknown LOSS_TYPE '${LOSS_TYPE}'"
        echo "  Available: wd, rankdvqa, wd-saliency, wd-saliency-temp, l1_ms-ssim"
        exit 1
        ;;
esac
```

- [ ] **Step 4: Quick sanity check on the shell script**

```bash
bash -n /Users/yang/PerceptualLoss/NVRC/scripts/train/nvrc_loss.sh
echo "Shell script syntax OK: $?"
```

Expected: `Shell script syntax OK: 0`

- [ ] **Step 5: Commit**

```bash
git add NVRC/scripts/configs/tasks/overfit/wd-saliency-temp.yaml NVRC/scripts/train/nvrc_loss.sh
git commit -m "feat: add wd-saliency-temp config and shell script support"
```

---

## Task 10: End-to-end smoke test

**Goal:** Confirm `wd-saliency-temp` runs end-to-end with no crash and the temporal loss term appears in the training log.

- [ ] **Step 1: Run a 3-epoch smoke test on one video**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual bash scripts/train/nvrc_loss.sh \
    --gpu_id 0 --vid Beauty --lamb 1.0 --scale s \
    --lr_s1 2e-3 --lr_s2 1e-4 --grad_accum 1 --batch_size 144 \
    --loss_type wd-saliency-temp 2>&1 | head -80
```

- [ ] **Step 2: Confirm precompute log lines appear**

Search the output for:
```
Precomputed temporal cache: flow=torch.Size([...
```

If the log line is missing, check that `main_nvrc.py` calls `precompute_temporal` and that the loss config is `wd-saliency-temp`.

- [ ] **Step 3: Confirm `d_loss` decreases over epochs**

In the training log, find lines like:
```
Train - Epoch N [...]    d_loss: X.XXXX
```

Verify `d_loss` at epoch 3 < epoch 1 (basic sanity that gradients flow).

- [ ] **Step 4: Run all unit tests**

```bash
cd /Users/yang/PerceptualLoss/NVRC
conda run -n perceptual python tests/test_temporal_loss.py
```

Expected: `All flow_warp tests passed.` (or equivalent for each test function).

- [ ] **Step 5: Commit smoke-test confirmation note (optional)**

If any adjustments were made during the smoke test, commit them with:
```bash
git add -p
git commit -m "fix: smoke-test adjustments for wd-saliency-temp"
```

---

## τ Calibration Note

After the first smoke-test run, inspect the logged `w_mean` value. Interpretation:
- `w_mean ≈ 1.0`: τ is too large — all frame pairs treated as static (no discrimination). Lower τ.
- `w_mean ≈ 0.0`: τ is too small — all pairs treated as scene cuts. Raise τ.
- `w_mean ≈ 0.3–0.7`: good discrimination range.

Adjust `temp_tau` in `wd-saliency-temp.yaml` accordingly and re-run.
