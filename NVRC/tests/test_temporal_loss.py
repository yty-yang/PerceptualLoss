import sys, os
_NVRC_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_ROOT_DIR = os.path.abspath(os.path.dirname(_NVRC_DIR))
for _p in (_NVRC_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
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
    frame = torch.zeros(1, 1, H, W)
    for x in range(W):
        frame[0, 0, :, x] = float(x)

    flow = torch.zeros(1, 2, H, W)
    flow[0, 0] = 1.0   # dx = +1

    warped = flow_warp(frame, flow)
    for x in range(1, W - 1):
        expected = float(x - 1)
        got = warped[0, 0, H // 2, x].item()
        assert abs(got - expected) < 1e-4, f"col {x}: expected {expected}, got {got}"


def test_flow_warp_output_shape():
    for N, C, H, W in [(1, 3, 32, 32), (4, 1, 48, 64)]:
        frame = torch.rand(N, C, H, W)
        flow = torch.zeros(N, 2, H, W)
        assert flow_warp(frame, flow).shape == (N, C, H, W)


def test_raft_model_output_shape():
    """RAFT should return a list whose last entry has shape [N, 2, H, W]."""
    import os, sys
    _NVRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _ROOT_DIR = os.path.abspath(os.path.join(_NVRC_DIR, '..'))
    for _p in (_NVRC_DIR, _ROOT_DIR):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from NVRC.loss_utils import get_raft_model
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


def test_compute_temp_loss_returns_zero_when_no_cache():
    """compute_temp_loss should return 0 tensor when _flow_cache is None."""
    import sys, os
    _NVRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _ROOT_DIR = os.path.abspath(os.path.join(_NVRC_DIR, '..'))
    for _p in (_NVRC_DIR, _ROOT_DIR):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from unittest.mock import MagicMock
    import torch
    from tasks import OverfitTask

    task = MagicMock()
    task._flow_cache = None
    task._w_cache = None

    output = torch.zeros(2, 3, 1, 8, 8)
    result = OverfitTask.compute_temp_loss(task, model=None, output=output,
                                           inputs={'idx': torch.zeros(2, 3, dtype=torch.long)})
    assert result.item() == 0.0
    print('compute_temp_loss no-cache returns 0 OK.')


def test_compute_temp_loss_skips_t0_samples():
    """All samples with t=0 should produce zero temporal loss."""
    import sys, os
    _NVRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _ROOT_DIR = os.path.abspath(os.path.join(_NVRC_DIR, '..'))
    for _p in (_NVRC_DIR, _ROOT_DIR):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from unittest.mock import MagicMock
    import torch
    from tasks import OverfitTask

    task = MagicMock()
    task._flow_cache = torch.zeros(1, 2, 8, 8)
    task._w_cache = torch.ones(1)
    task._saliency_cache = None
    task.temp_weight = 0.1
    task.parse_output = lambda x: x

    output = torch.rand(3, 3, 1, 8, 8, requires_grad=True)
    idx = torch.zeros(3, 3, dtype=torch.long)   # all t=0
    inputs = {
        'idx': idx, 'x': None, 'lamb': torch.tensor([1.0]),
        'vidx': torch.zeros(3, dtype=torch.int32),
        'vidx_max': 1, 'idx_max': (1, 1, 1),
        'rel_batch_size': 1.0,
        'video_size': (1, 8, 8), 'patch_size': (1, 8, 8), 'channels': 3
    }

    result = OverfitTask.compute_temp_loss(task, model=MagicMock(), output=output, inputs=inputs)
    assert result.item() == 0.0, f"Expected 0, got {result.item()}"
    print('compute_temp_loss t=0 returns 0 OK.')


if __name__ == '__main__':
    test_flow_warp_zero_flow_is_identity()
    test_flow_warp_shift_right_by_one()
    test_flow_warp_output_shape()
    test_raft_model_output_shape()
    print('All flow_warp tests passed.')
