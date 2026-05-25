"""
Losses
"""

from utils import *
from NVRC.losses_helpers import (
    check_shape,
    yuv444to420,
    create_ssim_win,
    get_rankdvqa_model,
    get_stanet_model,
    get_extractor,
    get_scaling_layer,
    get_wloss,
    get_saliency_model,
    compute_stanet_score,
)
import pytorch_msssim


# Loss functions
def mse(x, y):
    """
    Compute the per-frame MSE loss
    """
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.mse_loss(x, y, reduction="none").mean(dim=2)


def log_mse(x, y):
    """
    Compute the per-frame log MSE loss
    """
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return torch.log(F.mse_loss(x, y, reduction="none")).mean(dim=2)


def mse_yuv611_sum(x, y):
    """
    Compute the per-frame MSE loss
    """
    assert x.shape[1] == y.shape[1] == 3, "inputs are expected to have 3 channels"
    N, _, T, _, _ = x.shape
    x = [
        x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for x_i in yuv444to420(x)
    ]
    y = [
        y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for y_i in yuv444to420(y)
    ]
    yuv_weight = torch.tensor(
        [6.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0], device=x[0].device
    ).view(1, 1, 3)
    yuv_loss = torch.concat(
        [F.mse_loss(x_i, y_i, reduction="none").mean(dim=2) for x_i, y_i in zip(x, y)],
        dim=2,
    )
    return (yuv_weight * yuv_loss).sum(dim=2)


def mse_yuv611_product(x, y):
    """
    Compute the per-frame MSE loss
    """
    assert x.shape[1] == y.shape[1] == 3, "inputs are expected to have 3 channels"
    N, _, T, _, _ = x.shape
    x = [
        x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for x_i in yuv444to420(x)
    ]
    y = [
        y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for y_i in yuv444to420(y)
    ]
    yuv_weight = torch.tensor(
        [6.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0], device=x[0].device
    ).view(1, 1, 3)
    yuv_loss = torch.concat(
        [F.mse_loss(x_i, y_i, reduction="none").mean(dim=2) for x_i, y_i in zip(x, y)],
        dim=2,
    )
    return (yuv_loss**yuv_weight).prod(dim=2)


def l1(x, y):
    """
    Compute the per-frame L1 loss
    """
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    y = y.permute(0, 2, 3, 4, 1).contiguous().view(N, T, H * W * C)
    return F.l1_loss(x, y, reduction="none").mean(dim=2)


def l1_yuv611(x, y):
    """
    Compute the per-frame L1 loss
    """
    assert x.shape[1] == y.shape[1] == 3, "inputs are expected to have 3 channels"
    N, _, T, _, _ = x.shape
    x = [
        x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for x_i in yuv444to420(x)
    ]
    y = [
        y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for y_i in yuv444to420(y)
    ]
    yuv_weight = torch.tensor(
        [6.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0], device=x[0].device
    ).view(1, 1, 3)
    yuv_loss = torch.concat(
        [F.l1_loss(x_i, y_i, reduction="none").mean(dim=2) for x_i, y_i in zip(x, y)],
        dim=2,
    )
    return (yuv_weight * yuv_loss).sum(dim=2)


def psnr(x, y, v_max=1.0):
    """
    Compute the per-frame PSNR
    """
    return 10 * torch.log10((v_max**2) / (mse(x, y) + 1e-9))


def psnr_yuv611(x, y, v_max=1.0):
    """
    Compute the per-frame PSNR
    """
    assert x.shape[1] == y.shape[1] == 3, "inputs are expected to have 3 channels"
    N, _, T, _, _ = x.shape
    x = [
        x_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for x_i in yuv444to420(x)
    ]
    y = [
        y_i.permute(0, 2, 3, 4, 1).contiguous().view(N, T, -1, 1)
        for y_i in yuv444to420(y)
    ]
    yuv_weight = torch.tensor(
        [6.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0], device=x[0].device
    ).view(1, 1, 3)
    yuv_mse = torch.concat(
        [F.mse_loss(x_i, y_i, reduction="none").mean(dim=2) for x_i, y_i in zip(x, y)],
        dim=2,
    )
    return ((10 * torch.log10((v_max**2) / (yuv_mse + 1e-9))) * yuv_weight).sum(dim=2)


def ssim(x, y, v_max=1.0, win_size=11, win_sigma=1.5):
    """
    Compute the per-frame SSIM
    """
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    win = create_ssim_win(x, win_size, 1.5)
    return pytorch_msssim.ssim(
        x, y, v_max, win_size=win_size, win_sigma=win_sigma, win=win, size_average=False
    ).view(N, T)


def ms_ssim(x, y, v_max=1.0, win_size=11, win_sigma=1.5):
    """
    Compute the per-frame MS-SSIM
    """
    N, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    y = y.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    win = create_ssim_win(x, win_size, 1.5)
    return pytorch_msssim.ms_ssim(
        x, y, v_max, win_size=win_size, win_sigma=win_sigma, win=win, size_average=False
    ).view(N, T)


def ssim_y(x, y, v_max=1.0, win_size=11):
    """
    Compute the per-frame SSIM for Y channel
    """
    assert x.shape[1] == y.shape[1] == 3, "inputs are expected to have 3 channels"
    return ssim(x[:, 0:1], y[:, 0:1], v_max, win_size)


def ms_ssim_y(x, y, v_max=1.0, win_size=11):
    """
    Compute the per-frame MS-SSIM for Y channel
    """
    assert x.shape[1] == y.shape[1] == 3, "inputs are expected to have 3 channels"
    return ms_ssim(x[:, 0:1], y[:, 0:1], v_max, win_size)


def rankdvqa(x, y):
    """
    Compute RankDVQA loss using Stage 1 (LPIPS_3D_Diff) + Stage 2 (STANet).

    STANet outputs a perceptual quality score derived from weighted patch scores.
    """
    model = get_rankdvqa_model(x.device)
    stanet = get_stanet_model(x.device)
    extractor = get_extractor(x.device)
    scaling_layer = get_scaling_layer(x.device)

    N, _, T, _, _ = x.shape
    per_sample = []

    for n in range(N):
        quality = compute_stanet_score(
            y[n : n + 1], x[n : n + 1], model, stanet, extractor, scaling_layer
        )
        # quality is the STANet quality score (higher = better quality).
        # Sigmoid bounds to (0, 1), negate so minimizing loss = maximizing quality.
        # unsqueeze then expand keeps the autograd graph intact (no in-place ops).
        per_sample.append(-quality.unsqueeze(0).expand(T))

    return torch.stack(per_sample, dim=0)  # (N, T)


def wd(x, y, sigma_const=8.0, scale=0.02):
    """
    Compute the per-frame Wasserstein distance
    """
    N, _, T, H, W = x.shape
    # Create constant log2_sigma map [N, 1, H, W]
    log2_sigma = torch.zeros(
        N, 1, H, W, device=x[0].device, dtype=x.dtype
    ) + torch.log2(torch.tensor(sigma_const))
    wloss = get_wloss(x[0].device)
    loss = torch.empty(N, T, device=x[0].device)
    for t in range(T):
        loss[:, t] = wloss(x[:, :, t], y[:, :, t], log2_sigma)
    return loss * scale


def wd_saliency(x, y, sigma_max=16.0, pmin=0.5, scale=0.02):
    """
    Compute per-frame WD loss with saliency-based sigma-map.

    Args:
        x: original frame [N, C, T, H, W]
        y: reconstructed frame [N, C, T, H, W]
        sigma_max: maximal sigma value (default 16.0)
        pmin: lower bound for density p (default 0.5)
        scale: scaling factor for the final loss (default 0.02)

    Saliency → sigma map conversion (from paper):
        p = pmin + (1 - pmin) · s / s̄
        sigma = sigma_max · pmin / p
    where s is the saliency map and s̄ is its spatial mean.
    """
    N, C, T, H, W = x.shape
    saliency_model = get_saliency_model(x[0].device)
    wloss = get_wloss(x[0].device)
    loss = torch.empty(N, T, device=x[0].device)

    # Batch all T frames into one forward pass: [N, C, T, H, W] → [N*T, C, H, W]
    frames_all = x.permute(0, 2, 1, 3, 4).contiguous().view(N * T, C, H, W)
    with torch.no_grad():
        s_all = saliency_model(frames_all)  # [N*T, 1, h_s, w_s]
    _, _, h_s, w_s = s_all.shape
    s_all = s_all.view(N, T, 1, h_s, w_s)  # [N, T, 1, h_s, w_s]

    for t in range(T):
        frame = x[:, :, t]  # [N, C, H, W]
        s = s_all[:, t]  # [N, 1, h_out, w_out] — decoder output resolution
        s_mean = s.mean(dim=[1, 2, 3], keepdim=True)  # [N, 1, 1, 1]
        # Eq (3): p = pmin + (1 - pmin) * s / s_mean
        p = pmin + (1 - pmin) * s / (s_mean + 1e-8)
        # Eq (4): sigma = sigma_max * pmin / p — computed at low resolution
        log2_sigma = torch.log2(sigma_max * pmin / p)
        # Upsample sigma-map to full resolution for wloss
        log2_sigma = F.interpolate(
            log2_sigma, size=(H, W), mode="bilinear", antialias=True
        )
        loss[:, t] = wloss(frame, y[:, :, t], log2_sigma)
    return loss * scale


def compute_loss(name, x, y):
    check_shape(x, y)
    x, y = x.float(), y.float()

    if name == "mse":
        loss = mse(x, y)
    elif name == "log-mse":
        loss = log_mse(x, y)
    elif name == "mse-yuv611-s":
        loss = mse_yuv611_sum(x, y)
    elif name == "mse-yuv611-p":
        loss = mse_yuv611_product(x, y)
    elif name == "l1-yuv611":
        loss = l1_yuv611(x, y)
    elif name == "l1":
        loss = l1(x, y)
    elif name == "ssim":
        loss = 1.0 - ssim(x, y)
    elif name == "ssim-5x5":
        loss = 1.0 - ssim(x, y, win_size=5)
    elif name == "ms-ssim":
        loss = 1.0 - ms_ssim(x, y)
    elif name == "ms-ssim-5x5":
        loss = 1.0 - ms_ssim(x, y, win_size=5)
    elif name == "ssim-y":
        loss = 1.0 - ssim_y(x, y)
    elif name == "ssim-y-5x5":
        loss = 1.0 - ssim_y(x, y, win_size=5)
    elif name == "ms-ssim-y":
        loss = 1.0 - ms_ssim_y(x, y)
    elif name == "ms-ssim-y-5x5":
        loss = 1.0 - ms_ssim_y(x, y, win_size=5)
    elif name == "rankdvqa":
        loss = rankdvqa(x, y)
    elif name == "wd":
        loss = wd(x, y)
    elif name == "wd-saliency":
        loss = wd_saliency(x, y)
    else:
        raise ValueError
    assert loss.ndim == 2, "loss is expected to have 2D ([N, T])"
    return loss


def compute_metric(name, x, y):
    check_shape(x, y)
    x, y = x.float(), y.float()

    if name == "psnr":
        metric = psnr(x, y)
    elif name == "psnr-yuv611":
        metric = psnr_yuv611(x, y)
    elif name == "ssim":
        metric = ssim(x, y)
    elif name == "ssim-5x5":
        metric = ssim(x, y, win_size=5)
    elif name == "ms-ssim":
        metric = ms_ssim(x, y)
    elif name == "ms-ssim-5x5":
        metric = ms_ssim(x, y, win_size=5)
    elif name == "ssim-y":
        metric = ssim_y(x, y)
    elif name == "ssim-y-5x5":
        metric = ssim_y(x, y, win_size=5)
    elif name == "ms-ssim-y":
        metric = ms_ssim_y(x, y)
    elif name == "ms-ssim-y-5x5":
        metric = ms_ssim_y(x, y, win_size=5)
    else:
        raise ValueError
    assert metric.ndim == 2, "metric is expected to have 2D ([N, T])"
    return metric
