"""
Losses
"""

from utils import *
import pytorch_msssim
from NVRC.loss_models.RankDVQA.networks import LPIPS_3D_Diff
from NVRC.loss_models.WassersteinDistortion.wasserstein_distortion import VGG16WassersteinDistortion
import NVRC.loss_models.EMLNETSaliency.resnet as resnet
import NVRC.loss_models.EMLNETSaliency.decoder as decoder


# Helper functions
def check_shape(x, y):
    assert x.shape == y.shape, "shape of tensors must be the same!"
    assert x.ndim == y.ndim == 5, "inputs are expected to have 5D ([N, C, T, H, W])"


def yuv444to420(x):
    assert x.shape[3] % 2 == x.shape[4] % 2 == 0, "height and width must be even"
    assert x.shape[1] == 3, "inputs are expected to have 3 channels"
    N, C, T, H, W = x.shape
    x_down = F.avg_pool2d(x.view(N, C * T, H, W), kernel_size=2, stride=2).view(
        N, C, T, H // 2, W // 2
    )
    return x[:, 0:1], x_down[:, 1:2], x_down[:, 2:3]


def _create_ssim_win(x, size, sigma):
    """
    Create 1-D Gaussian kernel on the target device
    Modified from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    coords = torch.arange(size, dtype=torch.float, device=x.device)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0).repeat([x.shape[1]] + [1] * (len(x.shape) - 1))


_rankdvqa_model = None


def _get_rankdvqa_model(device):
    global _rankdvqa_model
    if _rankdvqa_model is None:
        _rankdvqa_model = LPIPS_3D_Diff(net="multiscale_v33").to(device)
        checkpoint = torch.load(
            os.path.join(
                os.path.dirname(__file__), "loss_models", "RankDVQA", "models", "FR_model"
            )
        )
        _rankdvqa_model.load_state_dict(checkpoint["model_state_dict"])
        _rankdvqa_model.eval()
        for p in _rankdvqa_model.parameters():
            p.requires_grad_(False)
    return _rankdvqa_model


_wloss = None


def _get_wloss(device):
    global _wloss
    if _wloss is None:
        _wloss = VGG16WassersteinDistortion().to(device)
    return _wloss


# EMLNETSaliency wrapper for arbitrary input sizes
class SaliencyEMLNET:
    """EMLNETSaliency model wrapper that handles arbitrary input sizes."""

    _instance = None

    def __init__(self, backbone_dir=None, device="cuda"):
        if backbone_dir is None:
            backbone_dir = os.path.join(
                os.path.dirname(__file__), "loss_models", "EMLNETSaliency", "backbone"
            )
        self.device = device
        self.size = (480, 640)
        self.num_feat = 5
        self._load_models(backbone_dir)

    def _load_models(self, backbone_dir):
        img_model_path = os.path.join(backbone_dir, "res_imagenet.pth")
        pla_model_path = os.path.join(backbone_dir, "res_places.pth")
        dec_model_path = os.path.join(backbone_dir, "res_decoder.pth")
        self.img_model = resnet.resnet50(img_model_path).to(self.device).eval()
        self.pla_model = resnet.resnet50(pla_model_path).to(self.device).eval()
        self.decoder_model = (
            decoder.build_decoder(
                dec_model_path, self.size, self.num_feat, self.num_feat
            )
            .to(self.device)
            .eval()
        )
        for p in self.img_model.parameters():
            p.requires_grad_(False)
        for p in self.pla_model.parameters():
            p.requires_grad_(False)
        for p in self.decoder_model.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        """
        Args:
            x: Tensor [N, C, H, W], values in [0, 1]
        Returns:
            Tensor [N, 1, H, W], saliency map in [0, 1]
        """
        _, _, H, W = x.shape
        # Resize to model input size
        x_resized = F.interpolate(x, size=self.size, mode="bilinear", antialias=True)
        with torch.no_grad():
            img_feat = self.img_model(x_resized, decode=True)
            pla_feat = self.pla_model(x_resized, decode=True)
            pred = self.decoder_model([img_feat, pla_feat])
            # Resize back to original size
            saliency = F.interpolate(pred, size=(H, W), mode="bilinear", antialias=True)
        return saliency


_saliency_model = None


def _get_saliency_model(device):
    global _saliency_model
    if _saliency_model is None:
        _saliency_model = SaliencyEMLNET(device=device)
    return _saliency_model


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
    win = _create_ssim_win(x, win_size, 1.5)
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
    win = _create_ssim_win(x, win_size, 1.5)
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
    Compute RANKDVQA loss using sliding window (V=12 frames per forward pass)
    x: original frames [N, C, T, H, W]
    y: reconstructed frames [N, C, T, H, W]
    """
    model = _get_rankdvqa_model(x[0].device)
    N, C, T, H, W = x.shape
    V = 12  # RankDVQA model expects V=12

    if T <= V:
        # Pad to V by repeating last frame
        pad = V - T
        x_padded = torch.cat([x, x[:, :, -1:].expand(N, C, pad, H, W)], dim=2)
        y_padded = torch.cat([y, y[:, :, -1:].expand(N, C, pad, H, W)], dim=2)
        x_swin = x_padded.permute(0, 2, 1, 3, 4).contiguous()  # (N, V, C, H, W)
        y_swin = y_padded.permute(0, 2, 1, 3, 4).contiguous()
        score = model(x_swin, y_swin)  # (N, 1, 1, 1)
        return score.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1).repeat(1, T)  # (N, T)
    else:
        # Sliding windows to cover T frames
        stride = max(1, (T - V) // max(1, (T - V) // (V // 2)))
        starts = list(range(0, T - V + 1, stride))
        if starts[-1] + V < T:
            starts.append(T - V)

        x_swin = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
        y_swin = y.permute(0, 2, 1, 3, 4).contiguous()

        scores = []
        for start in starts:
            seg_x = x_swin[:, start:start+V]  # (N, V, C, H, W)
            seg_y = y_swin[:, start:start+V]
            score = model(seg_x, seg_y)  # (N, 1, 1, 1)
            scores.append(score.squeeze(-1).squeeze(-1).squeeze(-1))  # (N,)

        # Average over windows, then broadcast back to per-frame
        final_score = torch.stack(scores, dim=1).mean(dim=1)  # (N,)
        return final_score.unsqueeze(1).repeat(1, T)  # (N, T) — will be broadcast in compute_loss


def wd(x, y, log2_sigma_const=2.0):
    """
    Compute the per-frame Wasserstein distance
    """
    N, _, T, H, W = x.shape
    # Create constant log2_sigma map [N, 1, H, W]
    log2_sigma = (
        torch.zeros(N, 1, H, W, device=x[0].device, dtype=x.dtype) + log2_sigma_const
    )
    wloss = _get_wloss(x[0].device)
    loss = torch.empty(N, T, device=x[0].device)
    for t in range(T):
        loss[:, t] = wloss(x[:, :, t], y[:, :, t], log2_sigma)
    return loss


def wd_saliency(x, y, sigma_max=5.0, pmin=0.5):
    """
    Compute per-frame WD loss with saliency-based sigma-map.

    Args:
        x: original frame [N, C, T, H, W]
        y: reconstructed frame [N, C, T, H, W]
        sigma_max: maximal sigma value (default 5.0)
        pmin: lower bound for density p (default 0.5)

    Saliency → sigma map conversion (from paper):
        p = pmin + (1 - pmin) · s / s̄
        sigma = sigma_max · pmin / p
    where s is the saliency map and s̄ is its spatial mean.
    """
    N, _, T, _, _ = x.shape
    saliency_model = _get_saliency_model(x[0].device)
    wloss = _get_wloss(x[0].device)
    loss = torch.empty(N, T, device=x[0].device)

    for t in range(T):
        frame = x[:, :, t]  # [N, C, H, W]
        with torch.no_grad():
            s = saliency_model(frame)  # [N, 1, H, W], in [0, 1]
        s_mean = s.mean()  # spatial mean s̄
        # Eq (3): p = pmin + (1 - pmin) * s / s_mean
        p = pmin + (1 - pmin) * s / (s_mean + 1e-8)
        # Eq (4): sigma = sigma_max * pmin / p
        sigma = sigma_max * pmin / p
        log2_sigma = torch.log2(sigma)
        loss[:, t] = wloss(frame, y[:, :, t], log2_sigma)
    return loss


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

    if name == "mse":
        metric = mse(x, y)
    elif name == "l1":
        metric = l1(x, y)
    elif name == "psnr":
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
    elif name == "rankdvqa":
        metric = rankdvqa(x, y)
    elif name == "wd":
        metric = wd(x, y)
    elif name == "wd-saliency":
        metric = wd_saliency(x, y)
    else:
        raise ValueError
    assert metric.ndim == 2, "metric is expected to have 2D ([N, T])"
    return metric
