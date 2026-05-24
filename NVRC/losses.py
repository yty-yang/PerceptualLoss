"""
Losses
"""

from utils import *
import pytorch_msssim
from NVRC.loss_models.RankDVQA.networks import LPIPS_3D_Diff
from NVRC.loss_models.RankDVQA.STANet.networks.network import STANet
from NVRC.loss_models.RankDVQA.STANet.networks.multi_scale import Extractor
from NVRC.loss_models.RankDVQA.STANet.networks.common import ScalingLayer
from NVRC.loss_models.WassersteinDistortion.wasserstein_distortion import VGG16WassersteinDistortion
import NVRC.loss_models.EMLNETSaliency.resnet as resnet
import NVRC.loss_models.EMLNETSaliency.decoder as decoder


# Helper functions
_rankdvqa_model = None
_stanet_model = None
_extractor = None
_scaling_layer = None
_wloss = None
_saliency_model = None


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


def _get_stanet_model(device):
    global _stanet_model
    if _stanet_model is None:
        _stanet_model = STANet().to(device)
        stanet_checkpoint = torch.load(
            os.path.join(
                os.path.dirname(__file__), "loss_models", "RankDVQA", "STANet", "exp", "stanet", "stanet_epoch_20.pth"
            )
        )
        _stanet_model.load_state_dict(stanet_checkpoint)
        _stanet_model.eval()
        for p in _stanet_model.parameters():
            p.requires_grad_(False)
    return _stanet_model


def _get_extractor(device):
    global _extractor
    if _extractor is None:
        _extractor = Extractor().to(device)
        rankdvqa_checkpoint = torch.load(
            os.path.join(
                os.path.dirname(__file__), "loss_models", "RankDVQA", "models", "FR_model"
            )
        )
        extractor_state_dict = {
            k.replace('net.moduleExtractor.', ''): v
            for k, v in rankdvqa_checkpoint.items()
            if 'net.moduleExtractor' in k
        }
        _extractor.load_state_dict(extractor_state_dict, strict=False)
        _extractor.eval()
        for p in _extractor.parameters():
            p.requires_grad_(False)
    return _extractor


def _get_scaling_layer(device):
    global _scaling_layer
    if _scaling_layer is None:
        _scaling_layer = ScalingLayer()
        _scaling_layer.to(device)
    return _scaling_layer



def _get_wloss(device):
    global _wloss
    if _wloss is None:
        _wloss = VGG16WassersteinDistortion().to(device)
    return _wloss


def _get_saliency_model(device):
    global _saliency_model
    if _saliency_model is None:
        _saliency_model = SaliencyEMLNET(device=device)
    return _saliency_model


def _compute_stanet_score(x, y, model, stanet, extractor, scaling_layer, N, C, T, H, W, WP, HP, patch_size):
    """
    Compute STANet score for a single window of T frames.

    x, y: [N, C, T, H, W] where T=16 and H,W >= 256*10, 256*9
    Returns: normalized loss [N, 1]
    """
    # Check spatial dimensions (must be at least 256*10 x 256*9)
    required_H = patch_size * WP
    required_W = patch_size * HP

    if H < required_H or W < required_W:
        # Fallback to old method if spatial dimensions insufficient
        x_swin = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
        y_swin = y.permute(0, 2, 1, 3, 4).contiguous()
        score = model(x_swin, y_swin)  # (N, 1, 1, 1)
        score = score.squeeze(-1).squeeze(-1).squeeze(-1)  # (N,)
        return ((100 - score * 10) / 100).unsqueeze(1)  # Convert to loss

    # Process patches: for each frame, extract WP×HP patches of size patch_size×patch_size
    # STANet expects 10×9 spatial grid with 16 frames → 1440 patches total

    # Transpose to (N, T, C, H, W) for easier slicing
    x_perm = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
    y_perm = y.permute(0, 2, 1, 3, 4).contiguous()

    all_scores = []
    all_features = []

    # Iterate through spatial grid
    for i in range(WP):
        for j in range(HP):
            # Extract patch from each frame
            h_start = i * patch_size
            w_start = j * patch_size

            x_patch = x_perm[:, :, :, h_start:h_start+patch_size, w_start:w_start+patch_size]  # (N, T, C, 256, 256)
            y_patch = y_perm[:, :, :, h_start:h_start+patch_size, w_start:w_start+patch_size]

            # Stage 1: LPIPS_3D_Diff for quality score
            # Input to LPIPS_3D_Diff expects (N, V, C, H, W) where V=12
            # But STANet expects V=16, so we use full T
            x_patch_perm = x_patch.permute(0, 1, 3, 4, 2).contiguous()  # (N, T, 256, 256, C)
            y_patch_perm = y_patch.permute(0, 1, 3, 4, 2).contiguous()
            x_patch_5d = x_patch.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W)
            y_patch_5d = y_patch.permute(0, 2, 1, 3, 4).contiguous()

            # Get Stage 1 score
            with torch.no_grad():
                score = model(x_patch_5d, y_patch_5d)  # (N, 1, 1, 1)
                score = score.squeeze(-1).squeeze(-1).squeeze(-1).data.cpu().numpy().flatten() / 10  # to ~0-100 range
            all_scores.append(torch.tensor(score, device=x.device).unsqueeze(0))  # (1, N)

            # Get Stage 1 features for STANet
            # Scale input and extract features
            x_scaled = scaling_layer(x_patch)  # Normalize to [-1, 1]
            B, V, C_p, H_p, W_p = x_scaled.shape
            x_scaled_flat = x_scaled.view(B * V, C_p, H_p, W_p)  # (N*V, C, 256, 256)

            with torch.no_grad():
                features = extractor(x_scaled_flat)  # List of 6 feature maps

                # Feature combine as in dataset.py:
                # concat([interpolate(Featuremap[2], 4x4), Featuremap[5]]) → 256-ch
                feat_L3 = features[2]  # Level 3
                feat_L6 = features[5]  # Level 6

                # Interpolate Level 3 to 4x4 spatial size
                feat_L3_down = F.interpolate(feat_L3, size=(4, 4), mode='bilinear', align_corners=True)
                feat_combined = torch.cat([feat_L3_down, feat_L6], dim=1)  # (N*V, 256, 4, 4)

                # Reorganize to match STANet input format
                feat_combined = feat_combined.view(B, V, 256, 4, 4)  # (N, V, 256, 4, 4)

            all_features.append(feat_combined)

    # Stack all patches: (WP*HP, N, V, 256, 4, 4) → reorganize to STANet format
    all_scores = torch.stack(all_scores, dim=0).squeeze(-1).transpose(0, 1)  # (N, 1440)
    all_features = torch.stack(all_features, dim=0)  # (1440, N, V, 256, 4, 4)

    # Transpose to (N, 1440, ...)
    all_features = all_features.transpose(0, 1)  # (N, 1440, V, 256, 4, 4)

    # Reorganize as in STANet.network.py line 52:
    # reorganized_tensor: torch.stack(combined_features).reshape(10, 9, 16, 12, 256, 4, 4)
    # input_tensor = reorganized_tensor.permute(0, 4, 1, 2, 3, 5, 6)
    # combined_features_reshaped = input_tensor.reshape(10, 256, 16, 9, 4*4*12)

    B = N
    # all_features: (N, 1440, 16, 256, 4, 4) where 1440 = 10*9*16
    # But we need to reorganize to (10, 9, 16, 12, 256, 4, 4) -> (10, 256, 16, 9, 48)

    all_features_flat = all_features.reshape(B, WP, HP, T, 256, 4, 4)  # (N, 10, 9, 16, 256, 4, 4)
    all_features_perm = all_features_flat.permute(0, 4, 1, 2, 3, 5, 6)  # (N, 256, 10, 9, 16, 4, 4)
    all_features_reshaped = all_features_perm.reshape(B, 256, 10, 16, 9, 4*4*12)  # (N, 256, 10, 16, 9, 48)
    all_features_final = all_features_reshaped.permute(0, 1, 2, 4, 3, 5)  # (N, 256, 10, 9, 16, 48)

    # patch_quality_indices: (N, 1, 10, 9, 16) after interpolate
    patch_scores = all_scores.reshape(B, 1, WP, HP, T)  # (N, 1, 10, 9, 16)

    # Forward through STANet
    with torch.no_grad():
        stanet_output = stanet(patch_scores, all_features_final)  # (N, 1)

    # Convert to loss: (100 - score) / 100
    stanet_output = stanet_output.squeeze(1)  # (N,)
    loss = (100 - stanet_output) / 100

    return loss.unsqueeze(1)  # (N, 1)


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
    Compute RANKDVQA loss using Stage 1 (PQANet) + Stage 2 (STANet).
    Stage 2 outputs 0-100 quality score (higher is better).
    Loss = (100 - score) / 100, normalized to [0, 1], lower is better.

    x: original frames [N, C, T, H, W]
    y: reconstructed frames [N, C, T, H, W]

    Stage 2 expects grid [10, 9, 16] = 1440 patches per window.
    Each patch is 256x256 spatially, with 16 frames per window.
    """
    model = _get_rankdvqa_model(x[0].device)
    stanet = _get_stanet_model(x[0].device)
    extractor = _get_extractor(x[0].device)
    scaling_layer = _get_scaling_layer(x[0].device)

    N, C, T, H, W = x.shape

    # STANet expects 16 frames per window (TP=16), 10x9 spatial patches (HP=9, WP=10)
    TP, WP, HP = 16, 10, 9
    patch_size = 256

    # Calculate if T fits one window or needs sliding
    if T <= TP:
        # Pad to TP frames by repeating last frame
        pad = TP - T
        x_padded = torch.cat([x, x[:, :, -1:].expand(N, C, pad, H, W)], dim=2)
        y_padded = torch.cat([y, y[:, :, -1:].expand(N, C, pad, H, W)], dim=2)
        return _compute_stanet_score(x_padded, y_padded, model, stanet, extractor, scaling_layer, N, C, TP, H, W, WP, HP, patch_size)
    else:
        # Sliding windows
        stride = TP // 2  # 50% overlap
        starts = list(range(0, T - TP + 1, stride))
        if starts[-1] + TP < T:
            starts.append(T - TP)

        window_scores = []
        for start in starts:
            x_win = x[:, :, start:start+TP]
            y_win = y[:, :, start:start+TP]
            score = _compute_stanet_score(x_win, y_win, model, stanet, extractor, scaling_layer, N, C, TP, H, W, WP, HP, patch_size)
            window_scores.append(score)  # (N, 1)

        # Average over windows, broadcast to per-frame
        final_score = torch.stack(window_scores, dim=1).mean(dim=1)  # (N,)
        return final_score.unsqueeze(1).repeat(1, T)  # (N, T)


def wd(x, y, sigma_const=8.0):
    """
    Compute the per-frame Wasserstein distance
    """
    N, _, T, H, W = x.shape
    # Create constant log2_sigma map [N, 1, H, W]
    log2_sigma = (
        torch.zeros(N, 1, H, W, device=x[0].device, dtype=x.dtype) + torch.log2(torch.tensor(sigma_const))
    )
    wloss = _get_wloss(x[0].device)
    loss = torch.empty(N, T, device=x[0].device)
    for t in range(T):
        loss[:, t] = wloss(x[:, :, t], y[:, :, t], log2_sigma)
    return loss


def wd_saliency(x, y, sigma_max=16.0, pmin=0.5):
    """
    Compute per-frame WD loss with saliency-based sigma-map.

    Args:
        x: original frame [N, C, T, H, W]
        y: reconstructed frame [N, C, T, H, W]
        sigma_max: maximal sigma value (default 16.0)
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
