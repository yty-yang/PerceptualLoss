"""
Loss helpers - internal utilities for loss computation.
"""

from utils import *
from NVRC.loss_models.RankDVQA.STANet.networks.multi_scale import Extractor
from NVRC.loss_models.RankDVQA.STANet.networks.common import ScalingLayer
from NVRC.loss_models.WassersteinDistortion.wasserstein_distortion import (
    VGG16WassersteinDistortion,
)
import NVRC.loss_models.EMLNETSaliency.resnet as resnet
import NVRC.loss_models.EMLNETSaliency.decoder as decoder

# Model singletons
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


def create_ssim_win(x, size, sigma):
    """
    Create 1-D Gaussian kernel on the target device
    Modified from: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    coords = torch.arange(size, dtype=torch.float, device=x.device)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0).repeat([x.shape[1]] + [1] * (len(x.shape) - 1))


def get_rankdvqa_model(device):
    global _rankdvqa_model
    if _rankdvqa_model is None:
        from NVRC.loss_models.RankDVQA.networks import LPIPS_3D_Diff

        _rankdvqa_model = LPIPS_3D_Diff(net="multiscale_v33").to(device)
        checkpoint = torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "loss_models",
                "RankDVQA",
                "models",
                "FR_model",
            ),
            weights_only=False,
        )
        _rankdvqa_model.load_state_dict(checkpoint["model_state_dict"])
        _rankdvqa_model.eval()
        for p in _rankdvqa_model.parameters():
            p.requires_grad_(False)
    return _rankdvqa_model


def get_stanet_model(device):
    global _stanet_model
    if _stanet_model is None:
        from NVRC.loss_models.RankDVQA.STANet.networks.network import STANet

        _stanet_model = STANet().to(device)
        stanet_checkpoint = torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "loss_models",
                "RankDVQA",
                "STANet",
                "exp",
                "stanet",
                "stanet_epoch_20.pth",
            ),
            weights_only=False,
        )
        _stanet_model.load_state_dict(stanet_checkpoint)
        _stanet_model.eval()
        for p in _stanet_model.parameters():
            p.requires_grad_(False)
    return _stanet_model


def get_extractor(device):
    global _extractor
    if _extractor is None:
        _extractor = Extractor().to(device)
        rankdvqa_checkpoint = torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "loss_models",
                "RankDVQA",
                "models",
                "FR_model",
            ),
            weights_only=False,
        )
        extractor_state_dict = {
            k.replace("net.moduleExtractor.", ""): v
            for k, v in rankdvqa_checkpoint["model_state_dict"].items()
            if "net.moduleExtractor" in k
        }
        _extractor.load_state_dict(extractor_state_dict, strict=False)
        _extractor.eval()
        for p in _extractor.parameters():
            p.requires_grad_(False)
    return _extractor


def get_scaling_layer(device):
    global _scaling_layer
    if _scaling_layer is None:
        _scaling_layer = ScalingLayer()
        _scaling_layer.to(device)
    return _scaling_layer


def get_wloss(device):
    global _wloss
    if _wloss is None:
        _wloss = VGG16WassersteinDistortion().to(device)
    return _wloss


def get_saliency_model(device):
    global _saliency_model
    if _saliency_model is None:
        _saliency_model = SaliencyEMLNET(device=device)
    return _saliency_model


def compute_stanet_score(x_single, y_single, model, stanet, extractor, scaling_layer):
    """
    Compute STANet quality score for a single video clip.

    Replicates the data preparation in RankDVQA/STANet/data/dataset_test.py:
    - 12 frames per temporal window (non-overlapping)
    - 256x256 spatial patches with stride 110 (width) and 103 (height)
    - STANet requires exactly 1440 = 10x9x16 patches; tiles patches if fewer are available

    x_single: original frames [1, C, T, H, W] in [0, 1]
    y_single: reconstructed (distorted) frames [1, C, T, H, W] in [0, 1]
    Returns: scalar quality score (higher = better quality, range depends on patch scores)
    """
    FRAMES_PER_WIN = 12  # V: frames per temporal window (hardcoded in SwinDiffTiny)
    PATCH_SIZE = 256
    H_STRIDE = 103  # height patch stride (from dataset)
    W_STRIDE = 110  # width patch stride (from dataset)
    N_TARGET = 10 * 9 * 16  # 1440: STANet's hardcoded requirement

    C, T, H, W = (
        x_single.shape[1],
        x_single.shape[2],
        x_single.shape[3],
        x_single.shape[4],
    )
    device = x_single.device

    # Ensure 3 channels (Extractor and LPIPS_3D_Diff require 3-channel input)
    if C == 1:
        x_single = x_single.expand(-1, 3, -1, -1, -1).contiguous()
        y_single = y_single.expand(-1, 3, -1, -1, -1).contiguous()
        C = 3
    elif C > 3:
        x_single = x_single[:, :3].contiguous()
        y_single = y_single[:, :3].contiguous()
        C = 3

    # Transpose to [1, T, C, H, W] for temporal/spatial slicing
    xp = x_single.permute(0, 2, 1, 3, 4)
    yp = y_single.permute(0, 2, 1, 3, 4)

    # Ensure spatial dims >= 256 by upsampling if needed
    if H < PATCH_SIZE or W < PATCH_SIZE:
        new_H, new_W = max(H, PATCH_SIZE), max(W, PATCH_SIZE)
        xp = F.interpolate(
            xp.reshape(T, C, H, W),
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        ).view(1, T, C, new_H, new_W)
        yp = F.interpolate(
            yp.reshape(T, C, H, W),
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        ).view(1, T, C, new_H, new_W)
        H, W = new_H, new_W

    # Temporal windows: non-overlapping, 12 frames each (matches dataset)
    tw_starts = list(range(0, T - FRAMES_PER_WIN + 1, FRAMES_PER_WIN))
    if len(tw_starts) == 0:
        # T < 12: pad to 12 frames by repeating the last frame
        pad = FRAMES_PER_WIN - T
        xp = torch.cat([xp, xp[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        yp = torch.cat([yp, yp[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
        tw_starts = [0]

    # Spatial patches (same stride as dataset: width stride 110, height stride 103)
    w_starts = list(range(0, W - PATCH_SIZE + 1, W_STRIDE)) or [0]
    h_starts = list(range(0, H - PATCH_SIZE + 1, H_STRIDE)) or [0]

    all_scores = []  # list of scalar tensors (LPIPS/10 per patch)
    all_features = []  # list of (12, 256, 4, 4) tensors (extractor features per patch)

    # Loop order matches dataset: temporal × width × height
    for tw_start in tw_starts:
        xw = xp[:, tw_start : tw_start + FRAMES_PER_WIN]  # [1, 12, C, H, W]
        yw = yp[:, tw_start : tw_start + FRAMES_PER_WIN]

        for w_start in w_starts:
            for h_start in h_starts:
                xpatch = xw[
                    :,
                    :,
                    :,
                    h_start : h_start + PATCH_SIZE,
                    w_start : w_start + PATCH_SIZE,
                ]
                ypatch = yw[
                    :,
                    :,
                    :,
                    h_start : h_start + PATCH_SIZE,
                    w_start : w_start + PATCH_SIZE,
                ]
                # both: [1, 12, 3, 256, 256] in [0, 1]

                # Stage 1: LPIPS_3D_Diff score.
                # model.forward expects (B, V, C, H, W) in [-1,1]; normalize=True converts [0,1]→[-1,1].
                score = model(xpatch, ypatch, normalize=True)  # [1, 1, 1, 1]
                all_scores.append(score.squeeze() / 10)  # scalar tensor

                # Stage 2 features: extractor uses distorted (y) video only, as in dataset.
                # Dataset applies scaling_layer to [-1,1] input → we convert [0,1]→[-1,1] first.
                y_flat = ypatch.view(
                    FRAMES_PER_WIN, C, PATCH_SIZE, PATCH_SIZE
                )  # [12, 3, 256, 256]
                y_scaled = scaling_layer(2.0 * y_flat - 1.0)  # ImageNet normalization
                feats = extractor(y_scaled)
                # Level 3 (index 2): [12, 64, 32, 32] → downsample to [12, 64, 4, 4]
                feat_l3 = F.interpolate(
                    feats[2], size=(4, 4), mode="bilinear", align_corners=True
                )
                feat_l6 = feats[5]  # [12, 192, 4, 4]
                all_features.append(
                    torch.cat([feat_l3, feat_l6], dim=1)
                )  # [12, 256, 4, 4]

    # Tile patches to exactly N_TARGET = 1440 (STANet hardcodes this shape)
    n = len(all_scores)
    if n < N_TARGET:
        repeats = (N_TARGET + n - 1) // n
        all_scores = (all_scores * repeats)[:N_TARGET]
        all_features = (all_features * repeats)[:N_TARGET]
    else:
        all_scores = all_scores[:N_TARGET]
        all_features = all_features[:N_TARGET]

    # scores_tensor: [1440] — STANet views this as (1, 1, 10, 9, 16) inside forward()
    scores_tensor = torch.stack(all_scores)

    # STANet forward: torch.stack(combined_features) → (1440, 12, 256, 4, 4)
    quality_score = stanet(scores_tensor, all_features)  # [1]
    return quality_score.squeeze()  # scalar


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
        with torch.no_grad(), torch.autocast(device_type=x.device.type):
            img_feat = self.img_model(x_resized, decode=True)
            pla_feat = self.pla_model(x_resized, decode=True)
            pred = self.decoder_model([img_feat, pla_feat])
        # Return at decoder output resolution; caller upsamples after sigma computation
        return pred.float()
