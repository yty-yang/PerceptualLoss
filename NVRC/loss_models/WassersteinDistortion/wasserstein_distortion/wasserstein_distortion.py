# Copyright 2025 Yueyu Hu, Jona Ballé.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Implementation of Wasserstein Distortion in PyTorch."""

from typing import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models as tv

from torch import Tensor


class LowpassFilter2D(nn.Module):
    kernel: Tensor

    def __init__(self):
        super().__init__()
        kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("kernel", kernel_2d[None, None, :, :])

    @override
    def forward(self, x, stride=1):
        kernel = self.kernel.expand((x.shape[1], 1, -1, -1))
        x = F.conv2d(x, kernel, stride=stride, padding=1, groups=x.shape[1])  # pylint: disable=not-callable
        return x


class MultiLevelStats(nn.Module):
    def __init__(self, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.lowpass = LowpassFilter2D()

    @override
    def forward(self, x):
        squared = x**2
        means = []
        variances = []
        for _ in range(self.num_levels):
            m = self.lowpass(x, stride=1)
            p = self.lowpass(squared, stride=1)
            means.append(m)
            variances.append(p - m**2)
            x = m[..., ::2, ::2]
            squared = p[..., ::2, ::2]
        return means, variances


class WassersteinDistortionFeature(nn.Module):
    """Calculates the Wasserstein distortion between two feature maps."""

    def __init__(self, num_levels: int = 5):
        super().__init__()
        self.multi_level_stats = MultiLevelStats(num_levels)
        self.num_levels = num_levels
        self.lowpass = LowpassFilter2D()

    def extract_stats(
        self, features: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Compute multi-level mean/variance pyramid from a feature map.

        Separates the statistics extraction step from the comparison step so
        that callers can run the pyramid once and reuse it across multiple
        comparisons (e.g. adjacent sigma pairs in sigma-map generation).

        Returns:
            means, variances — each a list of num_levels tensors at halving
            spatial resolutions.
        """
        if features.dtype == torch.float16:
            features = features.to(torch.bfloat16)
        return self.multi_level_stats(features)

    def compare_stats(
        self,
        feat_a: Tensor,
        means_a: list[Tensor],
        vars_a: list[Tensor],
        feat_b: Tensor,
        means_b: list[Tensor],
        vars_b: list[Tensor],
        log2_sigma: Tensor,
        return_map: bool = False,
    ) -> Tensor:
        """Compare pre-extracted statistics without re-running the pyramid.

        Reuses (means, variances) from extract_stats() to skip redundant
        multi_level_stats calls — useful when the same feature map participates
        in multiple comparisons.

        Args:
            feat_a / feat_b:  raw feature tensors (needed for wd_maps[0])
            means_a/b, vars_a/b:  outputs of extract_stats for each side
            log2_sigma:  spatial sigma map [N, 1, H_feat, W_feat]
            return_map:  when True, return per-pixel map [N, 1, H_feat, W_feat]
                         instead of a scalar; contributions from coarser pyramid
                         levels are bilinearly upsampled to H_feat × W_feat.

        Returns:
            Scalar Wasserstein distortion, or per-pixel map when return_map=True.
        """
        dtype = torch.bfloat16 if feat_a.dtype == torch.float16 else feat_a.dtype
        feat_a      = feat_a.to(dtype)
        feat_b      = feat_b.to(dtype)
        log2_sigma  = log2_sigma.to(dtype)

        # Level 0: direct squared difference; levels 1..num_levels: pyramid stats
        wd_maps = [torch.square(feat_a - feat_b)]
        for ma, va, mb, vb in zip(means_a, vars_a, means_b, vars_b):
            std_a = torch.sqrt(torch.clamp(va.to(dtype), min=1e-8))
            std_b = torch.sqrt(torch.clamp(vb.to(dtype), min=1e-8))
            wd_maps.append(
                torch.square(ma.to(dtype) - mb.to(dtype)) + torch.square(std_a - std_b)
            )

        # Sigma-weighted reduction.
        # Size schedule (see MultiLevelStats):
        #   wd_maps[0]: [H, W]  (raw diff)
        #   wd_maps[1]: [H, W]  (pyramid level 0, same size)
        #   wd_maps[2]: [H/2, W/2]  (pyramid level 1)
        #   wd_maps[i≥2]: [H/2^(i-1), W/2^(i-1)]
        # log2_sigma is downsampled AFTER weights for the current iteration are
        # computed, so sizes always match between weights_i and wd_maps[i].
        if return_map:
            H0, W0 = log2_sigma.shape[-2:]
            ls      = log2_sigma
            result  = log2_sigma.new_zeros(feat_a.shape[0], 1, H0, W0)
            for i, wd_map in enumerate(wd_maps):
                w_i    = F.relu(1 - torch.abs(ls - i))
                contrib = (w_i * wd_map).mean(dim=1, keepdim=True)   # [N,1,Hi,Wi]
                if contrib.shape[-2:] != (H0, W0):
                    contrib = F.interpolate(
                        contrib.float(), size=(H0, W0), mode="bilinear", antialias=False
                    ).to(dtype)
                result += contrib
                if i > 0:
                    ls = self.lowpass(ls, stride=2)
            return result
        else:
            wasserstein_dist: Tensor | int = 0
            ls = log2_sigma
            for i, wd_map in enumerate(wd_maps):
                w_i = F.relu(1 - torch.abs(ls - i))
                if i > 0:
                    ls = self.lowpass(ls, stride=2)
                wasserstein_dist = wasserstein_dist + (w_i * wd_map).mean()
            assert isinstance(wasserstein_dist, Tensor)
            return wasserstein_dist

    @override
    def forward(
        self,
        features_a: Tensor,
        features_b: Tensor,
        log2_sigma: Tensor,
    ) -> Tensor:
        """Calculates the Wasserstein distortion between two feature maps."""
        # fp16 overflows when squaring deep VGG features; bf16 has fp32 exponent range
        if features_a.dtype == torch.float16:
            features_a = features_a.to(torch.bfloat16)
            features_b = features_b.to(torch.bfloat16)
            log2_sigma = log2_sigma.to(torch.bfloat16)
        means_a, vars_a = self.multi_level_stats(features_a)
        means_b, vars_b = self.multi_level_stats(features_b)
        return self.compare_stats(
            features_a, means_a, vars_a,
            features_b, means_b, vars_b,
            log2_sigma, return_map=False,
        )


# pyright: reportIndexIssue=false
class MultiscaleTruncatedVGG16(nn.Module):
    """
    A VGG module that supports executing only the first few blocks
    (i.e. truncated) for computation saving. It supports multiscale
    feature extraction, where the input image is downsampled to
    different resolutions and processed through the VGG network.
    """
    mean: Tensor
    std: Tensor

    def __init__(
        self,
        requires_grad=False,
        pretrained=True,
        truncate_slice=5,
        replace_with_avg_pooling=True,
    ):
        """Initialize the MultiscaleTruncatedVGG module.
        The JAX version replaces the max pooling layers with average pooling, so
        this option is available here as well.
        """
        super().__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.num_slices = 5
        self.truncate_slice = truncate_slice
        if not 1 <= truncate_slice <= self.num_slices:
            raise ValueError(
                f"truncate_slice must be between 1 and {self.num_slices}, inclusive, "
                f"but is {truncate_slice}."
            )

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 2:
            for x in range(4, 9):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice2.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 3:
            for x in range(9, 16):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice3.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 4:
            for x in range(16, 23):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice4.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if self.truncate_slice >= 5:
            for x in range(23, 30):
                if replace_with_avg_pooling and isinstance(
                    vgg_pretrained_features[x], nn.MaxPool2d
                ):
                    self.slice5.add_module(str(x), nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.slice_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        self.valid_slices = self.slice_names[: self.truncate_slice]

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.lowpass = LowpassFilter2D()

    @override
    def forward(self, x: Tensor, num_scales: int = 3) -> list[Tensor]:
        """
        Forward pass through the truncated VGG network.
        Args:
            X (Tensor): Input image tensor of shape (N, 3, H, W). Assumed
            to be RGB and normalized to [0, 1].
        Returns:
            Dict[int, Tensor]: A dictionary where keys are
            slice indices (1 to truncate_slice) and values are the feature maps
             from the respective slices.
        """
        x = (x - self.mean) / self.std
        features = [x]
        for _ in range(num_scales):
            h = self.slice1(x)
            h_relu1_2 = h
            output_slices = [h_relu1_2]
            if self.truncate_slice >= 2:
                h = self.slice2(h)
                h_relu2_2 = h
                output_slices.append(h_relu2_2)
            if self.truncate_slice >= 3:
                h = self.slice3(h)
                h_relu3_3 = h
                output_slices.append(h_relu3_3)
            if self.truncate_slice >= 4:
                h = self.slice4(h)
                h_relu4_3 = h
                output_slices.append(h_relu4_3)
            if self.truncate_slice >= 5:
                h = self.slice5(h)
                h_relu5_3 = h
                output_slices.append(h_relu5_3)
            features += output_slices
            x = self.lowpass(x, stride=2)

        return features


class VGG16WassersteinDistortion(nn.Module):
    """Calculates the VGG-16 Wasserstein Distortion between two images."""

    def __init__(
        self,
        feature_net: str = "vgg16",
        num_levels: int = 5,
        grayscale: bool = False,
        normalize_center_to_zero: bool = False,
    ):
        super().__init__()
        self.wasserstein_distortion_feature = WassersteinDistortionFeature(num_levels)
        self.grayscale = grayscale
        self.normalize_center_to_zero = normalize_center_to_zero
        if feature_net == "vgg16":
            truncate_slice = 5
            self.feature_backbone = MultiscaleTruncatedVGG16(
                requires_grad=False, pretrained=True, truncate_slice=truncate_slice
            )
            self.truncate_slice = truncate_slice
        else:
            raise ValueError(f"Unsupported feature network: {feature_net}.")

    @override
    def forward(
        self,
        pred: Tensor,
        gt: Tensor,
        log2_sigma: Tensor,
        num_scales: int = 3,
    ) -> Tensor:
        if self.grayscale:
            pred = pred.expand(-1, 3, -1, -1)
            gt = gt.expand(-1, 3, -1, -1)
        if self.normalize_center_to_zero:
            pred = pred * 2 - 1
            gt = gt * 2 - 1
        if pred.shape != gt.shape:
            raise ValueError(
                f"Predicted and ground truth images must have the same shape, "
                f"but got {pred.shape} and {gt.shape}."
            )
        feats_pred = self.feature_backbone(pred, num_scales=num_scales)
        feats_gt = self.feature_backbone(gt, num_scales=num_scales)

        wasserstein_dist = 0
        assert len(feats_pred) == len(feats_gt)
        for fp, fgt in zip(feats_pred, feats_gt):
            ls = F.interpolate(
                log2_sigma, size=fgt.shape[-2:], mode="bilinear", antialias=False
            )
            # Rescale sigma to match the feature arrays. For example, if a feature array
            # has a very low spatial resolution, we make sigma correspondingly smaller,
            # because each element in the feature array covers a larger portion of the
            # image. Since we are in log space, we subtract the log of the size ratio and
            # then cap at zero.
            log_ratio_h = np.log2(log2_sigma.shape[-2] / fgt.shape[-2])
            log_ratio_w = np.log2(log2_sigma.shape[-1] / fgt.shape[-1])
            mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
            ls = F.relu(ls - mean_log_ratio)
            wasserstein_dist += self.wasserstein_distortion_feature(fp, fgt, ls)
        assert isinstance(wasserstein_dist, Tensor)
        return wasserstein_dist
