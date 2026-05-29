from utils import *
from io_utils import *
from losses import *
from NVRC.loss_utils import (
    get_saliency_model, set_saliency_context, clear_saliency_context,
    get_raft_model, get_wloss,
)


class OverfitTask:
    def __init__(self, logger, video, loss_cfg, metric_cfg, lamb,
                 channel_scale=None, channel_shift=None,
                 enable_log=True, training=True, device=None,
                 temp_weight=0.1, temp_tau=1.0):
        self.logger = logger
        self.video = video
        self.channels = self.video.get_num_channels()

        self.loss_cfg = [1.0, loss_cfg[0]] if len(loss_cfg) == 1 else loss_cfg
        self.metric_cfg = metric_cfg

        assert channel_scale is None or channel_shift is None or \
            (len(channel_scale) == self.channels and len(channel_shift) == self.channels)
        self.channel_scale, self.channel_shift = \
            compute_scale_shift(self.channels, self.video.get_bit_depth(), channel_scale, channel_shift)

        self.enable_log = enable_log
        self.training = training
        self.device = device
        self.metrics_buffer = {}
        self._saliency_cache: torch.Tensor | None = None  # [T_total, 1, h_s, w_s] on CPU
        self.temp_weight = float(temp_weight)
        self.temp_tau = float(temp_tau)
        self._flow_cache: torch.Tensor | None = None
        self._w_cache: torch.Tensor | None = None

        assert isinstance(lamb, (list, tuple)) and len(lamb) == 1, 'lamb should be a list/tuple with a single value'
        self.lamb = torch.tensor(sorted(lamb), dtype=torch.float32, device=self.device)

        logger.info(f'OverfitTask:')
        logger.info(f'     Root: {self.video.get_path()}')
        logger.info(f'     Training: {self.training}')
        logger.info(f'     Losses: {self.loss_cfg}    Metrics: {self.metric_cfg}')
        logger.info(f'     Lamb: {self.lamb.tolist()}')
        logger.info(f'     Enable log: {self.enable_log}')

    def get_metrics(self):
        return self.metric_cfg

    def get_video_size(self):
        return tuple(v + sum(p) for v, p in zip(self.video.get_video_size(), self.video.get_padding()))

    def get_patch_size(self):
        return self.video.get_patch_size()

    def get_start_frame(self):
        return self.video.get_start_frame()

    def get_num_frames(self):
        return self.video.get_num_frames()

    def set_frames(self, start_frame, num_frames):
        self.video.set_frames(start_frame, num_frames)

    def create_cache(self):
        self.video.create_cache(enable=self.enable_log)

    def precompute_saliency(self, dataset, batch_size: int = 8) -> None:
        """Precompute per-frame saliency maps once before training.

        Runs the saliency model on full frames (not crops), storing [T, 1, h_s, w_s] on CPU.
        During training, wd_saliency crops the appropriate spatial region per patch.
        Only supported when the video cache is a 4-D numpy array (PNGVideo).
        """
        if not any(loss_type in ('wd-saliency', 'wd-saliency-temp') for loss_type in self.loss_cfg[1::2]):
            return

        video = dataset.video
        cache = getattr(video, 'cache', None)
        if cache is None or not (hasattr(cache, 'ndim') and cache.ndim == 4):
            self.logger.info('Saliency precomputation skipped: unsupported video format.')
            return

        device = self.device
        T_total = dataset.get_num_frames()
        saliency_model = get_saliency_model(device)

        scale = dataset.channel_scale.view(1, -1, 1, 1)
        shift = dataset.channel_shift.view(1, -1, 1, 1)

        results = []
        for i in range(0, T_total, batch_size):
            # cache: [T, H, W, C] numpy → [B, C, H, W] float tensor
            batch_np = cache[i:i + batch_size]
            batch_t = torch.from_numpy(batch_np.astype(np.float32)).permute(0, 3, 1, 2)
            batch_t = (batch_t * scale + shift).clamp(0, 1).to(device)
            with torch.no_grad():
                results.append(saliency_model(batch_t).cpu())

        self._saliency_cache = torch.cat(results, dim=0)  # [T, 1, h_s, w_s]
        self.logger.info(f'Precomputed saliency cache: {self._saliency_cache.shape}')

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
                img1_t, img2_t = raft_transforms(prev_frame, frame)
                with torch.no_grad():
                    flow_preds = raft(img1_t, img2_t)
                flow_list.append(flow_preds[-1].squeeze(0).cpu())  # [2, H, W]

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

    def compute_temp_loss(self, model, output, inputs):
        """Temporal consistency loss term for wd-saliency-temp.

        For each batch sample with temporal index t > 0:
          1. Run a second model forward pass at t-1 (detached).
          2. Warp f_{t-1} to frame t using precomputed RAFT flow.
          3. Weight the squared difference by sigma_t (saliency) * w_t (WD weight).

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
        # Invalid samples (t=0) keep their original index — outputs unused.
        idx_prev = idx.clone()
        idx_prev[valid_mask, 0] = t_vals[valid_mask] - 1
        inputs_prev = dict(inputs)
        inputs_prev['idx'] = idx_prev
        inputs_prev['x'] = None

        N, C, T_p, H_pad, W_pad = output.shape
        if T_p != 1:
            return torch.tensor(0.0, device=device)

        with torch.no_grad():
            f_prev_batch = model(inputs_prev, compute_outputs=True, compute_rates=False)
        f_prev_batch = self.parse_output(f_prev_batch).detach()  # [N, C, T_p, H_pad, W_pad]

        temp_losses: list[torch.Tensor] = []

        for n in range(N):
            if not valid_mask[n]:
                continue

            t = t_vals[n].item()
            h_start = idx[n, 1].item() * H_pad
            w_start = idx[n, 2].item() * W_pad

            # Crop flow to this patch's spatial region — flow is in native pixel units,
            # no magnitude rescaling needed since the patch is at full resolution.
            flow_orig = self._flow_cache[t - 1].to(device)  # [2, H_orig, W_orig]
            H_orig = flow_orig.shape[-2]
            if H_orig != H_pad:
                flow_up = flow_orig[:, h_start:h_start + H_pad, w_start:w_start + W_pad]
            else:
                flow_up = flow_orig  # already patch-sized (T_patch=H_video, shouldn't occur)

            # w scalar
            w_t = self._w_cache[t - 1].to(device)  # scalar

            # sigma map: upsample full-frame saliency then crop to patch region
            if self._saliency_cache is not None:
                s = self._saliency_cache[t].to(device)  # [1, h_s, w_s]
                # upsample to full frame resolution, then crop to patch
                s_full = F.interpolate(
                    s.unsqueeze(0), size=(H_orig, flow_orig.shape[-1]),
                    mode='bilinear', antialias=False,
                ).squeeze(0)  # [1, H_orig, W_orig]
                s = s_full[:, h_start:h_start + H_pad, w_start:w_start + W_pad]  # [1, H_pad, W_pad]
                s_mean = s.mean()
                p = 0.5 + (1 - 0.5) * s / (s_mean + 1e-8)
                sigma_t = 16.0 * 0.5 / p   # [1, H_pad, W_pad]
            else:
                sigma_t = torch.ones(1, H_pad, W_pad, device=device)

            # Warp and loss
            f_t = output[n, :, 0]         # [C, H_pad, W_pad]
            f_p = f_prev_batch[n, :, 0]   # [C, H_pad, W_pad]

            warped = flow_warp(f_p.unsqueeze(0), flow_up.unsqueeze(0)).squeeze(0)  # [C, H_pad, W_pad]

            diff = sigma_t * w_t * (f_t - warped)   # [C, H_pad, W_pad]
            temp_losses.append(diff.pow(2).mean())

        if not temp_losses:
            return torch.tensor(0.0, device=device)

        return self.temp_weight * torch.stack(temp_losses).mean()

    def parse_batch(self, batch):
        """
        Parse the input and output batch during training/evaluation step
        """
        idx, x = batch
        idx_max = self.video.get_idx_max()
        assert idx.ndim == 2, \
            'idx should have 2 dimensions with shape [N, 3], where each row is the 3D patch coordinate'
        assert x.ndim == 5,  \
            'x should have 5 dimensions with shape [N, C, T, H, W], where each sample is a 3D patch'

        inputs = {
            'vidx': torch.zeros(idx.shape[0], dtype=torch.int32, device=idx.device),
            'vidx_max': 1,
            'idx': idx,
            'idx_max': idx_max,
            'x': x if self.training else None,
            'lamb': self.lamb,
            'rp': None,
            'rel_batch_size': x.shape[0] * x.shape[2] * \
                math.prod(self.get_patch_size()[1:]) / math.prod(self.get_video_size()[1:]),
            'video_size': (self.get_num_frames(),) + self.get_video_size()[1:],
            'patch_size': self.get_patch_size(),
            'channels': self.video.get_num_channels()
        }

        return inputs, x

    def parse_output(self, output):
        """
        Parse the output from the model during training/evaluation step
        """
        return output.contiguous()

    def compute_d_loss(self, x, y, lamb, patch_coords=None, idx_max=None):
        if self._saliency_cache is not None and patch_coords is not None:
            set_saliency_context(self._saliency_cache, patch_coords.cpu(), idx_max)
        try:
            loss = 0.
            for i in range(len(self.loss_cfg) // 2):
                weight = float(self.loss_cfg[i * 2])
                loss_type = self.loss_cfg[i * 2 + 1]
                loss += weight * lamb * compute_loss(loss_type, x, y).mean()
        finally:
            clear_saliency_context()
        return loss

    def compute_r_loss(self, r):
        return r

    def compute_metrics(self, x, y):
        metrics = {}
        with torch.no_grad():
            for metric_type in self.metric_cfg:
                metrics[metric_type] = compute_metric(metric_type, x, y)
        return metrics

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

    def r_step(self, model, batch, sub_step=0, num_sub_steps=1):
        inputs, _ = self.parse_batch(batch)
        inputs['r_sub_step'] = sub_step
        inputs['r_num_sub_steps'] = num_sub_steps
        rate, rate_loss = model(inputs, compute_outputs=False, compute_rates=True)
        rate = rate / math.prod(inputs['video_size'])
        rate_loss = self.compute_r_loss(rate_loss / math.prod(inputs['video_size']))
        return rate, rate_loss

    def log_outputs(self, inputs, outputs, metrics):
        """
        Log the outputs
        """
        if self.enable_log:
            N, _, _, _, _ = outputs.shape
            idx = inputs['idx']
            outputs = outputs.detach()
            metrics = {k: v.detach() for k, v in metrics.items()}

            scale = self.channel_scale.to(outputs.device)
            shift = self.channel_shift.to(outputs.device)

            # Loop over all samples
            for n in range(N):
                # Save the patch
                if isinstance(self.video, PNGVideo):
                    self.video.write_patch(idx[n].cpu().numpy(),
                                           ((outputs[n] - shift.view(self.channels, 1, 1, 1)) \
                                            / scale.view(self.channels, 1, 1, 1)) \
                                           .permute(1, 2, 3, 0).round().cpu().numpy())
                else:
                    yuv420_patch = yuv444_to_yuv420(outputs[n].permute(1, 0, 2, 3), mode='avg_pool')
                    self.video.write_patch(idx[n].cpu().numpy(),
                                           [((patch_i - shift[i]) / scale[i]).round().cpu().numpy() \
                                            for i, patch_i in enumerate(yuv420_patch)])

                # Save the averge metrics for each patch
                idx_str = '(' + ' '.join([str(i) for i in idx[n].tolist()]) + ')'
                self.metrics_buffer[idx_str] = \
                    ','.join([idx_str] + \
                             [f'{k}: {v[n].mean().item():.4f}' for k, v in metrics.items() if k in self.metric_cfg])

    def flush(self, dir):
        """
        Flush the outputs to the disk
        """
        if self.enable_log:
            # Save video outputs
            self.video.flush(dir)

            # Save metrics
            with open(os.path.join(dir, 'metrics.txt'), 'w') as f:
                f.write('\n'.join(self.metrics_buffer.values()))
            self.metrics_buffer = {}


def create_overfit_task(args, logger, video, channel_scale=None, channel_shift=None, training=True, device=None):
    # Create task
    if training:
        config = args.train_task
    else:
        config = args.eval_task

    task = OverfitTask(logger, video, loss_cfg=config.loss, metric_cfg=config.metric,
                       lamb=config.lamb, channel_scale=channel_scale, channel_shift=channel_shift,
                       enable_log=config.enable_log, training=training, device=device,
                       temp_weight=config.temp_weight,
                       temp_tau=config.temp_tau)
    return task