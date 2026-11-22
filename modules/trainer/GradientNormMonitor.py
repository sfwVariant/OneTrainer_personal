import math
from collections import deque
from statistics import fmean, median

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class GradientNormMonitor:
    def __init__(self, tensorboard: SummaryWriter):
        self.tensorboard = tensorboard
        self.step_norms: list[tuple[Tensor, Tensor]] = []
        self.history: deque[tuple[float, bool]] = deque(maxlen=100)

    def record(self, pre_clip_norm: Tensor, post_clip_norm: Tensor):
        self.step_norms.append((pre_clip_norm, post_clip_norm))

    def report(self, global_step: int):
        if not self.step_norms:
            return

        # Fused back pass measures each parameter separately. Combine their L2 norms
        # after backward so the hooks never have to synchronize with the CPU.
        device = self.step_norms[0][0].device
        norms = torch.stack([norm.to(device) for pair in self.step_norms for norm in pair]).tolist()
        self.step_norms.clear()
        pre_norms, post_norms = norms[::2], norms[1::2]
        pre_clip = math.hypot(*pre_norms)
        post_clip = math.hypot(*post_norms)

        self.tensorboard.add_scalar("gradient norm/pre_clip", pre_clip, global_step)
        if math.isfinite(pre_clip) and math.isfinite(post_clip):
            clip_ratio = post_clip / pre_clip if pre_clip else 1.0
            clipped = any(post < pre for pre, post in zip(pre_norms, post_norms, strict=True))
            self.history.append((pre_clip, clipped))
        else:
            # Mixed-precision overflow can produce non-finite gradients for a single update.
            clip_ratio = float("nan")
        self.tensorboard.add_scalar("gradient norm/clip_ratio", clip_ratio, global_step)

        if self.history:
            norm_history, clip_history = zip(*self.history, strict=True)
            self.tensorboard.add_scalar("gradient norm/mean", fmean(norm_history), global_step)
            self.tensorboard.add_scalar("gradient norm/median", median(norm_history), global_step)
            self.tensorboard.add_scalar("gradient norm/max", max(norm_history), global_step)
            self.tensorboard.add_scalar("gradient norm/clip_frequency", fmean(clip_history), global_step)
