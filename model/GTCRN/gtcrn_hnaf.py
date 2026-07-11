# -*- coding: utf-8 -*-
"""GTCRN-Drone with an HNAF frontend.

This file mirrors the DPCRN split: the plain GTCRN backbone lives in
``gtcrn_drone_01.py``, while this model inserts HNAF before the ERB mapping
stage.
"""

import numpy as np
import torch
import torch.nn as nn

from model.GTCRN.gtcrn import GTCRN as BaseGTCRN


class HarmonicNoiseAdaptiveFrontend(nn.Module):
    """HNAF operating on [|Yc|, Re(Yc), Im(Yc)] shaped as (B, C, T, F)."""

    def __init__(self, freq_bins=257, channels=3):
        super().__init__()

        self.noise_estimator = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        init_af = np.log(np.expm1(1.0))
        self.gain_adjust = nn.Parameter(torch.full((1, 1, 1, freq_bins), init_af))
        self.smooth = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2))
        nn.init.constant_(self.smooth.weight, 0.2)
        nn.init.zeros_(self.smooth.bias)

    def forward(self, x):
        noise_prob = self.noise_estimator(x)
        noise_response = self.smooth(noise_prob)
        noise_response = torch.where(
            torch.isfinite(noise_response),
            noise_response,
            torch.zeros_like(noise_response),
        )
        noise_response = torch.clamp(noise_response, min=0.0, max=1.0)
        freq_weight = nn.functional.softplus(self.gain_adjust)
        gain = 1.0 / (1.0 + noise_response * freq_weight * 5.0)
        gain = torch.clamp(gain, min=0.1, max=1.0)
        return x * gain, noise_response


class GTCRN_HNAF(BaseGTCRN):
    """GTCRN with HNAF inserted before the GTCRN encoder."""

    def __init__(self):
        super().__init__()
        self.hnaf = HarmonicNoiseAdaptiveFrontend(freq_bins=257, channels=3)

    def forward(self, spec):
        spec_ref, feat = self._prepare_feat(spec)
        feat, _ = self.hnaf(feat)
        m_feat = self._run_backbone(feat)
        return self._apply_mask(m_feat, spec_ref)


GTCRN = GTCRN_HNAF
gtcrn_hnaf = GTCRN_HNAF


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 257, 63, 2, device=device)
    model = GTCRN_HNAF().to(device).eval()
    with torch.no_grad():
        y = model(x)
    print(f"GTCRN_HNAF input={tuple(x.shape)}, output={tuple(y.shape)}")
