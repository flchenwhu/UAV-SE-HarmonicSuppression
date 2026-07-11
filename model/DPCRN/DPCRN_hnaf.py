import torch
import torch.nn as nn
from model.DPCRN.DPCRN import DPCRN as BaseDPCRN


class HarmonicNoiseAdaptiveFrontend(nn.Module):
    """
    GTCRN-Drone-style harmonic noise-adaptive frontend.

    The estimator receives magnitude, real and imaginary spectra with shape
    (B, 3, T, F). The resulting gain is applied to the real and imaginary DPCRN
    input channels with shape (B, 2, T, F).
    """

    def __init__(self, freq_bins=257, channels=3, gain_strength=5.0):
        super().__init__()
        self.gain_strength = gain_strength
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
        self.gain_adjust = nn.Parameter(torch.ones(1, 1, 1, freq_bins))
        self.smooth = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2))

    def _match_freq_bins(self, freq_len):
        if self.gain_adjust.size(-1) == freq_len:
            return self.gain_adjust
        return nn.functional.interpolate(self.gain_adjust, size=(1, freq_len), mode="nearest")

    def forward(self, x):
        if x.size(1) != 2:
            raise ValueError(f"HNAF expects a DPCRN input shaped as (B, 2, T, F), got {tuple(x.shape)}")

        real = x[:, 0:1]
        imag = x[:, 1:2]
        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12)
        feat = torch.cat([mag, real, imag], dim=1)

        noise_response = self.noise_estimator(feat)
        noise_response = self.smooth(noise_response)

        gain_adjust = self._match_freq_bins(x.size(-1))
        denominator = 1.0 + self.gain_strength * noise_response * gain_adjust
        denominator = torch.clamp(denominator, min=0.1)
        gain = 1.0 / denominator

        return x * gain, noise_response


HNAF = HarmonicNoiseAdaptiveFrontend


class DPCRN_HNAF(BaseDPCRN):
    """
    DPCRN_01 with a switchable HNAF inserted before the encoder.

    The public interface remains compatible with DPCRN_01:
    input:  (B, F, T, 2), (B, 1, F, T, 2), or grouped 5-D complex spectra
    output: the same spectral layout as the input.

    HNAF follows the connection pattern used in GTCRN-Drone: it preprocesses the
    encoder input to help the backbone estimate a better mask, while the final
    complex mask is still applied to the original normalized mixture spectrum.
    """

    def __init__(self, *args, use_hnaf=True, return_noise_response=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_hnaf = use_hnaf
        self.return_noise_response = return_noise_response
        if use_hnaf:
            self.hnaf = HarmonicNoiseAdaptiveFrontend(freq_bins=self.freq_bins)
            self._initialize_hnaf_weights()

    def _initialize_hnaf_weights(self):
        for module in self.hnaf.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        original_shape = x.shape

        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() == 5:
            batch, streams, freq, frames, channels = x.shape
            x = x.reshape(batch * streams, freq, frames, channels)

        if x.dim() != 4:
            raise ValueError(f"Input should be shaped as (B, F, T, 2), got {x.dim()} dimensions")
        if x.size(-1) != 2:
            raise ValueError(f"The last input dimension should be 2 for real and imaginary parts, got {x.size(-1)}")

        mix = x.permute(0, 3, 2, 1)

        energy = torch.sqrt(mix[:, 0, :, :] ** 2 + mix[:, 1, :, :] ** 2 + 1e-8)
        max_energy, _ = torch.max(energy.reshape(energy.size(0), -1), dim=1, keepdim=True)
        scale = 1.0 / (max_energy.reshape(-1, 1, 1, 1) + 1e-8)
        mix_normalized = mix * scale

        encoder_input = mix_normalized
        noise_response = None
        if self.use_hnaf:
            encoder_input, noise_response = self.hnaf(encoder_input)

        encoded, encoder_features = self.encoder(encoder_input)

        dprnn_out1 = self.dprnn1(encoded)
        dprnn_out2 = self.dprnn2(dprnn_out1)

        mask = self.decoder(dprnn_out2, encoder_features)

        if mask.size(1) != 2:
            raise ValueError(f"Mask channel dimension should be 2, got {mask.size(1)}")

        mask_real = mask[:, 0, :, :]
        mask_imag = mask[:, 1, :, :]
        mix_real = mix_normalized[:, 0, :, :]
        mix_imag = mix_normalized[:, 1, :, :]

        real = mix_real * mask_real - mix_imag * mask_imag
        imag = mix_real * mask_imag + mix_imag * mask_real

        enhanced = torch.stack([real, imag], dim=1)
        enhanced = enhanced / scale
        enhanced_spec = enhanced.permute(0, 3, 2, 1)

        if original_shape[0] != enhanced_spec.shape[0]:
            enhanced_spec = enhanced_spec.reshape(*original_shape)

        if self.return_noise_response:
            return enhanced_spec, noise_response
        return enhanced_spec


class DPCRN_HNAF_Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = DPCRN_HNAF(**kwargs)

    def forward(self, x):
        if torch.is_complex(x):
            x = torch.stack([x.real, x.imag], dim=-1)
        return self.model(x)

    def stft(self, audio, return_complex=True):
        return self.model.stft(audio, return_complex)

    def istft(self, complex_spec, length=None):
        return self.model.istft(complex_spec, length)

    def enhance_audio(self, audio, length=None):
        return self.model.enhance_audio(audio, length)


dpcrn = DPCRN_HNAF_Wrapper
DPCRN = DPCRN_HNAF
DPCRN_Wrapper = DPCRN_HNAF_Wrapper
