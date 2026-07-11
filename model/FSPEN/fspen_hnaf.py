import torch
from torch import nn, Tensor
from torch.nn import functional as F

from model.FSPEN.configs.train_configs import TrainConfig
from model.FSPEN.fspen import FullBandDecoder, FullBandEncoder, SubBandDecoder, SubBandEncoder
from model.FSPEN.modules.sequence_modules import DualPathExtensionRNN


class HarmonicNoiseAdaptiveFrontend(nn.Module):
    """
    Noise-adaptive frontend for spectral features.

    Input and output are both shaped as (B, C, T, F). The module estimates
    a time-frequency noise probability map and applies a bounded attenuation
    gain before the backbone model sees the spectrum.
    """
    def __init__(self, freq_bins: int = 257, channels: int = 3):
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
        self.gain_adjust = nn.Parameter(torch.ones(1, 1, 1, freq_bins))
        self.smooth = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2))

    def forward(self, x: Tensor):
        noise_prob = self.noise_estimator(x)
        noise_prob = torch.sigmoid(self.smooth(noise_prob))

        gain_adjust = F.softplus(self.gain_adjust)
        gain = 1.0 / (1.0 + 5.0 * noise_prob * gain_adjust)
        gain = torch.clamp(gain, min=0.1, max=1.0)

        return x * gain, noise_prob


class SpectralNAF(nn.Module):
    """
    Adapter for models whose input spectrum is shaped as (B, F, T, 2).
    """
    def __init__(self, freq_bins: int = 257):
        super().__init__()
        self.naf = HarmonicNoiseAdaptiveFrontend(freq_bins=freq_bins, channels=3)

    def forward(self, spec: Tensor):
        if spec.dim() == 5:
            spec = spec.squeeze(1)

        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-12)

        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)
        feat, noise_prob = self.naf(feat)

        real = feat[:, 1].permute(0, 2, 1)
        imag = feat[:, 2].permute(0, 2, 1)
        enhanced_spec = torch.stack([real, imag], dim=-1)

        return enhanced_spec, noise_prob


class FullSubPathExtension(nn.Module):
    """
    FSPEN with a plug-and-play Noise-Adaptive Frontend.

    The public interface is kept identical to fspen_01.FullSubPathExtension:
    input  (B, F, T, 2)
    output (B, F, T, 2)
    """
    def __init__(self, configs=None, use_naf: bool = True):
        super().__init__()
        if configs is None:
            configs = TrainConfig()
        self.configs = configs
        self.use_naf = use_naf
        self.naf_frontend = SpectralNAF(freq_bins=configs.n_fft // 2 + 1)

        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs)

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels // compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands // compress_rate, kernel_size=1, stride=1),
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands // compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels // compress_rate, out_features=merge_channels),
            nn.ELU(),
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs)
        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)

    def forward(self, spec: Tensor):
        if spec.dim() == 5:
            spec = spec.squeeze(1)

        if self.use_naf:
            spec, _ = self.naf_frontend(spec)

        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
        amplitude_spectrum = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-12)

        complex_spectrum = spec.permute(0, 2, 3, 1)
        batch, frames, channels, frequency = complex_spectrum.shape
        in_complex_spectrum = complex_spectrum
        complex_spectrum = torch.reshape(complex_spectrum, shape=(batch * frames, channels, frequency))

        amplitude_spectrum = amplitude_spectrum.permute(0, 2, 1)
        amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch * frames, 1, frequency))

        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)
        merge_feature = self.feature_merge_layer(merge_feature)
        _, channels, compressed_frequency = merge_feature.shape

        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, compressed_frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()

        hidden_size = self.configs.dual_path_extension["parameters"]["inter_hidden_size"]
        groups = self.configs.dual_path_extension["parameters"]["groups"]
        in_hidden_state = [
            [
                torch.zeros(1, batch * compressed_frequency, hidden_size // groups, device=spec.device)
                for _ in range(groups)
            ]
            for _ in range(self.configs.dual_path_extension["num_modules"])
        ]

        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, _ = rnn_layer(merge_feature, in_hidden_state[idx])

        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, compressed_frequency))

        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, _ = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_complex_spectrum * sub_band_mask.repeat(1, 1, 2, 1)
        output_spectrum = full_band_out + sub_band_out

        return output_spectrum.permute(0, 3, 1, 2)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 257, 251, 2).to(device)
    net = FullSubPathExtension().to(device)

    with torch.no_grad():
        y = net(x)

    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
    print("total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
