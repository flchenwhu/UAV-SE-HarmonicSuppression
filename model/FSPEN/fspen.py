import torch
from torch import nn, Tensor
from torch.fft import fft, ifft
from torch.nn import functional as F
from model.FSPEN.modules.en_decoder import FullBandEncoderBlock, FullBandDecoderBlock
from model.FSPEN.modules.en_decoder import SubBandEncoderBlock, SubBandDecoderBlock
from model.FSPEN.modules.sequence_modules import DualPathExtensionRNN
from model.FSPEN.configs.train_configs import TrainConfig


class FullBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        last_channels = configs.full_band_encoder["encoder1"]["in_channels"]
        self.full_band_encoder = nn.ModuleList()
        for encoder_name, conv_parameter in configs.full_band_encoder.items():
            conv_parameter["in_channels"] = last_channels
            self.full_band_encoder.append(FullBandEncoderBlock(**conv_parameter))
            last_channels = conv_parameter["out_channels"]
        self.global_features = nn.Conv1d(in_channels=last_channels, out_channels=last_channels, kernel_size=1, stride=1)

    def forward(self, complex_spectrum: Tensor):
        full_band_encodes = []
        for encoder in self.full_band_encoder:
            complex_spectrum = encoder(complex_spectrum)
            full_band_encodes.append(complex_spectrum)
        global_feature = self.global_features(complex_spectrum)
        return full_band_encodes[::-1], global_feature


class SubBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.sub_band_encoders = nn.ModuleList()
        for encoder_name, conv_parameters in configs.sub_band_encoder.items():
            self.sub_band_encoders.append(SubBandEncoderBlock(**conv_parameters["conv"]))

    def forward(self, amplitude_spectrum: Tensor):
        sub_band_encodes = list()
        for encoder in self.sub_band_encoders:
            encode_out = encoder(amplitude_spectrum)
            sub_band_encodes.append(encode_out)
        local_feature = torch.cat(sub_band_encodes, dim=2)
        return sub_band_encodes, local_feature


class FullBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        for decoder_name, parameters in configs.full_band_decoder.items():
            self.full_band_decoders.append(FullBandDecoderBlock(**parameters))

    def forward(self, feature: Tensor, encode_outs: list):
        for decoder, encode_out in zip(self.full_band_decoders, encode_outs):
            feature = decoder(feature, encode_out)
        return feature


class SubBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        start_idx = 0
        self.sub_band_decoders = nn.ModuleList()
        for (decoder_name, parameters), bands in zip(configs.sub_band_decoder.items(), configs.bands_num_in_groups):
            end_idx = start_idx + bands
            self.sub_band_decoders.append(SubBandDecoderBlock(start_idx=start_idx, end_idx=end_idx, **parameters))
            start_idx = end_idx

    def forward(self, feature: Tensor, sub_encodes: list):
        sub_decoder_outs = []
        for decoder, sub_encode in zip(self.sub_band_decoders, sub_encodes):
            sub_decoder_out = decoder(feature, sub_encode)
            sub_decoder_outs.append(sub_decoder_out)
        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)
        return sub_decoder_outs


class FullSubPathExtension(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        if configs is None:
            configs = TrainConfig()
        self.configs = configs

        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs)

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        # ---------------------- 移除：原模型内部的STFT窗口（输入已为频谱） ----------------------
        # self.hamming_window = torch.hamming_window(configs.n_fft).to(device=configs.device)

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels // compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands // compress_rate, kernel_size=1, stride=1)
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands // compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels // compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)

    # ---------------------- 修改：输入参数改为spec（频谱），类型标注为Tensor ----------------------
    def forward(self, spec: Tensor):
        # ---------------------- 新增：输入维度处理（(B,F,T,2) → 适配模型内部格式） ----------------------
        if spec.dim() == 5:
            spec = spec.squeeze(1)
        B, F, T, _ = spec.shape  # 输入维度：(B, 257, 251, 2)
        # 1. 从输入频谱计算幅度谱（原模型需要amplitude_spectrum）
        spec_real = spec[..., 0]  # (B,F,T)：实部
        spec_imag = spec[..., 1]  # (B,F,T)：虚部
        amplitude_spectrum = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)  # (B,F,T)：幅度谱

        # 2. 调整复数频谱维度到原模型内部格式：(B,F,T,2) → (B,T,2,F)（匹配原模型permute后结果）
        complex_spectrum = spec.permute(0, 2, 3, 1)  # 维度转换：(B,F,T,2) → (B,T,2,F)

        # ---------------------- 保留原模型后续维度调整逻辑（仅适配输入来源） ----------------------
        batch, frames, channels, frequency = complex_spectrum.shape
        in_complex_spectrum = complex_spectrum  # 保存原始输入频谱（用于后续mask相乘）
        complex_spectrum = torch.reshape(complex_spectrum, shape=(batch * frames, channels, frequency))

        # 调整幅度谱维度到原模型要求：(B,F,T) → (B,T,F) → (batch*frames,1,F)
        amplitude_spectrum = amplitude_spectrum.permute(0, 2, 1)  # (B,F,T) → (B,T,F)
        in_amplitude_spectrum = amplitude_spectrum.unsqueeze(2)  # (B,T,F) → (B,T,1,F)
        amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch * frames, 1, frequency))

        # 初始化隐藏状态（保持原逻辑，设备与输入一致）
        in_hidden_state = [[torch.zeros(1, batch * sum(self.configs.bands_num_in_groups),
                                        self.configs.dual_path_extension["parameters"]["inter_hidden_size"] //
                                        self.configs.dual_path_extension["parameters"]["groups"]).to(spec.device)
                            for _ in range(self.configs.dual_path_extension["parameters"]["groups"])]
                           for _ in range(self.configs.dual_path_extension["num_modules"])]

        # 模型核心处理逻辑（完全保留原逻辑）
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)
        merge_feature = self.feature_merge_layer(merge_feature)
        _, channels, frequency = merge_feature.shape

        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()

        out_hidden_state = list()
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, in_hidden_state[idx])
            out_hidden_state.append(state)

        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))

        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        # 应用mask得到增强后频谱（修复复数+实数相加的问题）
        full_band_out = in_complex_spectrum * full_band_mask  # 复数输出
        
        # 将sub_band_mask转换为复数形式并应用到复数频谱
        sub_band_mask_complex = sub_band_mask.repeat(1, 1, 2, 1)  # 扩展为复数mask
        sub_band_out = in_complex_spectrum * sub_band_mask_complex  # 复数频谱 * 复数mask
        
        # 现在两个都是复数，可以正确相加
        output_spectrum = full_band_out + sub_band_out

        # ---------------------- 修改：输出维度还原为(B,F,T,2)（去掉原ISTFT） ----------------------
        # 维度转换：(B,T,2,F) → (B,F,T,2)，与输入格式完全一致
        output_spectrum = output_spectrum.permute(0, 3, 1, 2)
        # 无需ISTFT，直接输出频谱张量

        return output_spectrum  # 输出维度：(B, F, T, 2)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------------------- 修改：测试输入改为(B,F,T,2)格式（1,257,251,2） ----------------------
    x = torch.randn(1, 257, 251, 2).to(device)  # 符合要求的输入维度
    net = FullSubPathExtension().to(device)
    est = net(x)

    # 验证输出维度（预期：torch.Size([1,257,251,2])）
    print(f"输入维度：{x.shape}")
    print(f"输出维度：{est.shape}")

    # 计算参数量和 FLOPs（保持原逻辑）
    from thop import profile, clever_format
    macs, params = profile(net, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f'thop计算flops和params的结果: {macs, params}')
    print("thop计算Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(net, (257, 251, 2), as_strings=True,
                                              print_per_layer_stat=False, verbose=False)
    print(f'ptflops方法计算 Flops: {flops}; 参数量: {params}')