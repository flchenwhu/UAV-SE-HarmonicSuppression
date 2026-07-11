import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 固定随机种子
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)


class Chomp_T(nn.Module):
    """时间维度裁剪模块"""

    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t

    def forward(self, x):
        return x[:, :, 0:-self.chomp_t, :]


class Encoder(nn.Module):
    """编码器模块，将复数谱编码为高级特征"""

    def __init__(self):
        super(Encoder, self).__init__()

        # 时间维度填充
        pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)

        # 编码器层定义
        self.en1 = nn.Sequential(
            pad1,
            nn.Conv2d(2, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.en2 = nn.Sequential(
            pad1,
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.en3 = nn.Sequential(
            pad1,
            nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.en4 = nn.Sequential(
            pad1,
            nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.PReLU())

        self.en5 = nn.Sequential(
            pad1,
            nn.Conv2d(64, 128, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.PReLU())

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量，形状为(B, 2, T, F)

        Returns:
            x: 编码后的特征
            x_list: 各层特征列表
        """
        x_list = []

        x = self.en1(x)
        x_list.append(x)

        x = self.en2(x)
        x_list.append(x)

        x = self.en3(x)
        x_list.append(x)

        x = self.en4(x)
        x_list.append(x)

        x = self.en5(x)
        x_list.append(x)

        return x, x_list


class DPRNN(nn.Module):
    """双路径循环神经网络模块"""

    def __init__(self):
        super(DPRNN, self).__init__()

        # Intra-RNN: 频率维度循环
        self.intra_rnn = nn.LSTM(128, 64, 2, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(128, 128)

        # Inter-RNN: 时间维度循环
        self.inter_rnn = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=False)
        self.inter_fc = nn.Linear(128, 128)

        # 层归一化
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入张量，形状为(B, C, T, F)

        Returns:
            处理后的张量
        """
        batch_size, chan_len, seq_len, freq_len = x.shape

        # 保存原始输入用于残差连接
        x_orig = x

        # 处理频率维度 (Intra-RNN)
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        x_reshaped = x_permuted.reshape(-1, freq_len, chan_len)

        intra_out, _ = self.intra_rnn(x_reshaped)
        intra_out = self.intra_fc(intra_out)

        intra_out = intra_out.reshape(batch_size, seq_len, freq_len, chan_len)
        intra_out = self.ln1(intra_out)
        intra_out = intra_out.permute(0, 3, 1, 2)
        intra_out = intra_out + x_orig

        # 处理时间维度 (Inter-RNN)
        inter_input = intra_out.permute(0, 3, 2, 1).contiguous()
        inter_input = inter_input.reshape(-1, seq_len, chan_len)

        inter_out, _ = self.inter_rnn(inter_input)
        inter_out = self.inter_fc(inter_out)

        inter_out = inter_out.reshape(batch_size, freq_len, seq_len, chan_len)
        inter_out = inter_out.permute(0, 3, 2, 1)
        inter_out = self.ln2(inter_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        inter_out = inter_out + intra_out

        return inter_out


class Decoder(nn.Module):
    """解码器模块，将高级特征解码为复数掩码"""

    def __init__(self):
        super(Decoder, self).__init__()

        # 频率维度填充
        pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)

        # 解码器层定义
        self.de1 = nn.Sequential(
            nn.ConvTranspose2d(128 * 2, 64, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(64),
            nn.PReLU())

        self.de2 = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 32, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 32, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.de4 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 32, kernel_size=(2, 3), stride=(1, 2)),
            pad1,
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.PReLU())

        self.de5 = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 2, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1))

    def forward(self, x, x_list):
        """前向传播

        Args:
            x: 输入特征
            x_list: 编码器各层特征列表

        Returns:
            解码后的复数掩码
        """
        x = torch.cat((x, x_list[4]), dim=1)
        x = self.de1(x)

        x = torch.cat((x, x_list[3]), dim=1)
        x = self.de2(x)

        x = torch.cat((x, x_list[2]), dim=1)
        x = self.de3(x)

        x = torch.cat((x, x_list[1]), dim=1)
        x = self.de4(x)

        x = torch.cat((x, x_list[0]), dim=1)
        x = self.de5(x)

        return x


class DPCRN(nn.Module):
    """完整的DPCRN模型，专为复数谱输入设计

    接口要求:
    1. 输入: 复数谱，形状为(B, F, T, 2)
    2. 输出: 增强后的复数谱，形状为(B, F, T, 2)
    3. 与训练框架完全兼容
    """

    def __init__(self, freq_bins=257, time_frames=251, n_fft=512, hop_length=256, win_length=512):
        """初始化DPCRN模型

        Args:
            freq_bins: 频率维度大小，默认257
            time_frames: 时间维度大小，默认251
            n_fft: STFT的FFT大小，默认512
            hop_length: STFT的跳数，默认256
            win_length: STFT的窗口长度，默认512
        """
        super(DPCRN, self).__init__()

        # 保存参数
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 构建模型组件
        self.encoder = Encoder()
        self.dprnn1 = DPRNN()
        self.dprnn2 = DPRNN()
        self.decoder = Decoder()

        # 初始化模型权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def stft(self, audio, return_complex=True):
        """计算STFT

        Args:
            audio: 输入音频，形状为(B, L)或(L,)
            return_complex: 是否返回复数格式

        Returns:
            复数谱，形状为(B, F, T, 2)或(B, F, T)复数格式
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        window = torch.hann_window(self.win_length, device=audio.device)
        stft_result = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            normalized=False,
            return_complex=True
        )

        if return_complex:
            return stft_result
        else:
            # 转换为实部和虚部格式
            real = stft_result.real
            imag = stft_result.imag
            complex_spec = torch.stack([real, imag], dim=-1)
            return complex_spec

    def istft(self, complex_spec, length=None):
        """计算iSTFT

        Args:
            complex_spec: 复数谱，可以是复数格式或(B, F, T, 2)格式
            length: 输出音频长度

        Returns:
            音频波形
        """
        if complex_spec.dim() == 4 and complex_spec.size(-1) == 2:
            # 从(B, F, T, 2)转换为复数格式
            real = complex_spec[..., 0]
            imag = complex_spec[..., 1]
            complex_spec = torch.complex(real, imag)

        window = torch.hann_window(self.win_length, device=complex_spec.device)
        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            normalized=False,
            length=length,
            return_complex=False
        )

        return audio

    def forward(self, x):
        """前向传播

        Args:
            x: 输入复数谱，形状为(B, F, T, 2)

        Returns:
            增强后的复数谱，形状为(B, F, T, 2)
        """
        # 记录输入形状
        original_shape = x.shape

        # 如果输入是5维(B, 1, F, T, 2)，压缩为4维
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() == 5:
            # 如果5维且第二维不为1，则需要特殊处理
            B, S, F, T, C = x.shape
            x = x.reshape(B * S, F, T, C)

        # 检查输入维度
        if x.dim() != 4:
            raise ValueError(f"输入应该是4维 (B, F, T, 2)，但得到: {x.dim()}维")

        if x.size(-1) != 2:
            raise ValueError(f"输入的最后一个维度应该是2（实部和虚部），但得到: {x.size(-1)}")

        # 获取输入维度
        B, F, T, C = x.shape

        # 从 (B, F, T, 2) 转换为 (B, 2, T, F)
        mix = x.permute(0, 3, 2, 1)

        # 能量归一化
        energy = torch.sqrt(mix[:, 0, :, :] ** 2 + mix[:, 1, :, :] ** 2 + 1e-8)
        max_energy, _ = torch.max(energy.reshape(energy.size(0), -1), dim=1, keepdim=True)
        scale = 1.0 / (max_energy.reshape(-1, 1, 1, 1) + 1e-8)
        mix_normalized = mix * scale

        # 编码器
        encoded, encoder_features = self.encoder(mix_normalized)

        # 双路径RNN处理
        dprnn_out1 = self.dprnn1(encoded)
        dprnn_out2 = self.dprnn2(dprnn_out1)

        # 解码器
        mask = self.decoder(dprnn_out2, encoder_features)

        # 确保掩码形状正确
        if mask.size(1) != 2:
            raise ValueError(f"掩码的通道维度应为2，但得到: {mask.size(1)}")

        # 复数掩码乘法
        mask_real = mask[:, 0, :, :]
        mask_imag = mask[:, 1, :, :]
        mix_real = mix_normalized[:, 0, :, :]
        mix_imag = mix_normalized[:, 1, :, :]

        # 复数乘法: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        real = mix_real * mask_real - mix_imag * mask_imag
        imag = mix_real * mask_imag + mix_imag * mask_real

        # 重新组合
        enhanced = torch.stack([real, imag], dim=1)

        # 反归一化
        enhanced = enhanced / scale

        # 转换回原始格式
        enhanced_spec = enhanced.permute(0, 3, 2, 1)

        # 如果需要，恢复原始维度
        if original_shape[0] != enhanced_spec.shape[0]:
            # 如果批次维度被改变了，恢复原始形状
            enhanced_spec = enhanced_spec.reshape(*original_shape)

        return enhanced_spec

    def enhance_audio(self, audio, length=None):
        """完整的音频增强流程

        Args:
            audio: 输入音频，形状为(B, L)或(L,)
            length: 输出音频长度

        Returns:
            增强后的音频
        """
        # 保存原始形状
        original_shape = audio.shape

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # 计算STFT
        complex_spec = self.stft(audio, return_complex=False)

        # 模型处理
        enhanced_spec = self.forward(complex_spec)

        # 计算iSTFT
        enhanced_audio = self.istft(enhanced_spec, length=audio.shape[-1] if length is None else length)

        # 恢复原始维度
        if len(original_shape) == 1:
            enhanced_audio = enhanced_audio.squeeze(0)

        return enhanced_audio

    def get_num_params(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        """打印模型摘要"""
        print("=" * 60)
        print("DPCRN模型摘要")
        print("=" * 60)
        print(f"模型名称: DPCRN")
        print(f"输入形状: (B, {self.freq_bins}, {self.time_frames}, 2)")
        print(f"输出形状: (B, {self.freq_bins}, {self.time_frames}, 2)")
        print(f"STFT参数: n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}")
        print(f"模型参数量: {self.get_num_params():,} ({self.get_num_params() / 1e6:.2f}M)")
        print("=" * 60)


# 兼容性包装类
class DPCRN_Wrapper(nn.Module):
    """DPCRN包装类，提供与训练框架的完全兼容接口"""

    def __init__(self, **kwargs):
        super(DPCRN_Wrapper, self).__init__()
        self.model = DPCRN(**kwargs)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入复数谱，可以是多种格式
                - (B, F, T, 2): 标准复数谱格式
                - (B, 1, F, T, 2): 带额外维度的复数谱
                - 复数张量: (B, F, T)复数格式

        Returns:
            增强后的复数谱，形状与输入对应
        """
        # 处理复数张量输入
        if torch.is_complex(x):
            # 转换为实部虚部格式
            real = x.real
            imag = x.imag
            x = torch.stack([real, imag], dim=-1)

        return self.model(x)

    def stft(self, audio, return_complex=True):
        """STFT接口"""
        return self.model.stft(audio, return_complex)

    def istft(self, complex_spec, length=None):
        """iSTFT接口"""
        return self.model.istft(complex_spec, length)

    def enhance_audio(self, audio, length=None):
        """音频增强接口"""
        return self.model.enhance_audio(audio, length)


# 主函数
if __name__ == "__main__":
    # 测试模型
    print("测试DPCRN模型与训练框架的兼容性...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = DPCRN().to(device)
    model.summary()

    # 测试各种输入格式
    test_cases = [
        (torch.randn(4, 257, 251, 2).to(device), "4维复数谱输入 (B, F, T, 2)"),
        (torch.randn(4, 1, 257, 251, 2).to(device), "5维复数谱输入 (B, 1, F, T, 2)"),
    ]

    for input_data, description in test_cases:
        print(f"\n测试: {description}")
        print(f"输入形状: {input_data.shape}")

        with torch.no_grad():
            output_data = model(input_data)

        print(f"输出形状: {output_data.shape}")

        # 检查输出是否合理
        if input_data.dim() == 5 and input_data.size(1) == 1:
            # 对于5维输入，输出应该也是5维
            expected_shape = input_data.shape
        else:
            # 对于4维输入，输出应该是4维
            expected_shape = input_data.shape

        print(f"形状匹配: {output_data.shape == expected_shape}")
        print(f"输出范围: [{output_data.min():.4f}, {output_data.max():.4f}]")

    # 测试音频处理流程
    print("\n测试音频处理流程:")
    test_audio = torch.randn(1, 16000).to(device)
    enhanced_audio = model.enhance_audio(test_audio)
    print(f"输入音频形状: {test_audio.shape}")
    print(f"输出音频形状: {enhanced_audio.shape}")

    print("\n模型测试完成!")
