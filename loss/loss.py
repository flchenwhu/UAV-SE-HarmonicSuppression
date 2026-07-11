import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as tnf
# from model.stft import STFT
from .tools_for_loss import get_array_mel_loss  # , pmsqe_loss, pmsqe_stft

# from asteroid_filterbanks import transforms

win_len, win_inc, nfft = 512, 256, 512
# win_len, win_inc, nfft = 400, 200, 400
# win_len, win_inc, nfft = 1200, 600, 1200
WINDOW = torch.sqrt(torch.hann_window(win_len, device='cuda') + 1e-8)
import torch.nn.functional as F
# from model.uctnet.pydct import sdct_torch, isdct_torch
from . import ConvSTFT


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    # B, C, D = s2.size()
    # s2 = torch.reshape(s2, [-1, D])
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def sisnr_loss(inputs, labels):
    return -(si_snr(inputs, labels))

    # s2 = torch.reshape(s2, [-1, D])

def drone_hybrid_loss(
    pred,
    target,
    eps=1e-8
):
    """
    pred   : (B,1,T)
    target : (B,1,T)

    Drone-oriented Hybrid Loss
    Compatible with current training pipeline
    """

    # =========================================
    # 1. shape
    # =========================================
    if pred.dim() == 3:
        pred = pred.squeeze(1)

    if target.dim() == 3:
        target = target.squeeze(1)

    # =========================================
    # 2. SI-SNR Loss
    # =========================================
    target_energy = torch.sum(
        target ** 2,
        dim=-1,
        keepdim=True
    )

    proj = (
        torch.sum(
            pred * target,
            dim=-1,
            keepdim=True
        )
        * target
        / (target_energy + eps)
    )

    noise = pred - proj

    sisnr = 10 * torch.log10(
        (
            torch.sum(proj ** 2, dim=-1)
            + eps
        )
        /
        (
            torch.sum(noise ** 2, dim=-1)
            + eps
        )
    )

    loss_sisnr = -sisnr.mean()

    # =========================================
    # 3. Multi-Resolution STFT Loss
    # =========================================
    fft_sizes = [256, 512, 1024]

    stft_loss = 0.0

    for fft_size in fft_sizes:

        hop = fft_size // 4

        window = torch.hann_window(
            fft_size
        ).to(pred.device)

        pred_spec = torch.stft(
            pred,
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            window=window,
            return_complex=True
        )

        target_spec = torch.stft(
            target,
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            window=window,
            return_complex=True
        )

        pred_mag = torch.abs(pred_spec)

        target_mag = torch.abs(target_spec)

        stft_loss += F.l1_loss(
            pred_mag,
            target_mag
        )

    stft_loss /= len(fft_sizes)

    # =========================================
    # 4. Total Loss
    # =========================================
    loss = (
        1.0 * loss_sisnr
        + 0.3 * stft_loss
    )

    return loss



def si_snr_e(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    # B, C, D = s2.size()
    # s2 = torch.reshape(s2, [-1, D])
    a = 0.3
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    e = -(s2 * torch.log2(s1) + (1 - s2) * torch.log2(1 - s2))
    snr_e = snr - e
    return torch.mean(snr_e)


def sisnr_e_loss(inputs, labels):
    return -(si_snr_e(inputs, labels))


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def seg_sisnr_loss(inputs, labels, win_size, hop_size):
    """
        据论文AEC任务中效果较好，区分单讲双讲
        参考论文：F-T-LSTM based Complex Network for Joint Acoustic Echo Cancellation and Speech Enhancement
    """

    def si_snr1(s1, s2):
        s1_s2_norm = l2_norm(s1, s2)
        s2_s2_norm = l2_norm(s2, s2)
        s_target = s1_s2_norm / (s2_s2_norm + 1e-8) * s2
        e_nosie = s1 - s_target
        target_norm = l2_norm(s_target, s_target)
        noise_norm = l2_norm(e_nosie, e_nosie)
        # 在静音帧，target_norm非常小而noise_norm可能较大，会产生较大的负值结果，所以加1。这样做不需要VAD标记静音帧
        # 理论上原sisnr最小值为-∞，现在为0
        sisnr = 10 * torch.log10((target_norm) / (noise_norm + 1e-8) + 1)
        return torch.mean(sisnr)

    if labels.shape[-1] != inputs.shape[-1]:
        minlenth = min(labels.shape[-1], inputs.shape[-1])
        labels = labels[..., : minlenth]
        inputs = inputs[..., : minlenth]
    num_frame = (labels.shape[-1] - win_size + hop_size) // hop_size  # 计算帧的数量
    Seg_SISNR = torch.zeros(num_frame)
    # 计算每一帧的信噪比
    for i in range(num_frame):
        Seg_SISNR[i] = si_snr1(inputs[..., i * hop_size: i * hop_size + win_size],
                               labels[..., i * hop_size: i * hop_size + win_size])
    return - torch.mean(Seg_SISNR)


def ERLE(s1, s2):
    s1 = torch.mean(s1 ** 2)
    s2 = torch.mean(s2 ** 2)
    erel = 10 * torch.log10(s1 / s2)
    return erel


def sisnr_lms_loss(inputs, labels, stft):
    snr_loss = -(si_snr(inputs, labels))
    # for mel loss calculation          sisnr_loss + RMSE[log(est_mel), log(ref_mel)]
    clean_real, clean_imag = stft(labels, cplx=True)
    clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)
    est_real, est_imag = stft(inputs, cplx=True)
    est_mags = torch.sqrt(est_real ** 2 + est_imag ** 2 + 1e-7)
    mel_loss = get_array_mel_loss(clean_mags, est_mags)
    loss = (snr_loss + 2 * mel_loss) / 3
    return loss


def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1)  # Batch preserving sum for convenience.

    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean ** 2) / (bsum(clean ** 2) + bsum(noise ** 2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    return torch.mean(wSDR)


def stft_loss(y_pred, y_true):
    """来自DPARN"""
    pred_stft = torch.stft(y_pred.squeeze(1), n_fft=nfft, hop_length=win_inc, win_length=win_len, window=type,
                           return_complex=False)
    true_stft = torch.stft(y_true.squeeze(1), n_fft=nfft, hop_length=win_inc, win_length=win_len, window=type,
                           return_complex=False)
    pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
    true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
    pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
    pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
    pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
    true_real_c = true_stft_real / (true_mag ** (2 / 3))
    true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
    real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
    imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
    mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)
    # print('real_loss:', real_loss)
    # print('imag_loss:', imag_loss)
    # print('mag_loss:', mag_loss)
    return 10 * (real_loss + imag_loss + mag_loss)  # 乘10自己加的，只是为了看得更明显


def uformer_loss(est, labels):
    '''
    mode == 'Mix'
        est: [B, F*2, T]
        labels: [B, F*2,T]
    mode == 'SiSNR'
        est: [B, T]
        labels: [B, T]
    '''
    # if mode == 'SiSNR':
    #     if labels.dim() == 3:
    #         labels = torch.squeeze(labels, 1)
    #     if est.dim() == 3:
    #         est = torch.squeeze(est, 1)
    #     return -si_snr(est, labels)
    # elif mode == 'Mix':

    stft = ConvSTFT.ConvSTFT(512, 256, 512, 'hamming', 'complex', True)
    b, d, t = est.size()
    gth_cspec = stft(labels)
    est_cspec = stft(est)
    gth_mag_spec = torch.sqrt(
        gth_cspec[:, :257, :] ** 2
        + gth_cspec[:, 257:, :] ** 2 + 1e-8
    )
    est_mag_spec = torch.sqrt(
        est_cspec[:, :257, :] ** 2
        + est_cspec[:, 257:, :] ** 2 + 1e-8
    )
    # power compress
    gth_cprs_mag_spec = gth_mag_spec ** 0.3
    est_cprs_mag_spec = est_mag_spec ** 0.3
    amp_loss = F.mse_loss(
        gth_cprs_mag_spec, est_cprs_mag_spec
    ) * d
    compress_coff = (gth_cprs_mag_spec / (1e-8 + gth_mag_spec)).repeat(1, 2, 1)
    phase_loss = F.mse_loss(
        gth_cspec * compress_coff,
        est_cspec * compress_coff
    ) * d

    all_loss = amp_loss * 0.5 + phase_loss * 0.5
    return all_loss


def stdct_mse_loss(mix, lable):
    mix = sdct_torch(mix, 512, 256).contiguous()
    lable = sdct_torch(lable, 512, 256).contiguous()
    mse_loss = F.mse_loss(torch.abs(mix), torch.abs(lable))
    mse_loss1 = F.mse_loss(mix, lable)
    return mse_loss + mse_loss1


def hybrid_loss1(pred, true):
    # from "A Speech Enhancement Model Requiring Ultralow Computational Resources"
    """ B C L  """
    device = pred.device
    B, C, L = pred.shape
    pred = pred.reshape(B * C, L)
    true = true.reshape(B * C, L)
    pred_stft = torch.stft(pred, 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device), return_complex=False)
    true_stft = torch.stft(true, 512, 256, 512, window=torch.hann_window(512).pow(0.5).to(device), return_complex=False)
    pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
    true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]
    pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
    pred_real_c = pred_stft_real / (pred_mag ** (0.7))
    pred_imag_c = pred_stft_imag / (pred_mag ** (0.7))
    true_real_c = true_stft_real / (true_mag ** (0.7))
    true_imag_c = true_stft_imag / (true_mag ** (0.7))
    real_loss = nn.MSELoss()(pred_real_c, true_real_c)
    imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
    mag_loss = nn.MSELoss()(pred_mag ** (0.3), true_mag ** (0.3))
    # 注意STFT使用torch.stft(mix, 512, 256, 512, torch.hann_window(512).pow(0.5))，和ISTFT保持一致即可

    true = torch.sum(true * pred, dim=-1, keepdim=True) * true / (
                torch.sum(torch.square(true), dim=-1, keepdim=True) + 1e-8)
    sisnr = - torch.log10(torch.norm(true, dim=-1, keepdim=True) ** 2 / (
                torch.norm(pred - true, dim=-1, keepdim=True) ** 2 + 1e-8) + 1e-8).mean()

    return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr

def hybrid_loss_uav(pred, true):
    """
    针对无人机噪声优化的混合损失函数 (Hybrid Loss for UAV Ego-Noise)
    包含: SI-SNR + cMSE + Asymmetric Penalty + Harmonic Peak Penalty
    """
    device = pred.device
    B, C, L = pred.shape
    pred = pred.reshape(B * C, L)
    true = true.reshape(B * C, L)

    # 1. STFT 变换
    window = torch.hann_window(512).pow(0.5).to(device)
    pred_stft = torch.stft(pred, 512, 256, 512, window=window, return_complex=False)
    true_stft = torch.stft(true, 512, 256, 512, window=window, return_complex=False)

    pred_stft_real, pred_stft_imag = pred_stft[:, :, :, 0], pred_stft[:, :, :, 1]
    true_stft_real, true_stft_imag = true_stft[:, :, :, 0], true_stft[:, :, :, 1]

    # 幅度谱
    pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)

    # 压缩复数域
    pred_real_c = pred_stft_real / (pred_mag ** 0.7)
    pred_imag_c = pred_stft_imag / (pred_mag ** 0.7)
    true_real_c = true_stft_real / (true_mag ** 0.7)
    true_imag_c = true_stft_imag / (true_mag ** 0.7)

    # 基础 Loss (与你原来一致)
    real_loss = nn.MSELoss()(pred_real_c, true_real_c)
    imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
    mag_loss = nn.MSELoss()(pred_mag ** 0.3, true_mag ** 0.3)

    # -------------------------------------------------------------------------
    # 🔥 【创新新增】无人机专项损失 (UAV-Specific Penalties)
    # -------------------------------------------------------------------------

    # (a) 非对称噪声残留惩罚 (Asymmetric Noise Residual Penalty)
    # 用 ReLU 只截取 "预测幅度 > 真实幅度" 的部分（即没去干净的噪声）
    residual_noise = tnf.relu(pred_mag - true_mag)
    asym_loss = torch.mean(residual_noise ** 2)

    # (b) 谐波峰值惩罚 (Harmonic Peak / Comb Penalty)
    # 在频率轴 (dim=1) 上应用 Max-Pooling，抓取异常凸起的谐波尖峰
    # residual_noise: (Batch, Freq, Time)
    # 通过 MaxPool1d(kernel=5) 在相邻的 5 个频点中找最大残留值
    peak_residual = tnf.max_pool1d(residual_noise.transpose(1, 2), kernel_size=5, stride=1, padding=2)
    peak_loss = torch.mean(peak_residual ** 2)

    # -------------------------------------------------------------------------

    # SI-SNR 计算 (与你原来一致)
    true_scaled = torch.sum(true * pred, dim=-1, keepdim=True) * true / (
            torch.sum(torch.square(true), dim=-1, keepdim=True) + 1e-8)
    sisnr = - torch.log10(torch.norm(true_scaled, dim=-1, keepdim=True) ** 2 / (
            torch.norm(pred - true_scaled, dim=-1, keepdim=True) ** 2 + 1e-8) + 1e-8).mean()

    # 综合权重 (新加的项赋予较高权重以强制网络关注谐波消除)
    # 权重可以根据实际收敛情况微调
    total_loss = sisnr + 30 * (real_loss + imag_loss) + 70 * mag_loss + 10 * asym_loss + 20 * peak_loss

    return total_loss



def freq_loss(est_spec, spec_ref):
    """
    est_spec: (B, T, F, 2) 复数谱
    spec_ref: (B, T, F, 2) 参考复数谱
    """
    est_complex = torch.view_as_complex(est_spec)
    ref_complex = torch.view_as_complex(spec_ref)
    mag_est = torch.abs(est_complex)
    mag_ref = torch.abs(ref_complex)
    return tnf.mse_loss(mag_est, mag_ref)


if __name__ == "__main__":
    torch.manual_seed(1112)
    import time

    x = torch.randn(8, 1, 16000 * 4).clamp(-1, 1)
    y = torch.randn(8, 1, 16000 * 4)
    # zr, zi = th.chunk(stft.transform(z), 2, dim=1)
    # ref_r = torch.randn(3, 1, 257, 251)
    # ref_i = torch.randn(3, 1, 257, 251)
    # est_r = torch.randn(3, 1, 257, 251)
    # est_i = torch.randn(3, 1, 257, 251)
    # loss1 = sisnr_lms_loss(z, y)
    loss1 = si_snr(x, y)
    loss = uformer_loss(x, y) + loss1
    # past = time.time()
    # now = time.time()
    # print(now - past)
    # print(loss1)
    print(loss)

