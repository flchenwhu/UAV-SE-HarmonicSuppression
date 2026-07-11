import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as tnf
from conv_stft import ConvSTFT

win_len, win_inc, nfft = 512, 256, 512  # 320, 160, 320
WINDOW = torch.sqrt(torch.hann_window(win_len, device='cuda') + 1e-8)

def power_compress_loss(y_pred, y_true):
    """来自DPARN  https://github.com/Qinwen-Hu/dparn  
       A light-weight full-band speech enhancement model  """  # 压缩率α=1/3      # 也可改为0.5试试
    pred_stft = torch.stft(y_pred.squeeze(1), n_fft=nfft, hop_length=win_inc, win_length=win_len, window=WINDOW,
                           return_complex=True)
    true_stft = torch.stft(y_true.squeeze(1), n_fft=nfft, hop_length=win_inc, win_length=win_len, window=WINDOW,
                           return_complex=True) 
    # 使用torch.view_as_real将复数张量转换为实数张量
    pred_stft = torch.view_as_real(pred_stft)
    true_stft = torch.view_as_real(true_stft)
    
    pred_mag = torch.sqrt(pred_stft[..., 0] ** 2 + pred_stft[..., 1] ** 2 + 1e-12)
    true_mag = torch.sqrt(true_stft[..., 0] ** 2 + true_stft[..., 1] ** 2 + 1e-12)
    
    pred_real_c = pred_stft[..., 0] / (pred_mag ** (2 / 3))
    pred_imag_c = pred_stft[..., 1] / (pred_mag ** (2 / 3))
    true_real_c = true_stft[..., 0] / (true_mag ** (2 / 3))
    true_imag_c = true_stft[..., 1] / (true_mag ** (2 / 3))
    
    real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
    imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
    mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)
    # print('real_loss:', real_loss)
    return 10 * (real_loss + imag_loss + mag_loss)  # 乘10自己加的，只是为了loss看得更明显，别写在公式里

def mag_stftm_loss(outputs, labels):
    """ 这里是 频域中 幅度loss + 频域 实部虚部 loss  （与power_compress_loss很像，只是没有压缩）
        github: https://github.com/dangf15/DPT-FSNet  (开源中用这个loss，但论文中论述的loss不是这个)
        参考论文：DPT-FSNet: Dual-Path Transformer Based Full-Band and Sub-Band Fusion Network for Speech Enhancement 2022年
    """
    stft = ConvSTFT(512, 256, 512, 'hann', 'complex', fix=True).cuda()
    fft_len = 512

    # STFT Magnitude Loss
    def get_stftm(ipt):
        specs = stft(ipt)
        real = specs[:, :fft_len // 2 + 1]
        imag = specs[:, fft_len // 2 + 1:]
        return real, imag

    out_real, out_imag = get_stftm(outputs)
    lab_real, lab_imag = get_stftm(labels)
    stftm_loss_value = torch.mean(torch.abs(out_real - lab_real) + torch.abs(out_imag - lab_imag))

    # Magnitude Loss
    def get_mag(ipt):
        specs = stft(ipt)
        real = specs[:, :fft_len // 2 + 1]
        imag = specs[:, fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        return spec_mags

    out_mags = get_mag(outputs)
    lab_mags = get_mag(labels)
    mag_loss_value = torch.mean(torch.abs(out_mags - lab_mags))

    alpha1 = 1  # 可以根据需要调整
    alpha2 = 1  # 可以根据需要调整
    loss = alpha1 * mag_loss_value + alpha2 * stftm_loss_value

    return loss

def hybrid_loss(y_pred, y_true):
    """ Hybrid Loss 
    来自文献：2024 GTCRN：A Speech Enhancement Model Requiring Ultralow Computational Resources"""
    # Compute STFT
    pred_stft = torch.stft(y_pred.squeeze(1), n_fft=nfft, hop_length=win_inc, win_length=win_len, window=WINDOW,
                           return_complex=True)
    true_stft = torch.stft(y_true.squeeze(1), n_fft=nfft, hop_length=win_inc, win_length=win_len, window=WINDOW,
                           return_complex=True) 

    # Convert complex tensors to real
    pred_stft = torch.view_as_real(pred_stft)
    true_stft = torch.view_as_real(true_stft)

    device = pred_stft.device

    pred_stft_real, pred_stft_imag = pred_stft[..., 0], pred_stft[..., 1]
    true_stft_real, true_stft_imag = true_stft[..., 0], true_stft[..., 1]
    pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
    true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
    pred_real_c = pred_stft_real / (pred_mag**(0.7))
    pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
    true_real_c = true_stft_real / (true_mag**(0.7))
    true_imag_c = true_stft_imag / (true_mag**(0.7))
    real_loss = nn.MSELoss()(pred_real_c, true_real_c)
    imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
    mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
    
    y_pred = torch.istft(torch.view_as_complex(pred_stft), nfft, win_inc, win_len, window=WINDOW)
    y_true = torch.istft(torch.view_as_complex(true_stft), nfft, win_inc, win_len, window=WINDOW)
    y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)

    sisnr = -torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2 + 1e-8).mean()

    real_imag_loss = real_loss + imag_loss
    print("Real + Imag Loss:", real_imag_loss.item())
    print("Magnitude Loss:", mag_loss.item())
    print("Si-SNR Loss:", sisnr.item())
    return 25 * real_imag_loss + 75 * mag_loss + sisnr

def time_stftm_loss(outputs, labels):
    """ 时域+频域  loss
        github: https://github.com/dangf15/DPT-FSNet  (这个是论文中论述的loss，别人开源中实际上没有用这个loss)
        参考论文：DPT-FSNet: Dual-Path Transformer Based Full-Band and Sub-Band Fusion Network for Speech Enhancement 2022年
    """
    # 确保outputs和labels长度一致
    min_length = min(outputs.shape[-1], labels.shape[-1])
    outputs = outputs[..., :min_length]
    labels = labels[..., :min_length]

    # 时域损失 Laudio
    Laudio = torch.mean((outputs - labels) ** 2)
    # 频域损失 Lspectral
    frame_size = 512
    frame_shift = 256
    stft = ConvSTFT(frame_size, frame_shift, frame_size, 'hann', 'complex', fix=True).cuda()
    # 计算 STFT
    specs_out = stft(outputs)
    specs_lab = stft(labels)

    out_real = specs_out[:, :frame_size // 2 + 1]
    out_imag = specs_out[:, frame_size // 2 + 1:]
    lab_real = specs_lab[:, :frame_size // 2 + 1]
    lab_imag = specs_lab[:, frame_size // 2 + 1:]
    Lspectral = torch.mean(torch.abs(out_real - lab_real) + torch.abs(out_imag - lab_imag))
    # 合并损失
    alpha1 = 0.4  # 可以根据需要调整
    alpha2 = 0.6  # 可以根据需要调整
    loss = alpha1 * Laudio + alpha2 * Lspectral

    return loss


def fullsub_band_loss(est_stft_lst, clean_stft):
    """ Comes from THLNet """
    out1_real, out1_imag = torch.chunk(est_stft_lst[0], 2, dim=1)
    out2_real, out2_imag = torch.chunk(est_stft_lst[1], 2, dim=1)
    clean_real, clean_imag = torch.chunk(clean_stft, 2, dim=1)
    out1_mag = torch.sqrt(out1_real ** 2 + out1_imag ** 2 + 1e-8)
    out2_mag = torch.sqrt(out2_real ** 2 + out2_imag ** 2 + 1e-8)
    clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-8)
    F_size = clean_mag.shape[1]

    LcRI = (torch.mean((out1_real - clean_real) ** 2) + torch.mean((out1_imag - clean_imag) ** 2)) * F_size
    LcMag = torch.mean((out1_mag - clean_mag) ** 2) * F_size
    Lc = 0.5 * LcRI + 0.5 * LcMag

    LfRI = (torch.mean((out2_real - clean_real) ** 2) + torch.mean((out2_imag - clean_imag) ** 2)) * F_size
    LfMag = torch.mean((out2_mag - clean_mag) ** 2) * F_size
    Lf = 0.5 * LfRI + 0.5 * LfMag

    loss = Lc + Lf
    return loss


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm
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

    # s2 = torch.reshape(s2, [-1, D])
def sisnr_loss(inputs, labels):
    return -(si_snr(inputs, labels))

# def stoi_loss(est, ref):
#     """ https://github.com/mpariente/pytorch_stoi """
#     from torch_stoi import NegSTOILoss
#     loss_func = NegSTOILoss(sample_rate=16000, extended=False)  # extended=True时使用ESTOI
#     loss = loss_func(est, ref)
#     return loss


# def mask_loss(mix_r, mix_i, ref_r, ref_i, est_r, est_i):
#     """
#         return Loss_mask
#         基于cIRM的mask
#         参考论文：Neural Cascade Architecture With Triple-Domain Loss for Speech Enhancement
#     """
#     mix_s = torch.complex(mix_r, mix_i)
#     ref_s = torch.complex(ref_r, ref_i)
#     est_s = torch.complex(est_r, est_i)
#     IRM_mask_s = torch.sqrt(ref_s ** 2 / (mix_s ** 2 + 1e-8))
#     IRM_mask_s_est = torch.sqrt(est_s ** 2 / (mix_s ** 2 + 1e-8))
#     Loss_mask = torch.mean(torch.abs(IRM_mask_s_est - IRM_mask_s))
#     return Loss_mask


# def time_loss(mix_r, mix_i, ref_r, ref_i, est_r, est_i):
#     """
#         return Loss_time= Loss_Mag + Loss_noise_Mag
#         参考论文：Neural Cascade Architecture With Triple-Domain Loss for Speech Enhancement
#     """
#     ref_mag = torch.sqrt(ref_r ** 2 + ref_i ** 2 + 1e-8)
#     est_mag = torch.sqrt(est_r ** 2 + est_i ** 2 + 1e-8)
#     Loss_Mag = torch.mean(est_mag - ref_mag)
#     est_noise_mag = torch.sqrt((mix_r - est_r) ** 2 + (mix_i - est_i) ** 2 + 1e-8)
#     ref_noise_mag = torch.sqrt((mix_r - ref_r) ** 2 + (mix_i - ref_i) ** 2 + 1e-8)
#     Loss_noise_Mag = torch.mean(est_noise_mag - ref_noise_mag)
#     Loss_time = Loss_Mag + Loss_noise_Mag
#     return Loss_time


if __name__ == "__main__":
    import time
    import torchaudio

    torch.manual_seed(1234)

    # x = torch.randn(3, 1, 16000 * 4).clamp(-1, 1)
    # y = torch.randn(3, 1, 16000 * 4).clamp(-1, 1)

    x, _ = torchaudio.load("./Files/推理含噪语音/noisy.wav")
    y, _ = torchaudio.load("./Files/推理干净语音/clean.wav")

    t0 = time.time()
    # loss = power_compress_loss(x.cuda(), y.cuda())
    # loss = mag_stftm_loss(x.cuda(), y.cuda())
    # loss = time_stftm_loss(x.cuda(), y.cuda())
    loss = hybrid_loss(x.cuda(), y.cuda())
    print('Time:', time.time() - t0)
    print(loss)

