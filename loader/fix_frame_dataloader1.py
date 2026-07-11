import numpy as np
import soundfile as sf
import scipy.signal as sps
import librosa
import os
import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchaudio
# from loader.VAD import VoiceActivityDetector as VAD

eps = np.finfo(np.float32).eps


def audioread(path, fs=16000, start=0, stop=None):
    wave_data, sr = sf.read(path, start=start, stop=stop)
    if stop != None:
        if len(wave_data) < stop:
            wave_data = np.pad(wave_data, (0, stop - len(wave_data)), mode="constant", constant_values=0)
            # wave_data = np.array((stop // len(wave_data)) * list(wave_data) + list(wave_data[:(stop % len(wave_data))]))  # 重复语音到长度stop
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def parse_scp(scp, path_list): #read data
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip()
            path_list.append(tmp)


class FixDataset(Dataset):

    def __init__(self,
                 wav_scp,
                 mix_dir,
                 ref_dir,
                 repeat=1,
                 chunk=4,
                 sample_rate=16000,
                 segment=None):

        super(FixDataset, self).__init__()

        self.wav_list = []
        parse_scp(wav_scp, self.wav_list)  # wav.scp 文件解析
        self.mix_dir = mix_dir
        self.ref_dir = ref_dir

        self.segment_length = None if chunk is None else chunk * sample_rate
        self.wav_list *= repeat
        self.segment = segment
        self.fs = sample_rate
        self.chunk = chunk
        self.total_samples = 0
        self.audiodatabase = []

        # 构建分段样本列表
        if self.segment is not None:
            self.seg_len = int(self.segment * sample_rate)
            for file in self.wav_list:
                utt_id = file
                sample_in_file = int(chunk * sample_rate / self.seg_len)
                for i in range(sample_in_file):
                    self.audiodatabase.append([utt_id, i])
                    self.total_samples += 1
        else:
            self.audiodatabase = self.wav_list

        # === 新增：STFT 参数 ===
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        self.window = torch.hann_window(self.win_length)

    def __len__(self):
        return len(self.audiodatabase)



    def __getitem__(self, index):
        if self.segment is not None:
            [utt_id, start] = self.audiodatabase[index]
            mix_path = os.path.join(self.mix_dir, utt_id + '.wav')
            ref_path = os.path.join(self.ref_dir, utt_id + '.wav')
            mix = audioread(mix_path, self.fs, start * self.seg_len, (start + 1) * self.seg_len)
            ref = audioread(ref_path, self.fs, start * self.seg_len, (start + 1) * self.seg_len)
        else:
            utt_id = self.audiodatabase[index]
            mix_path = os.path.join(self.mix_dir, utt_id + '.wav')
            ref_path = os.path.join(self.ref_dir, utt_id + '.wav')
            mix = audioread(mix_path, self.fs, 0, self.segment_length)
            ref = audioread(ref_path, self.fs, 0, self.segment_length)

        mix = torch.from_numpy(mix).float()
        ref = torch.from_numpy(ref).float()

        if mix.ndim > 1:
            mix = mix.mean(dim=-1)
        if ref.ndim > 1:
            ref = ref.mean(dim=-1)

        spec_mix = torch.stft(
            mix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
        )  # (F, T, 2)

        spec_ref = torch.stft(
            ref,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=False,
        )

        # 添加 batch 通道维度
        spec_mix = spec_mix.unsqueeze(0)  # (1, F, T, 2)
        spec_ref = spec_ref.unsqueeze(0)  # (1, F, T, 2)

        egs = {
            "mix": spec_mix,  # 模型输入
            "ref": ref,
            "spec_ref": spec_ref,  # 用于 loss
            "utt_id": utt_id,
        }
        return egs


def make_fix_loader(wav_scp, mix_dir, ref_dir,
                    batch_size=4, repeat=1, num_workers=4,
                    chunk=4, sample_rate=16000):
    dataset = FixDataset(    #get data
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        repeat=repeat,
        chunk=chunk,
        sample_rate=sample_rate,
    )


    loader = tud.DataLoader(   #pick at random dataset
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    return loader

def test_loader():
    wav_scp = "D:/dataset/lzc/data/test_0db.scp"
    mix_dir = "D:/dataset/lzc/data/test/0db/mix"
    ref_dir = "D:/dataset/lzc/data/test/0db/ref"
    repeat = 1
    num_worker = 4
    chunk = 4
    sample_rate = 16000
    batch_size = 3

    loader = make_fix_loader(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        batch_size=batch_size,
        repeat=repeat,
        num_workers=num_worker,
        chunk=chunk,
        sample_rate=sample_rate,
    )

    cnt = 0
    print('len: ', len(loader))
    for idx, egs in enumerate(loader):
        cnt = cnt + 1
        print('cnt: {}'.format(cnt))
        print('egs["mix"].shape:', egs["mix"].shape)
        print('egs["ref"].shape:', egs["ref"].shape)
        # print('egs["ref_vad"].shape:', egs["ref_vad"].shape)
        print()

        # torchaudio.save('C:/Users/x/Desktop/' + str(i) + '.wav', out, 16000)
        # #sf.write(str(cnt) + '.wav', out, 16000)
        if cnt >= 3:
            break
    print('done!')


if __name__ == "__main__":
    test_loader()