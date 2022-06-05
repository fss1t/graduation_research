import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial

from StarGANv2VC.common.tool.melspectrogram import load_wav, mel_spectrogram
from common.tool.f0 import f0_vuv


class SpeF0Dataset(Dataset):
    def __init__(self,
                 path_list_wav,
                 path_norm,
                 n_fft,
                 win_size,
                 hop_size,
                 num_mels,
                 sampling_rate,
                 fmin,
                 fmax,
                 f0_floor,
                 f0_ceil,
                 segment_size,
                 random_seed):
        if random_seed is not None:
            random.seed(random_seed)
            self.random = 1
        else:
            self.random = 0

        # load

        with open(path_list_wav, encoding="utf-8") as txt:
            self.list_wav = txt.read().splitlines()

        with open(path_norm) as js:
            norm = json.loads(js.read())
        self.mean = norm["mean"]
        self.std = norm["std"]

        # initialize

        self.segment_size = segment_size

        self.f_melspe = partial(mel_spectrogram,
                                n_fft=n_fft,
                                win_size=win_size,
                                hop_size=hop_size,
                                num_mels=num_mels,
                                sampling_rate=sampling_rate,
                                fmin=fmin,
                                fmax=fmax)

        self.f_f0_vuv = partial(f0_vuv,
                                hop_size=hop_size,
                                sampling_rate=sampling_rate,
                                f0_floor=f0_floor,
                                f0_ceil=f0_ceil)

    def __getitem__(self, index):
        path_wav = self.list_wav[index]
        wav, _ = load_wav(path_wav)

        if wav.size(-1) < self.segment_size:
            wav = F.pad(wav, (0, self.segment_size - wav.size(-1)))

        # cut segment

        if self.random:
            # randomize segment range and gain

            i_start = random.randint(0, wav.size(-1) - self.segment_size)
            wav = wav[:, i_start:i_start + self.segment_size]
            gain = (1.0 + random.random()) / 2.0
            wav = gain * wav
        else:
            wav = wav[:, :self.segment_size]

        # get mel spectrogram

        spe = self.f_melspe(wav)
        spe = (spe - self.mean) / self.std

        # get f0, vuv

        f0, vuv = self.f_f0_vuv(wav.double())

        return spe, f0, vuv

    def __len__(self):
        return len(self.list_wav)
