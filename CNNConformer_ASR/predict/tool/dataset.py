import json
import random
import torch
from torch.utils.data import Dataset
from functools import partial

from StarGANv2VC.common.tool.melspectrogram import load_wav, mel_spectrogram


class SpeDataset(Dataset):
    def __init__(self,
                 list_wav,
                 path_norm,
                 n_fft,
                 win_size,
                 hop_size,
                 num_mels,
                 sampling_rate,
                 fmin,
                 fmax):
        # load

        self.list_wav = list_wav

        with open(path_norm) as js:
            norm = json.loads(js.read())
        self.mean = norm["mean"]
        self.std = norm["std"]

        # initialize

        self.f_melspe = partial(mel_spectrogram,
                                n_fft=n_fft,
                                win_size=win_size,
                                hop_size=hop_size,
                                num_mels=num_mels,
                                sampling_rate=sampling_rate,
                                fmin=fmin,
                                fmax=fmax)

    def __getitem__(self, index):
        wavfile = self.list_wav[index]
        wav, _ = load_wav(wavfile)

        spe = self.f_melspe(wav)
        spe = (spe - self.mean) / self.std
        spe = spe.unsqueeze(0)

        return spe

    def __len__(self):
        return len(self.list_wav)
