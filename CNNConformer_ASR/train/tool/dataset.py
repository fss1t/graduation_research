import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial

from StarGANv2VC.common.tool.melspectrogram import load_wav, mel_spectrogram


class SpeLabDataset(Dataset):
    def __init__(self,
                 path_list_wav,
                 path_list_lab,
                 path_norm,
                 path_phoneme,
                 n_fft,
                 win_size,
                 hop_size,
                 num_mels,
                 sampling_rate,
                 fmin,
                 fmax,
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
        with open(path_list_lab, encoding="utf-8") as txt:
            self.list_lab = txt.read().splitlines()

        with open(path_norm) as js:
            norm = json.loads(js.read())
        self.mean = norm["mean"]
        self.std = norm["std"]

        with open(path_phoneme) as js:
            self.dict_phoneme = json.loads(js.read())

        # initialize

        self.sampling_rate = sampling_rate
        self.hop_size_lab = hop_size * 4
        self.segment_size = segment_size
        self.len_lab = segment_size // self.hop_size_lab

        self.f_melspe = partial(mel_spectrogram,
                                n_fft=n_fft,
                                win_size=win_size,
                                hop_size=hop_size,
                                num_mels=num_mels,
                                sampling_rate=sampling_rate,
                                fmin=fmin,
                                fmax=fmax)

    def __getitem__(self, index):
        path_wav = self.list_wav[index]
        wav, _ = load_wav(path_wav)

        if wav.size(-1) < self.segment_size:
            wav = F.pad(wav, (0, self.segment_size - wav.size(-1)))

        # randomize wav samples

        if self.random:
            i_start = random.randint(0, wav.size(-1) - self.segment_size)
            wav = wav[:, i_start:i_start + self.segment_size]
            gain = (1.0 + random.random()) / 2.0
            wav = gain * wav
        else:
            wav = wav[:, :self.segment_size]
            i_start = 0

        # get mel spectrogram

        spe = self.f_melspe(wav)
        spe = (spe - self.mean) / self.std

        # encode lab

        path_lab = self.list_lab[index]
        with open(path_lab, "r") as labf:
            lines_lab = labf.read().splitlines()

        lab = torch.zeros(self.len_lab, dtype=torch.long)
        i = 0
        for line in lines_lab:
            items = line.split()
            time_start_phoneme, time_end_phoneme, phoneme = float(items[0]), float(items[1]), items[2]
            i_start_phoneme = (self.sampling_rate * time_start_phoneme - i_start) / self.hop_size_lab
            i_end_phoneme = (self.sampling_rate * time_end_phoneme - i_start) / self.hop_size_lab
            code = self.dict_phoneme[phoneme]
            while(i_start_phoneme <= i and i < i_end_phoneme and i < self.len_lab):
                lab[i] = code
                i += 1

        return spe, lab

    def __len__(self):
        return len(self.list_wav)
