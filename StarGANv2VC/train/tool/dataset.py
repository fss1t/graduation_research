import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial

from common.tool.melspectrogram import load_wav, mel_spectrogram


class SpeDataset(Dataset):
    def __init__(self,
                 list_path_list_wav,
                 path_norm,
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

        self.list_wav = []  # for each utterance
        self.dict_speaker = []  # for each utterance
        self.list_range_speaker = []  # for each speaker

        for num_speaker, path_list_wav in enumerate(list_path_list_wav):
            with open(path_list_wav, encoding="utf-8") as txt:
                lines = txt.read().splitlines()

                self.dict_speaker.extend([num_speaker for _ in range(len(lines))])
                self.list_range_speaker.append([len(self.list_wav) + len(lines), len(self.list_wav) - 1])
                self.list_wav.extend(lines)
        self.len_list_wav = len(self.list_wav)
        for range_speaker in self.list_range_speaker:
            range_speaker[1] += self.len_list_wav

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

    def __getitem__(self, index_input):
        num_input = self.dict_speaker[index_input]

        # select target utterance

        index_target = random.randint(*self.list_range_speaker[num_input]) % self.len_list_wav
        num_target = self.dict_speaker[index_target]

        # process wav

        list_spe = []
        for index in [index_input, index_target]:
            path_wav = self.list_wav[index]
            wav, _ = load_wav(path_wav)

            if wav.size(-1) < self.segment_size:
                wav = F.pad(wav, (0, self.segment_size - wav.size(-1)))

            # cut segment

            i_start = random.randint(0, wav.size(-1) - self.segment_size)
            wav = wav[:, i_start:i_start + self.segment_size]

            if self.random:
                # randomize gain

                gain = (1.0 + random.random()) / 2.0
                wav = gain * wav

            # get mel spectrogram

            spe = self.f_melspe(wav)
            spe = (spe - self.mean) / self.std
            list_spe.append(spe)

        return list_spe[0], num_input, num_target, list_spe[1]

    def __len__(self):
        return self.len_list_wav
