import json
import random
import torch
from torch.utils.data import Dataset
from functools import partial

from common.tool.melspectrogram import load_wav, mel_spectrogram


class SpeDataset(Dataset):
    def __init__(self,
                 list_wavin_name_wavref,
                 path_norm,
                 list_path_list_train,
                 n_fft,
                 win_size,
                 hop_size,
                 num_mels,
                 sampling_rate,
                 fmin,
                 fmax):
        # load

        self.list_wavin_name_wavref = list_wavin_name_wavref

        with open(path_norm) as js:
            norm = json.loads(js.read())
        self.mean = norm["mean"]
        self.std = norm["std"]

        # make speaker dictionary

        self.dict_num_speaker = {name.stem: num for num, name in enumerate(list_path_list_train)}

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
        path_wav_input, name_target, path_wav_target = self.list_wavin_name_wavref[index]

        list_spe = []
        for path_wav in [path_wav_input, path_wav_target]:
            wav, _ = load_wav(path_wav_input)
            spe = self.f_melspe(wav)
            spe = (spe - self.mean) / self.std
            spe = spe.unsqueeze(0)
            list_spe.append(spe)

        num_target = self.dict_num_speaker[name_target]
        num_target = torch.LongTensor([num_target])

        return list_spe[0], num_target, list_spe[1]

    def __len__(self):
        return len(self.list_wavin_name_wavref)
