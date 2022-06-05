import os
from pathlib import Path
import json
from attrdict import AttrDict
from tqdm import tqdm
import math as m
import torch

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parents[2]))

from StarGANv2VC.common.tool.melspectrogram import load_wav, mel_spectrogram


def calc_norm(path_dir_list=Path("./list"),
              path_dir_param=Path("./param"),
              path_config="../../HiFiGAN/config_v1.json"):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- calculate norm ---")

    # prepare directory

    path_dir_param.mkdir(exist_ok=1)

    # load wav path

    with open(path_dir_list / "wav_train.txt", "r", encoding="utf_8") as txt:
        list_path_wav = txt.read().splitlines()

    # load config

    with open(path_config, "r") as js:
        h = json.loads(js.read())
    h = AttrDict(h)

    # calculate norm

    with torch.no_grad():
        mean = 0.0
        std = 0.0
        n_frame = 0

        print(" -- calculate mean --")
        for path_wav in tqdm(list_path_wav):
            wav, sampling_rate = load_wav(path_wav)
            assert sampling_rate == h.sampling_rate
            melspe = mel_spectrogram(wav,
                                     h.n_fft,
                                     h.win_size,
                                     h.hop_size,
                                     h.num_mels,
                                     h.sampling_rate,
                                     h.fmin,
                                     h.fmax)
            mean += torch.sum(melspe)
            n_frame += melspe.size(-1)
        mean = mean / (n_frame * h.num_mels)

        print(f"mean = {mean}")
        print(" -- calculate standard deviation --")
        for path_wav in tqdm(list_path_wav):
            wav, _ = load_wav(path_wav)
            melspe = mel_spectrogram(wav,
                                     h.n_fft,
                                     h.win_size,
                                     h.hop_size,
                                     h.num_mels,
                                     h.sampling_rate,
                                     h.fmin,
                                     h.fmax)
            std += torch.sum((melspe - mean)**2)
        std = m.sqrt(std / (n_frame * h.num_mels))

        print(f"std = {std}")

        dict_norm = {"mean": mean.item(),
                     "std": std}

        with open(path_dir_param / "norm.json", "w") as js:
            json.dump(dict_norm,
                      js, indent=4)


if __name__ == "__main__":
    calc_norm()
