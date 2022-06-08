import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from importlib import import_module
import json
from attrdict import AttrDict
import torch

import sys
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[1]))
    sys.path.append(str(Path(__file__).parents[2]))

from predict.tool.dataset import SpeDataset
from predict.tool.standardizer import Standardizer
from common.model.models import Generator, StyleEncoder
from common.tool.melspectrogram import write_wav


def predict(list_wavin_name_wavref,
            path_dir_param=Path("../dataset/param"),
            path_dir_list_train=Path("../dataset/list/train"),
            path_dir_checkpoint=Path("../train/checkpoint"),
            path_config_data="../../HiFiGAN/config_v1.json",
            path_config_train="../train/config.json",
            path_package_HiFiGAN=Path("../../HiFiGAN"),
            path_checkpoint_HiFiGAN=Path("../../HiFiGAN/checkpoint/g_01000000"),
            path_out=Path("./out"),
            device="cuda:0"):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- predict ---")

    # dynamic import

    path_package_HiFiGAN = path_package_HiFiGAN.resolve()
    sys.path.append(str(path_package_HiFiGAN.parent))
    sys.path.append(str(path_package_HiFiGAN))
    Generator_HiFiGAN = import_module(f"{path_package_HiFiGAN.name}.models").Generator

    # prepare directory

    path_out.mkdir(exist_ok=1)

    # load config

    with open(path_config_data, "r") as js:
        hd = json.loads(js.read())
    hd = AttrDict(hd)

    with open(path_config_train, "r") as js:
        h = json.loads(js.read())
    h = AttrDict(h)

    # prepare dataset

    list_path_list_train = sorted((path_dir_list_train).glob("*.txt"))
    dataset = SpeDataset(list_wavin_name_wavref,
                         path_dir_param / "norm.json",
                         list_path_list_train,
                         hd.n_fft,
                         hd.win_size,
                         hd.hop_size,
                         hd.num_mels,
                         hd.sampling_rate,
                         hd.fmin,
                         hd.fmax)

    standardizer = Standardizer(path_dir_param / "norm.json")

    # prepare models

    num_domain = len(list_path_list_train)

    converter = Generator(style_dim=h.style_dim)
    style_encoder = StyleEncoder(num_domains=num_domain, style_dim=h.style_dim)

    vocoder = Generator_HiFiGAN(hd).to(device)

    assert path_checkpoint_HiFiGAN.exists()
    cp = torch.load(path_checkpoint_HiFiGAN, map_location=lambda storage, loc: storage)
    vocoder.load_state_dict(cp["generator"])
    print(f"loaded {path_checkpoint_HiFiGAN}")

    converter, style_encoder, vocoder = [item.eval().to(device) for item in [converter, style_encoder, vocoder]]
    vocoder.remove_weight_norm()

    # predict for each model

    list_path_cp_generator = sorted(path_dir_checkpoint.glob("generator_????.cp"))
    list_path_cp_style_encoder = sorted(path_dir_checkpoint.glob("style_encoder_????.cp"))

    for path_cp_generator, path_cp_style_encoder in zip(list_path_cp_generator, list_path_cp_style_encoder):
        epoch = path_cp_generator.name[10:14]
        print(f" -- Inference with epoch {epoch} models --")

        path_out_epoch = path_out / epoch
        path_out_epoch.mkdir(exist_ok=1)

        cp = torch.load(path_cp_generator, map_location=lambda storage, loc: storage)
        converter.load_state_dict(cp)
        print(f"loaded {path_cp_generator}")
        cp = torch.load(path_cp_style_encoder, map_location=lambda storage, loc: storage)
        style_encoder.load_state_dict(cp)
        print(f"loaded {path_cp_style_encoder}")
        del cp
        with torch.no_grad():
            for batch, wavin_name_wavref in zip(dataset, list_wavin_name_wavref):
                spe_input, num_target, spe_target = [item.to(device) for item in batch]

                style_target = style_encoder(spe_target, num_target)
                spe_output = converter(spe_input, style_target)
                spe_output = spe_output.squeeze(1)
                wav_output = vocoder(standardizer(spe_output))

                path_wav = wavin_name_wavref[0]
                write_wav(path_out_epoch / f"{path_wav.parents[2].name}_{path_wav.stem}_to_{wavin_name_wavref[1]}.wav", wav_output, hd.sampling_rate)


if __name__ == "__main__":
    list_wavin_name_wavref = [[]]
    predict(list_wavin_name_wavref)
