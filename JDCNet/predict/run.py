import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
import json
from attrdict import AttrDict
import torch

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parents[1]))
    sys.path.append(str(Path(__file__).parents[2]))

from predict.tool.dataset import SpeDataset
from predict.tool.plot_f0 import plot_f0
from common.model.jdcnet import JDCNet


def predict(list_wav,
            path_dir_param=Path("../dataset/param"),
            path_dir_checkpoint=Path("../train/checkpoint"),
            path_config_data="../../HiFiGAN/config_v1.json",
            path_out=Path("./out"),
            device="cuda:0"):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- predict ---")

    # prepare directory

    path_out.mkdir(exist_ok=1)

    # load config

    with open(path_config_data, "r") as js:
        hd = json.loads(js.read())
    hd = AttrDict(hd)

    # prepare model

    recognizer = JDCNet()
    recognizer = recognizer.to(device)
    recognizer.eval()

    path_cp = path_dir_checkpoint / "recognizer_best.cp"
    assert path_cp.exists()
    cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
    recognizer.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp}")

    # prepare dataset

    dataset = SpeDataset(list_wav,
                         path_dir_param / "norm.json",
                         hd.n_fft,
                         hd.win_size,
                         hd.hop_size,
                         hd.num_mels,
                         hd.sampling_rate,
                         hd.fmin,
                         hd.fmax)

    # predict

    with torch.no_grad():
        for batch, path_wav in zip(dataset, list_wav):
            spe = batch.to(device)

            f0_h, vuv_h = recognizer(spe)

            f0_h = f0_h * (vuv_h >= 0.5)
            f0_h = f0_h.detach().squeeze().to("cpu").numpy()
            plot_f0(path_out / (path_wav.stem + ".png"), f0_h, hd.sampling_rate, hd.hop_size)


if __name__ == "__main__":
    predict([])
