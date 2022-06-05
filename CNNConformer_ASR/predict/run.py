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
from predict.tool.decoder import CTCDecoder
from common.tool.get_num_class import get_num_class
from common.conformer.model import Conformer


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

    num_class = get_num_class(path_dir_param / "phoneme.json")
    recognizer = Conformer(num_class)
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

    # prepare decoder

    ctcdecoder = CTCDecoder(path_dir_param / "phoneme.json", blank=-1)

    # predict

    with torch.no_grad():
        for batch, path_wav in zip(dataset, list_wav):
            spe = batch.to(device)

            prob_lab = recognizer(spe)
            sentence = ctcdecoder(prob_lab)

            print(sentence)
            with open(path_out / (path_wav.stem + ".txt"), "w") as txt:
                txt.write(sentence)


if __name__ == "__main__":
    predict([])
