import os
from pathlib import Path
import argparse

import sys
sys.path.append(str(Path(__file__).parent))
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[1]))

from dataset.make_list import make_list_wav
from dataset.calc_norm import calc_norm
from train.run import train
from predict.run import predict


def main(path_jvs):
    os.chdir(os.path.dirname(__file__))  # cd .

    path_jvs = Path(path_jvs).resolve()
    get_wav(path_jvs)
    make_list_wav()
    calc_norm()
    train()

    list_wav = [path_jvs / "jvs068/nonpara30/wav24kHz16bit/TRAVEL1000_0929.wav"]
    predict(list_wav)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_jvs", default="../jvs_ver1")
    args = parser.parse_args()
    main(args.path_jvs)
