import os
from pathlib import Path
import argparse

import sys
sys.path.append(str(Path(__file__).parent))
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[1]))

from dataset.get_wav import get_wav
from dataset.make_list import make_list_wav
from dataset.calc_norm import calc_norm
from train.run import train
from predict.run import predict


def main(path_jvs, package_HiFiGAN, checkpoint_HiFiGAN):
    os.chdir(os.path.dirname(__file__))  # cd .

    path_jvs = Path(path_jvs).resolve()

    # get_wav(path_jvs)
    # make_list_wav()
    # calc_norm()
    # train()

    list_wav = [[path_jvs / "jvs068/nonpara30/wav24kHz16bit/TRAVEL1000_0929.wav",
                 "jvs004", path_jvs / "jvs051/nonpara30/wav24kHz16bit/VOICEACTRESS100_014.wav"],
                [path_jvs / "jvs068/nonpara30/wav24kHz16bit/TRAVEL1000_0929.wav",
                 "jvs005", path_jvs / "jvs051/nonpara30/wav24kHz16bit/VOICEACTRESS100_014.wav"],
                [path_jvs / "jvs068/nonpara30/wav24kHz16bit/TRAVEL1000_0929.wav",
                 "jvs010", path_jvs / "jvs051/nonpara30/wav24kHz16bit/VOICEACTRESS100_014.wav"],
                [path_jvs / "jvs068/nonpara30/wav24kHz16bit/TRAVEL1000_0929.wav",
                 "jvs068", path_jvs / "jvs051/nonpara30/wav24kHz16bit/VOICEACTRESS100_014.wav"], ]

    predict(list_wav, path_package_HiFiGAN=Path(package_HiFiGAN).resolve(), path_checkpoint_HiFiGAN=Path(checkpoint_HiFiGAN).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_jvs", default="../jvs_ver1")
    parser.add_argument("--package_HiFiGAN", default="../HiFiGAN")
    parser.add_argument("--checkpoint_HiFiGAN", default="../HiFiGAN/checkpoint/g_01000000")
    args = parser.parse_args()
    main(args.path_jvs, args.package_HiFiGAN, args.checkpoint_HiFiGAN)
