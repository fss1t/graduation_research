import os
from pathlib import Path
import json


def make_dict_phoneme(path_dir_list=Path("./list"),
                      path_dir_param=Path("./param")):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- make dict phoneme to code ---")

    # prepare directory

    path_dir_param.mkdir(exist_ok=1)

    # load lab

    with open(path_dir_list / "lab_train.txt", "r", encoding="utf_8") as txt:
        list_lab = txt.read().splitlines()

    phonemes = set()
    for labfile in list_lab:
        with open(labfile, "r") as labf:
            for line in labf.read().splitlines():
                phonemes.add(line.rstrip().split(" ")[2])

    # make phoneme dict

    phonemes_sil = {"sil", "pau"}
    phonemes = phonemes - phonemes_sil

    dict_phoneme = {}
    for phoneme_sil in phonemes_sil:
        dict_phoneme[phoneme_sil] = 0
    for i, phoneme in enumerate(sorted(phonemes), 1):
        dict_phoneme[phoneme] = i

    with open(path_dir_param / "phoneme.json", "w") as js:
        json.dump(dict_phoneme, js, indent=4)

    print(dict_phoneme)


if __name__ == "__main__":
    make_dict_phoneme()
