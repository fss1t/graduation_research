"""tool for copying folder tree"""

import sys
import os
from pathlib import Path
from shutil import copy, copytree


def main():
    #dir_o = "CNNConformer_ASR"
    #dir_o = "JDCNet"
    dir_o = "StarGANv2VC"
    dir_c = dir_o + "_c"
    distribute(dir_o, dir_c)


def distribute(dir_o, dir_c):
    os.chdir(os.path.dirname(__file__))  # cd .

    path_o = Path(dir_o)

    list_path_file = [*list(path_o.glob("**/*.py")), *list(path_o.glob("**/*.json"))]  # glob except checkpoints, logs
    for path_file_o in list_path_file:
        path_dir_c = Path(str(path_file_o).replace(dir_o, dir_c)).parent
        path_dir_c.mkdir(parents=1, exist_ok=1)
        copy(path_file_o, path_dir_c)


if __name__ == "__main__":
    main()
