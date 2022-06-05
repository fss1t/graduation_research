import os
from pathlib import Path, PurePosixPath


def make_list_wav(path_data=Path("./data"),
                  path_dir_list=Path("./list")):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- make wav file list ---")

    # prepare directory

    path_dir_list.mkdir(exist_ok=1)

    list_use = ["train", "valid"]

    # make wav list

    for use in list_use:
        list_path_wav = sorted((path_data / use).glob("*.wav"))
        with open(path_dir_list / f"wav_{use}.txt", "w", encoding="utf-8") as txt:
            for path_wav in list_path_wav:
                txt.write(str(PurePosixPath(path_wav.resolve())) + "\n")


if __name__ == "__main__":
    make_list_wav()
