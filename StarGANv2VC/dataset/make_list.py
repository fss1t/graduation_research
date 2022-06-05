import os
from pathlib import Path, PurePosixPath


def make_list_wav(path_data=Path("./data"),
                  path_dir_list=Path("./list")):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- make wav file list ---")

    # prepare directory

    path_dir_list.mkdir(exist_ok=1)

    list_use = ["train", "valid"]
    for use in list_use:
        (path_dir_list / use).mkdir(exist_ok=1)

    # make wav list

    for use in list_use:
        paths_dir_speaker = sorted((path_data / use).glob("*"))
        for path_dir_speaker in paths_dir_speaker:
            list_path_wav = sorted(path_dir_speaker.glob("*"))
            with open(path_dir_list / use / f"{path_dir_speaker.name}.txt", "w", encoding="utf-8") as txt:
                for path_wav in list_path_wav:
                    txt.write(str(PurePosixPath(path_wav.resolve())) + "\n")


if __name__ == "__main__":
    make_list_wav()
