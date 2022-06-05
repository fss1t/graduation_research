import os
from pathlib import Path, PurePosixPath


dict_set = {
    "train": "parallel100",
    "valid": "nonpara30"}
limit_num_file = {
    "train": 100,
    "valid": 10}
list_datalacker = ["jvs006", "jvs028", "jvs037", "jvs058"]


def make_list_wav(path_jvs,
                  path_dir_list=Path("./list")):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- make wav paths list ---")

    # prepare directory

    iters = path_jvs.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)
    cut_datalacked_speaker(speakers)

    path_dir_list.mkdir(exist_ok=1)

    # make wav list

    for use, set in dict_set.items():
        list_path_wav = []
        for speaker in speakers:
            path_dir_wav = path_jvs / speaker / set / "wav24kHz16bit"
            list_wav_sp = sorted(path_dir_wav.glob("*.wav"))
            if len(list_wav_sp) <= limit_num_file[use]:
                list_path_wav.extend(list_wav_sp)
            else:
                list_path_wav.extend(list_wav_sp[:limit_num_file[use]])
        with open(path_dir_list / f"wav_{use}.txt", "w", encoding="utf-8") as txt:
            for path_wav in list_path_wav:
                txt.write(str(PurePosixPath(path_wav)) + "\n")


def make_list_lab(path_jvs,
                  path_dir_list=Path("./list")):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # cd .
    print("--- make lab paths list ---")

    # prepare directory

    iters = path_jvs.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)
    cut_datalacked_speaker(speakers)

    path_dir_list.mkdir(exist_ok=1)

    # make lab list

    for use, set in dict_set.items():
        list_path_lab = []
        for speaker in speakers:
            path_dir_lab = path_jvs / speaker / set / "lab/mon"
            list_lab_speaker = sorted(path_dir_lab.glob("*.lab"))
            if len(list_lab_speaker) <= limit_num_file[use]:
                list_path_lab.extend(list_lab_speaker)
            else:
                list_path_lab.extend(list_lab_speaker[:limit_num_file[use]])
        with open(path_dir_list / f"lab_{use}.txt", "w", encoding="utf-8") as txt:
            for path_lab in list_path_lab:
                txt.write(str(PurePosixPath(path_lab)) + "\n")


def cut_datalacked_speaker(speakers):
    for lacker in list_datalacker:
        speakers.remove(lacker)
