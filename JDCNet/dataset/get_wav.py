import os
from pathlib import Path
from shutil import copyfile
import json
from attrdict import AttrDict
import numpy as np
from scipy.io import wavfile as iowav
import scipy.signal as sps
from pydub import AudioSegment
import pydub.silence as pds


dict_set = {
    "train": "parallel100",
    "valid": "nonpara30"}
limit_num_file = {
    "train": 100,
    "valid": 10}

args_split_on_silence = {
    "silence_thresh": -40,
    "min_silence_len": 500,
    "keep_silence": 250
}


def get_wav(path_jvs,
            path_data=Path("./data"),
            path_config="../../HiFiGAN/config_v1.json"):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- get wav files from jvs folder ---")

    # load config

    with open(path_config, "r") as js:
        h = json.loads(js.read())
    h = AttrDict(h)

    # prepare directory

    iters = path_jvs.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)

    path_data.mkdir(exist_ok=1)
    for use in dict_set.keys():
        (path_data / use).mkdir(exist_ok=1)

    # copy wav file

    for use, set in dict_set.items():
        list_path_wav = []
        for speaker in speakers:
            path_dir_wav = path_jvs / speaker / set / "wav24kHz16bit"
            list_path_wav_sp = sorted(path_dir_wav.glob("*.wav"))
            if len(list_path_wav_sp) <= limit_num_file[use]:
                list_path_wav.extend(list_path_wav_sp)
            else:
                list_path_wav.extend(list_path_wav_sp[:limit_num_file[use]])
        for path_wav_read in list_path_wav:
            sr_file, wav = iowav.read(path_wav_read)

            path_wav_write = path_data / use / f"{path_wav_read.parents[2].name}_{path_wav_read.name}"
            if sr_file == h.sampling_rate:
                copyfile(path_wav_read, path_wav_write)
            else:
                wav = decimate(wav, sr_file, h.sampling_rate)
                iowav.write(path_wav_write, h.sampling_rate, wav)
            cut_silence(path_wav_write)


def decimate(wav, fsr, fs):
    assert fsr % fs == 0, "warikiremasen"
    wavr = sps.decimate(wav, int(fsr / fs))
    return wavr.astype(np.int16)


def cut_silence(path_wav):
    data = AudioSegment.from_wav(path_wav)

    # if data.duration_seconds > 2.4:
    chunks = pds.split_on_silence(data, **args_split_on_silence)
    if len(chunks) > 1:
        datac = sum(chunks)
        datac.export(path_wav, "wav")
