import numpy as np
import pyworld as pw
#import pyreaper
import torch


def f0_vuv(wav, hop_size, sampling_rate, f0_floor, f0_ceil):
    frame_period = hop_size / sampling_rate

    wav = wav[0].numpy()

    #wav_int = (wav * MAX_WAV_VALUE).astype(np.int16)
    #_, _, _, f0_mask, _ = pyreaper.reaper(wav_int, sampling_rate, frame_period=frame_period)

    f0, _ = pw.harvest(wav, sampling_rate, frame_period=1000 * frame_period, f0_floor=f0_floor, f0_ceil=f0_ceil)
    f0 = f0[:-1]
    """
    f0_mask_shift = np.zeros_like(f0)
    for i in range(1, f0_mask.shape[0]):
        f0_mask_shift[i] = (f0_mask[i - 1] + f0_mask[i]) / 2.0

    f0 = np.where(f0_mask_shift == -1.0, 0.0, f0)
    """
    log2f0 = log2_f0(f0)
    vuv = np.where(f0 == 0.0, 0.0, 1.0)
    log2f0 = torch.from_numpy(log2f0.astype(np.float32))
    vuv = torch.from_numpy(vuv.astype(np.float32))
    return log2f0, vuv


def log2_f0(linf0):
    log2f0 = np.empty_like(linf0)
    for i in range(linf0.shape[0]):
        if linf0[i] != 0.0:
            log2f0[i] = np.log2(linf0[i])
        else:
            log2f0[i] = 0.0
    return log2f0


def vuv(linf0):
    v = np.empty_like(linf0)
    for i in range(linf0.shape[0]):
        if linf0[i] != 0.0:
            v[i] = 1.0
        else:
            v[i] = 0.0
    return v
