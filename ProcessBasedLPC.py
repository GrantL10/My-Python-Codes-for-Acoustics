# -*- coding: utf-8 -*-

import time
import librosa
import numpy as np
import audiolazy.lazy_lpc as alpc
# import matplotlib.pyplot as plt
from multiprocessing import Pool


# %% 预处理 & 后处理
def readWave(filename, frame_len):
    wave_data, fs = librosa.load(filename, sr=None, mono=False, dtype='float64')
    length = len(wave_data)
    n = frame_len / 2
    last = length % n
    if last != 0:  # If samples are not an integer multiple of N/2, then zeros should be filled in.
        new_len = length + n - last
        wave_data = np.append(wave_data, np.zeros(int(n - last)))
        frame = int(new_len / n - 1)
    else:
        frame = int(length / n - 1)
    return wave_data, frame, fs


def writeWave(path, data, fs):
    librosa.output.write_wav(path, data, fs)


# %% LPC模块
# The function used by multiprocessing can't be set as function's function
def calculateLPC(parameter):
    return alpc.lpc(parameter[0], parameter[1]).numerator


def getLPC(time, frame, frame_len, order):
    n = int(frame_len / 2)
    win = np.hanning(frame_len)
    with Pool() as p:  # default <missing the number in Pool()>: the same with the number of cores
        task = [(time[i * n: (i + 2) * n] * win, order) for i in range(frame)]
        data = p.map(calculateLPC, task)
    # The map function data return mechanism may have an error. When the speech frame is all 0, the LPCs should be all 1. 
    # But only one element 1 is returned in the list returned by the map, and the repeated elements are all omitted.
    # Post-processing to repair abnormal data
    lpc_frame = np.ones([frame, order + 1])
    for i in range(frame):
        if len(data[i]) == 1:
            continue
        else:
            lpc_frame[i,] = np.array(data[i])
    # lpc_frame = np.array(data, np.float64)
    return lpc_frame


# %% 主程序
# Without "if __name__ == '__main__':", the multiprocessing may be given an strange error in Windows platform
if __name__ == '__main__':
    # Without "__spec__ = ……", the multiprocessing program can't start in Spyder IDE
    __spec__ = "ModuleSpec(name='builtins', loader=<class xxxx'_frozen_importlib.BuiltinImporter'>)"
    tic = time.time()
    frame_len = 1024
    order = 20
    file_name = r"hvd_001_5.wav"
    wav_in, frame, fs = readWave(file_name, frame_len)
    lpc_frame = getLPC(wav_in, frame, frame_len, order)
    toc = time.time()
    print("%.2f mins" % ((toc - tic) / 60))
