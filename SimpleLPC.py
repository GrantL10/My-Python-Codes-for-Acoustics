# -*- coding: utf-8 -*-

import time
import librosa
import numpy as np
from audiolazy import lazy_lpc
# import matplotlib.pyplot as plt


def readWave(filename, frame_len):
    wave_data, fs = librosa.load(filename, sr=None, mono=False, dtype='float64')
    length = len(wave_data)
    n = frame_len / 2
    last = length % n
    if last != 0:  # 序列不是N/2的整数倍则需要补零
        new_len = length + n - last
        wave_data = np.append(wave_data, np.zeros(int(n - last)))
        frame = int(new_len / n - 1)
    else:
        frame = int(length / n - 1)
    return wave_data, frame, fs


def writeWave(path, data, fs):
    librosa.output.write_wav(path, data, fs)



def getLPC(wav, frame, frame_len, order):
    n = int(frame_len / 2)
    win = np.hanning(frame_len)
    wav_frame = np.empty([frame, frame_len])  # 默认float64
    lpc_frame = np.empty([frame, order + 1])
    for i in range(0, frame):
        wav_frame[i, :] = wav[i * n: (i + 2) * n] * win
        # temp = audiolazy.lazy_lpc.lpc(wav_frame[i, :], order)
        # lpc_frame[i, :] = temp.numerator
        lpc_frame[i, :] = lazy_lpc.lpc(wav_frame[i, :], order).numerator
    return lpc_frame, wav_frame


if __name__ == '__main__':
    tic = time.time()
    frame_len = 1024
    # file_name = r"E:\Database\Lombard German\Joint\all_normal_lombard\all_l00.wav"           # for Windows
    # file_name = r"/windows/E/Database/Lombard German/Joint/all_normal_lombard/all_l00.wav"   # for Linux
    # file_name = r"E:\Database\MPEG标准测试序列\单声道\48k采样\es01.wav"                        # for Windows
    # file_name = r"/windows/E/Database/MPEG标准测试序列/单声道/48k采样/es01.wav"                # for Linux
    file_name = r"E:\Database\Lombard German\After DTW\f1\l00\f1_s01_l00.wav"                # for Windows
    # file_name = r"/windows/E/Database/Lombard German/After DTW/f1/l00/f1_s01_l00.wav"        # for Linux
    wav_in, frame, fs = readWave(file_name, frame_len)
    lpc_frame, wav_frame = getLPC(wav_in, frame, frame_len, 8)
    toc = time.time()
    print(toc - tic)
