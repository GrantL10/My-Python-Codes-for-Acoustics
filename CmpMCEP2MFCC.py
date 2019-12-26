# -*- coding: utf-8 -*-

import time
import librosa
import numpy as np
import pysptk.sptk as sp
import matplotlib.pyplot as plt


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



def Analysis(wav, frame, frame_len, order):
    n = int(frame_len / 2)
    win = np.hanning(frame_len)
    wav_frame = np.empty([frame, frame_len])  # 默认float64
    lpc_frame = np.empty([frame, order + 1])
    mcep_frame = np.empty([frame, order + 1])
    mgc_frame = np.empty([frame, order + 1])
    mfcc_frame = np.empty([frame, order + 1])
    for i in range(0, frame):
        wav_frame[i, :] = wav[i * n: (i + 2) * n] * win
        lpc_frame[i, :] = sp.lpc(wav_frame[i, :], order)
        mcep_frame[i, :] = sp.mcep(wav_frame[i, :], order)
        mfcc_frame[i, :] = sp.mcep(wav_frame[i, :], order)
    return lpc_frame, mcep_frame, mgc_frame, mfcc_frame, wav_frame


if __name__ == '__main__':
    tic = time.time()
    frame_len = 512
    order = 40
    file_name = r"..\Vocoder\hvd_001_1.wav"                # Arbitrary Mono Audio
    wav_in, frame, fs = readWave(file_name, frame_len)
    lpc_frame, mcep_frame, mgc_frame, mfcc_frame, _= Analysis(wav_in, frame, frame_len, order)
    toc = time.time()
    print("Preprocessing time-consuming: %.2f seconds" % (toc - tic))

    i = 88
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.title('MCEP')
    plt.plot(mcep_frame[i,])
    plt.subplot(1, 3, 2)
    plt.title('MGC')
    plt.plot(mgc_frame[i,])
    plt.subplot(1, 3, 3)
    plt.title('MFCC')
    plt.plot(mfcc_frame[i,])
    plt.show()
