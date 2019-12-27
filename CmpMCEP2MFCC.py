# -*- coding: utf-8 -*-

import time
import librosa
import numpy as np
import pysptk.sptk as sp
import matplotlib.pyplot as plt
from matplotlib import cm


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
    mfcc_frame = np.empty([frame, order])
    for i in range(0, frame):
        wav_frame[i, :] = wav[i * n: (i + 2) * n] * win
        lpc_frame[i, :] = sp.lpc(wav_frame[i, :], order)
        mcep_frame[i, :] = sp.mcep(wav_frame[i, :], order)
        # mgc_frame[i, :] = sp.mgcep(wav_frame[i, :], order)    # RuntimeError
        mfcc_frame[i, :] = sp.mfcc(wav_frame[i, :], order, num_filterbanks=order * 2, alpha=0.97, eps=1, cepslift=22)
    return lpc_frame, mcep_frame, mfcc_frame, wav_frame


if __name__ == '__main__':
    tic = time.time()
    frame_len = 512
    order = 40
    file_name = r"hvd_001_1.wav"  # Arbitrary Mono Audio
    wav_in, frame, fs = readWave(file_name, frame_len)
    lpc_frame, mcep_frame, mfcc_frame, _ = Analysis(wav_in, frame, frame_len, order)
    toc = time.time()
    print("Preprocessing time-consuming: %.2f seconds" % (toc - tic))

    l_mel = librosa.feature.melspectrogram(y=wav_in, sr=16000, S=None, n_fft=frame_len, hop_length=int(frame_len / 2),
                                           win_length=frame_len, window='hann', center=True, pad_mode='reflect',
                                           power=2.0, n_mels=order)
    l_mel = l_mel.T
    l_mfcc = librosa.feature.mfcc(y=wav_in, sr=16000, S=None, n_fft=frame_len, hop_length=int(frame_len / 2),
                                  win_length=frame_len, window='hann', n_mfcc=order, dct_type=2, norm='ortho', lifter=0)
    l_mfcc = l_mfcc.T

    i = 80
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.title('MCEPs by PySPTK')
    plt.plot(mcep_frame[i,])
    plt.subplot(2, 2, 2)
    plt.title('MFCCs by PySPTK')
    plt.plot(mfcc_frame[i,])
    plt.subplot(2, 2, 3)
    plt.title('Mel-spectrogram by Librosa')
    plt.plot(l_mel[i + 1,])
    plt.subplot(2, 2, 4)
    plt.title('MFCCs by Librosa')
    plt.plot(l_mfcc[i + 1,])
    plt.show()

    # plt.figure(2)
    # plt.subplot(1, 3, 1)
    # plt.title('MCEPs')
    # plt.plot(mcep_frame[i,])
    # plt.subplot(1, 3, 2)
    # plt.title('MLSA')
    # plt.plot(sp.mc2b(mcep_frame[i,]))
    # plt.subplot(1, 3, 3)
    # plt.title('MCEPs')
    # plt.plot(sp.b2mc(sp.mc2b(mcep_frame[i,])))
    # plt.show()

    fig = plt.figure(3)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x, y = np.meshgrid(np.arange(0, mcep_frame.shape[0]), np.arange(0, mcep_frame.shape[1]))
    ax1.plot_surface(x.T, y.T, mcep_frame, cmap=cm.viridis)
    ax1.set_title('MCEPs by PySPTK')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    x, y = np.meshgrid(np.arange(0, mfcc_frame.shape[0]), np.arange(0, mfcc_frame.shape[1]))
    ax2.plot_surface(x.T, y.T, mfcc_frame, cmap=cm.viridis)
    ax2.set_title('MFCCs by PySPTK')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    x, y = np.meshgrid(np.arange(0, l_mel.shape[0]), np.arange(0, l_mel.shape[1]))
    ax3.plot_surface(x.T, y.T, l_mel, cmap=cm.viridis)
    ax3.set_title('Mel-spectrogram by Librosa')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    x, y = np.meshgrid(np.arange(0, l_mfcc.shape[0]), np.arange(0, l_mfcc.shape[1]))
    ax4.plot_surface(x.T, y.T, l_mfcc, cmap=cm.viridis)
    ax4.set_title('MFCCs by Librosa')
