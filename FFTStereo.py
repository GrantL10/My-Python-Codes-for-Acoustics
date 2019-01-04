# -*- coding: utf-8 -*-
# Only for two-channel files

import time
import librosa
import numpy as np
import matplotlib.pyplot as plt


def readWave(filename, frame_len):
    wave_data, fs = librosa.load(filename, sr=None, mono=False, dtype='float64')
    length = len(wave_data[0, :])
    n = frame_len / 2
    last = length % n
    if last != 0:  # If samples are not an integer multiple of N/2, then zeros should be filled in.
        new_len = length + n - last
        wave_data = np.append(wave_data, np.zeros([2, int(n - last)]), axis=1)
        frame = int(new_len / n - 1)
    else:
        frame = int(length / n - 1)
    return wave_data, frame, fs


def writeWave(path, data, fs):
    librosa.output.write_wav(path, data, fs)


def showWave(input_data, output_data, fs):
    length = np.arange(0, len(input_data[0, :])) / fs
    # print(params)
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(length, input_data[0, :])
    plt.subplot(2, 2, 3)
    plt.plot(length, input_data[0, :], c="r")
    plt.subplot(2, 2, 2)
    plt.plot(length, output_data[1, :])
    plt.subplot(2, 2, 4)
    plt.plot(length, output_data[1, :], c="r")
    plt.xlabel("time")
    plt.show()


def FFT(time, frame, nfft):
    n = int(nfft / 2)
    win = np.hanning(nfft)
    freq_l = np.empty([frame, nfft], dtype=complex)
    freq_r = np.empty([frame, nfft], dtype=complex)
    for i in range(0, frame):
        freq_l[i, :] = np.fft.fft(time[0, i * n: (i + 2) * n] * win)
        freq_r[i, :] = np.fft.fft(time[1, i * n: (i + 2) * n] * win)
    return freq_l, freq_r


def IFFT(freq_l, freq_r, frame, nfft):
    n = int(nfft / 2)
    time = np.zeros([2, (frame + 1) * n])
    for i in range(0, frame):
        time[0, i * n:i * n + nfft] += np.real(np.fft.ifft(freq_l[i, :]))
        time[1, i * n:i * n + nfft] += np.real(np.fft.ifft(freq_r[i, :]))
    return time


if __name__ == '__main__':
    tic = time.time()
    frame_len = 1024
    file_name = r"es01.wav"
    wav_in, frame, fs = readWave(file_name, frame_len)
    freq_l, freq_r = FFT(wav_in, frame, frame_len)
    wav_out = IFFT(freq_l, freq_r, frame, frame_len)
    # writeWave(r"abc.wav", wav_out, fs)
    # print((wav_in[0,:]-wav_out[0,:]).max())
    toc = time.time()
    print(toc - tic)
    showWave(wav_in, wav_out, fs)
