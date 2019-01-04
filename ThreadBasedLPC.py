# -*- coding: utf-8 -*-

import time
import threading, multiprocessing
import librosa
import numpy as np
import audiolazy.lazy_lpc as alpc


# import pysptk
# import matplotlib.pyplot as plt


# %% 具备返回多参数能力的多线程封装模块
class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

# %% 预处理 & 后处理
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

# %% LPC模块
def calculateLPC(wav_frame, order):
    # The function used by multiprocessing can't be set as function's function
    temp = np.zeros([len(wav_frame[:, 1]), order + 1])
    for i in range(len(wav_frame[:, 1])):
        temp[i, :] = alpc.lpc(wav_frame[i, :], order).numerator
    return temp


def getLPC(time, frame, frame_len, order):
    n = int(frame_len / 2)
    win = np.hanning(frame_len)
    wav_frame = np.zeros([frame, frame_len])  # 默认float64
    lpc_frame = np.zeros([frame, order + 1])  # 默认float64
    cores = multiprocessing.cpu_count()
    last = frame % cores
    frag = int((frame - last) / cores)
    threads = []
    for i in range(frame):
        wav_frame[i, :] = time[i * n: (i + 2) * n] * win
    for i in range(cores):
        t = MyThread(calculateLPC, args=(wav_frame[i * frag:(i + 1) * frag, :], order))
        threads.append(t)
        t.start()
    if last != 0:
        t = MyThread(calculateLPC, args=(wav_frame[cores * frag:cores * frag + last, :], order))
        threads.append(t)
        t.start()
    for i in range(cores):  # join()方法等待线程完成
        threads[i].join()
        lpc_frame[i * frag:(i + 1) * frag, :] = threads[i].getResult()
    if last != 0:
        threads[cores].join()
        lpc_frame[cores * frag:cores * frag + last, :] = threads[cores].getResult()
    return lpc_frame

# %% 主程序
if __name__ == '__main__':
    tic = time.time()
    frame_len = 512
    order = 20
    file_name = r"hvd_001_5.wav"
    wav_in, frame, fs = readWave(file_name, frame_len)
    lpc_frame = getLPC(wav_in, frame, frame_len, order)
    toc = time.time()
    print(toc - tic)
