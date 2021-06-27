import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def LowPass(f0, Q=1., fs=192000):
    """
    Biquad IIR Low-Pass Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :return: double 2-order coefficients
    """
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = (1 - np.cos(w0)) / 2
    b1 = 1 - np.cos(w0)
    b2 = (1 - np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h


def HighPass(f0, Q=1., fs=192000):
    """
    Biquad IIR High-Pass Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :return: double 2-order coefficients
    """
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = (1 + np.cos(w0)) / 2
    b1 = -1 - np.cos(w0)
    b2 = (1 + np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h


def LowShelf(f0, gain=0., Q=1., fs=192000):
    """
    Biquad IIR Low-Shelf Filter
    :param f0: center frequency
    :param Q: quality factor
    :param gain: gain
    :param fs: sampling rate
    :return: double 2-order coefficients
    """
    A = np.sqrt(10 ** (gain / 20))
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
    a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h


def HighShelf(f0, gain=0., Q=1., fs=192000):
    """
    Biquad IIR High-Shelf Filter
    :param f0: center frequency
    :param Q: quality factor
    :param gain: gain
    :param fs: sampling rate
    :return: double 2-order coefficients
    """
    A = np.sqrt(10 ** (gain / 20))
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
    b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
    a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h


def PeakNotch(f0, gain=0., Q=1., fs=192000):
    """
    Biquad IIR Peak/Notch Filter
    :param f0: center frequency
    :param Q: quality factor
    :param gain: gain; the positive value is peak filter and the negative value is notch filter
    :param fs: sampling rate
    :return: double 2-order coefficients
    """
    A = np.sqrt(10 ** (gain / 20))
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h

if __name__ == '__main__':
    # Demo
    fs = 192000
    sos = np.empty([3, 6])
    sos[0,] = HighShelf(50, 5, 1, fs)
    sos[1,] = LowShelf(350, 5, 0.7, fs)
    sos[2,] = PeakNotch(5500, -6, 0.7, fs)
    w, h = signal.sosfreqz(sos, worN=int(fs / 10), fs=fs)
    h_abs = np.abs(h)
    print('Maximum in dB: %.2f' %(20 * np.log10(np.max(h_abs))))
    print('Minimum in dB: %.2f' %(20 * np.log10(np.min(h_abs))))
    
    plt.close('all')
    axs = plt.figure(1, figsize=(6, 5)).subplots(2, 1)
    axs[0].set_xscale('log')
    axs[0].plot(w[1:], 20 * np.log10(np.abs(h[1:])))
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude (dB)')
    axs[0].grid()
    axs[1].set_xscale('log')
    axs[1].plot(w[1:], np.angle(h[1:]) / np.pi * 180)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (deg)')
    axs[1].grid()
    plt.tight_layout()
    plt.show()
