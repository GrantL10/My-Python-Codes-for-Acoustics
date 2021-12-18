import numpy as np


def lowPass(f0, Q=1., fs=48000):
    """
    Biquad IIR Low-Pass Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :return double 2-order coefficients
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


def highPass(f0, Q=1., fs=48000):
    """
    Biquad IIR High-Pass Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :return double 2-order coefficients
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

def bandPass(f0, Q=1., fs=48000, type=0):
    """
    Biquad IIR Band-Pass Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :param type: 0) constant 0 dB peak gain; 1) constant skirt gain, peak gain = Q
    :return double 2-order coefficients
    """
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    if type == 0:
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
    else:                      # type 1)
        b0 = np.sin(w0) / 2    # = Q * alpha
        b1 = 0
        b2 = -np.sin(w0) / 2   # = -Q * alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h

def allPass(f0, Q=1., fs=48000):
    """
    Biquad IIR All-Pass Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :return double 2-order coefficients
    """
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1 - alpha
    b1 = -2 * np.cos(w0)
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h


def lowShelf(f0, gain=0., Q=1., fs=48000):
    """
    Biquad IIR Low-Shelf Filter
    :param f0: center frequency
    :param Q: quality factor
    :param gain: gain
    :param fs: sampling rate
    :return double 2-order coefficients
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


def highShelf(f0, gain=0., Q=1., fs=48000):
    """
    Biquad IIR High-Shelf Filter
    :param f0: center frequency
    :param Q: quality factor
    :param gain: gain
    :param fs: sampling rate
    :return double 2-order coefficients
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


def peakNotch(f0, gain=0., Q=1., fs=48000):
    """
    Biquad IIR Peak/Notch Filter
    :param f0: center frequency
    :param Q: quality factor
    :param gain: gain; the positive value is peak filter and the negative value is notch filter
    :param fs: sampling rate
    :return double 2-order coefficients
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

def notch(f0, Q=1., fs=48000):
    """
    Biquad IIR Notch Filter
    :param f0: center frequency
    :param Q: quality factor
    :param fs: sampling rate
    :return double 2-order coefficients
    """
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0 = 1
    b1 = -2 * np.cos(w0)
    b2 = 1
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])
    h = np.hstack((b / a[0], a / a[0]))
    return h
