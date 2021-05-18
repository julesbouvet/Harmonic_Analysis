import numpy as np
import matplotlib.pyplot as plt

# Creation of signals


def one_periodic_sinus(N, interval):
    t = np.linspace(interval[0], interval[1], N)
    f = np.sin(2*np.pi*t)
    return t, f


def add_sine(t, f):
    add_sine_to_f = f + 0.1*np.sin(2*np.pi*t*100)
    return t, add_sine_to_f


def add_linear(t, f):
    add_linear_to_f = f + 0.5*t+0.75
    return t, add_linear_to_f


def dft(t, f, N, fig_name):
    plt.title(fig_name)
    plt.subplot(1, 3, 1)
    plt.plot(t, f)

    yf = np.fft.fft(f)
    sp = np.abs(yf)
    freq = np.fft.fftfreq(N)*N
    plt.subplot(1, 3, 2)
    plt.plot(freq, sp)
    plt.xlim((-105, 105))

    phase = np.angle(yf)
    plt.subplot(1, 3, 3)
    plt.plot(t, phase)
    plt.title(fig_name)
    plt.show()
    return yf


def filtering(N, signal):
    signal_fft = np.fft.fft(signal)
    sample_freq = np.fft.fftfreq(N) * N

    # Find the peak frequency: we can focus on only the positive frequencies
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[signal_fft[pos_mask].argmax()]

    high_freq_fft = signal_fft.copy()
    high_freq_fft[np.abs(sample_freq) >= peak_freq-10] = 0
    filtered_sig = np.fft.ifft(high_freq_fft)

    plt.plot(sample_freq, signal.real)
    plt.plot(sample_freq, filtered_sig.real)
    plt.show()


if __name__ == "__main__":
    N = 1024  # nb of points

    # f1 : a one-periodic sine wave uniformly sampled on [−1/2, 1/2]
    t1, f1 = one_periodic_sinus(N, (-0.5, 0.5))

    # f2 : one-periodic sine wave uniformly sampled on [0,1]
    t2, f2 = one_periodic_sinus(N, (0, 1))

    # f3 : add a sine wave with 100 times the frequency and 1 of the amplitude of f1 to f1
    t3, f3 = add_sine(t1, f1)

    # f4 : add a sine wave with 100 times the frequency and 1 of the amplitude of f2 to f2
    t4, f4 = add_sine(t2, f2)

    # f5 : add a linear increasing signal starting in (−1/2, 1/2) and going to (1/2,1) to f3
    t5, f5 = add_linear(t3, f3)

    # Question (ii)
    question_2 = False
    if question_2:
        num = 1
        for (t, f) in [(t1, f1), (t2, f2), (t3, f3), (t4, f4), (t5, f5)]:
            name_signal = f'Analysis of signal f{num}'
            dft(t, f, N, name_signal)
            num += 1

    # Question (iii)
    filtering(N, f3)