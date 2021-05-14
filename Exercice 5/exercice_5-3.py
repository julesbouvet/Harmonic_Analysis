import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

telefonsignal = sio.loadmat('telefonsignal.mat')

signal_1 = telefonsignal['signal_1'].reshape(4096, 1)
signal_2 = telefonsignal['signal_2'].reshape(4096, 1)
signal_3 = telefonsignal['signal_3'].reshape(4096, 1)

N = 4096
Ts = 1/4096


def fourier_matrix(n):
    F = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            F[i, j] = np.exp(-2j*np.pi*i*j/n)

    return F

dft_s1 = np.zeros((N, 1), dtype=complex)
dft_s2 = np.zeros((N, 1), dtype=complex)
dft_s3 = np.zeros((N, 1), dtype=complex)
Fn = fourier_matrix(N)

freqs = np.fft.fftfreq(N) * (1/Ts)

for k in range(N):
    dft_s1[k] = np.matmul(Fn[:, k].reshape(N, 1).T, signal_1)
    dft_s2[k] = np.matmul(Fn[:, k].reshape(N, 1).T, signal_2)
    dft_s3[k] = np.matmul(Fn[:, k].reshape(N, 1).T, signal_3)


print('Frequencies for signal 1: ', np.where(dft_s1 > 1000))
print('Frequencies for signal 2: ', np.where(dft_s2 > 1000))
print('Frequencies for signal 3: ', np.where(dft_s3 > 1000))

plt.plot(freqs, np.abs(dft_s1))
plt.plot(freqs, np.abs(dft_s2))
plt.plot(freqs, np.abs(dft_s3))
plt.show()