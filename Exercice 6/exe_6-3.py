import numpy as np


def exp_matrix(N, n):
    """
    :param N: int, comes from I_N = [-N/2, ..., N/2 - 1]
    :param n: int, comes from I_n = [-n/2, ..., n/2 - 1]
    :return: numpy array (N, n) exp(2i*pi*k*l/n) with k in I_N and l in I_n
    """
    I_N = np.linspace(-N/2, N/2 -1, 1)
    I_n = np.linspace(-n / 2, n / 2 - 1, 1)

    exp = np.zeros((N, n))

    for k in range(N):
        for l in range(n):
            exp[k, l] = np.exp(2j*np.pi*k*l/n)

    return exp


def NFFT(signal, phi, ck, sigma):
    N = signal.shape[0]

    # compute the g_k
    g_k = signal/(N*ck)

    # compute the g_l
    n = sigma*N
    exp_mat = exp_matrix(N, n)
    g_l = (1/n)*np.matmul(exp_mat.T, g_k)

    # compute the f_j
    ## ??
