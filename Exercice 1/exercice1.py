import numpy as np
import matplotlib.pyplot as plt


def dirichelet(interval, step_size, n):
    nb_points = int((interval[1]-interval[0])/step_size)
    x = np.linspace(interval[0], interval[1], nb_points)
    d_re = []
    d_im = []
    for xn in x:
        dn=0
        for k in range(-n, n):
            dn += np.exp(2j*np.pi*k*xn)
        d_re.append(dn.real)
        d_im.append(dn.imag)
    return x, d_re, d_im


if __name__ == '__main__':
    interval = [-0.5, 0.5]
    step_size = 0.001
    n = 16
    x, d_re, d_im = dirichelet(interval, step_size, n)
    plt.figure()
    plt.plot(d_re, d_im)
    plt.show()