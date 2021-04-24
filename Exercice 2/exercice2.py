import numpy as np
import scipy
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt


def convmtx(w,n):
    return scipy.linalg.toeplitz(np.hstack([w,np.zeros(2*n-len(w)-1)]),np.hstack([w[0],np.zeros(n-1)]))


def load_signals():
    sig=scipy.io.loadmat('signals')
    f=sig['f']
    g=sig['g']
    return np.reshape(f, f.shape[1]), np.reshape(g, g.shape[1])


def plot_signals_from_file():
    f,g=load_signals()
    fig=plt.figure()
    plt.plot(f)    
    plt.plot(g)    
    plt.legend(['original','noisy'])
    plt.show()
    plt.close(fig)


# Question i)

def apply_filter(f, filter, name):
    u = np.convolve(f, filter)
    plt.plot(u)
    plt.title(' Convolution with the '+ name +' filter')
    plt.show()
    return u

# Question ii)


def sampled_gaussian(std, nb_points):
    x_input = np.linspace(-3*std, 3*std, nb_points)
    weight_gauss = []
    for x in x_input:
        gauss = (1/np.sqrt(2*np.pi*(std**2)))*np.exp(-x**2/(2*std**2))
        weight_gauss.append(gauss)
    return weight_gauss


def conv_gauss(f, std, nb_points):
    gauss = sampled_gaussian(std, nb_points)
    w_conv_matrix = convmtx(gauss, f.shape[0])
    conv = np.matmul(w_conv_matrix, f)
    plt.plot(conv)
    plt.title(f'Gaussian convolution with std={std} and {nb_points} points')
    plt.show()


if __name__ == '__main__':
    f, g = load_signals()
    plot_signals_from_file()

    # Q1)
    show_question_1 = False

    if show_question_1 == True:
        filter_w1 = [-1, 1]+[0 for i in range(f.shape[0]-2)]
        filter_w2 = [1, 2, 1] + [0 for i in range(f.shape[0]-3)]

        u_w1 = apply_filter(f, filter_w1, 'first')
        u_w2 = apply_filter(f, filter_w2, 'second')

    # Q2)
    show_question_2 = True

    if show_question_2==True:
        std=5
        nb_points=100
        conv_gauss(g, std=std, nb_points=nb_points)
