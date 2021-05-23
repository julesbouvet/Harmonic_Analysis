import numpy as np

def is_prime(x):
    prime = True
    if x>1:
        for i in range(2, x):
            if (x%i) == 0:
                prime = False
                div = i
                break

    if prime == True:
        div = x
    return prime, div


def FFT(signal):
    print(signal.shape[0])
    N = signal.shape[0]
    prime, div = is_prime(N)

    if prime == True:
        a=1
    else:
        while is_prime(div)[0] == False:
            div = is_prime(div)[1]

    return div


div = FFT(np.zeros(3))
print(div)

