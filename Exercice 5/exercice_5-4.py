import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# load signal
load_data = sio.loadmat('gong.mat')
signal = load_data['y']
Fs = load_data['Fs']
print(signal.shape)

signal_fft = np.fft.fft(signal)
modified_signal_fft = [0 for i in range(5000)]+list(signal_fft)[5000:-5000]+ [0 for i in range(5000)]
new_signal = np.fft.ifft(modified_signal_fft)

print(new_signal)