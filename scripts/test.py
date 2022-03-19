import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

omega = 2 * np.pi / 0.1
b, a = signal.butter(10, omega, 'lowpass', analog=True)
domain = np.linspace(0, 200, 100)[..., np.newaxis] + np.linspace(0, 100, 100)[np.newaxis, ...]
w, h = signal.freqs(b, a, worN=domain)
plt.plot(w, np.abs(h))
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Response')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(omega, color='green') # cutoff frequency
plt.show()
