import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import chirp
from scipy.fftpack import hilbert,ihilbert

duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs

signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(signal)
ihil = ihilbert(analytic_signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

print("plot")
plt.rcParams["figure.figsize"] = (14, 5)
plt.plot(range(len(signal[0:300])), signal[0:300], label='inference')
plt.plot(range(len(ihil[0:300])), ihil[0:300], label='target')
plt.legend(fontsize='x-large')
plt.show()