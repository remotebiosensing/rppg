import numpy as np
import matplotlib.pyplot as plt
import math
import h5py

# Fs = 500
# T = 1 / Fs
# te = 0.5
# t = np.arange(0, te, T)
# print(t)
# print(len(t))
# print()
# # noise = np.random.normal(0, 0.01, len(t))
#
# x = 0.5 * np.sin(2 * np.pi * 60 * t)
# y = x# + noise
#
# plt.figure(num=1, dpi=100, facecolor='white')
# plt.plot(t, y, 'r')
# # plt.xlim(0, 0.111)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()


with h5py.File('/home/paperc/PycharmProjects/dataset/BPNet_uci/case(P+V+A)_750_train(cv1).hdf5') as f:
    train_ple = np.array(f['train/ple/0'])
    train_abp = np.array(f['train/abp/0'])

from scipy import signal

# t = np.arange(0,6,1/125)
Fs = 60
t = np.arange(0, 6, 1 / Fs)
print(len(t))

idx = 1

y = train_ple[idx][0]
abp = train_abp[idx]

y_temp = y
abp_temp = abp
print('Amplitude before dc removal :', np.max(y) - np.min(y))
# y = signal.resample(y, 750)
y = y - np.mean(y)
abp = abp - np.mean(abp)
print('ratio :', np.max(abp)-np.min(abp) / np.max(y)-np.min(y))
vpg = train_ple[0][1]
apg = train_ple[0][2]
# plt.plot(t, y)
# plt.show()
print('Amplitude after dc removal', np.max(y) - np.min(y))

n = len(y)
NFFT = n
k = np.arange(NFFT)
f0 = k * Fs / NFFT
f0 = f0[range(math.trunc(NFFT / 2))]

Y = np.fft.fft(y) / NFFT
Y = Y[range(math.trunc(NFFT / 2))]
A = np.fft.fft(abp) / NFFT
A = A[range(math.trunc(NFFT / 2))]
amplitude_Hz = 2 * abs(Y)
amplitude_Hz_abp = 2 * abs(A)
phase_ang = np.angle(Y) * 180 / np.pi
phase_ang_abp = np.angle(A) * 180 / np.pi

# figure 1 ..................................
plt.figure(num=2, dpi=100, facecolor='white')
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.subplot(4, 1, 1)

plt.plot(t, y, 'r')
plt.plot(t, y_temp,'k')
# plt.plot(t, abp, 'b')
plt.title('Signal FFT analysis')
plt.xlabel('time($sec$)')
plt.ylabel('y')
# plt.xlim( 0, 0.1)

''''''
# figure 1 ..................................
plt.figure(num=2, dpi=100, facecolor='white')
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.subplot(4, 1, 2)

plt.plot(t, abp, 'b')
plt.plot(t, abp_temp, 'k')
plt.xlabel('time($sec$)')
plt.ylabel('y')
# plt.xlim( 0, 0.1)
''''''
# Amplitude ....
# plt.figure(num=2,dpi=100,facecolor='white')
plt.subplot(4, 1, 3)

# Plot single-sided amplitude spectrum.

plt.plot(f0, amplitude_Hz, 'r')  # 2* ???
plt.plot(f0, amplitude_Hz_abp, 'b')
plt.xticks(np.arange(0, 30, 1))
plt.xlim(0, 30)
plt.ylim(0, 50)
# plt.title('Single-Sided Amplitude Spectrum of y(t)')
plt.xlabel('frequency($Hz$)')
plt.ylabel('amplitude')
plt.grid()

# Phase ....
# plt.figure(num=2,dpi=100,facecolor='white')
plt.subplot(4, 1, 4)
plt.plot(f0, phase_ang, 'r')  # 2* ???
plt.plot(f0, phase_ang_abp, 'b')
plt.xlim(0, 30)
plt.ylim(-180, 180)
# plt.title('Single-Sided Phase Spectrum of y(t)')
plt.xlabel('frequency($Hz$)')
plt.ylabel('phase($deg.$)')
plt.xticks(np.arange(0, 30, 1))
plt.yticks([-180, -90, 0, 90, 180])
plt.grid()

plt.savefig("./test_figure2.png", dpi=300)
plt.show()

print(sorted(abs(amplitude_Hz), reverse=True)[:5])

print(abs(amplitude_Hz).argsort()[-1] * 0.16666667)
print(abs(amplitude_Hz).argsort()[-2] * 0.16666667)
print(abs(amplitude_Hz).argsort()[-3] * 0.16666667)
print(abs(amplitude_Hz).argsort()[-4] * 0.16666667)
print(abs(amplitude_Hz).argsort()[-5] * 0.16666667)
# print(f0)
# print(len(f0))

# 1.2335811384723962e-16, 9.220390006653997e-05, 0.00011792373631798039, 0.00012569873377203113, 0.00014471788133792017


# test_ppg = 0.787147*np.sin(2*np.pi*)

import matplotlib.pyplot as plt
t = np.arange(0, 6, 1 / 60)
y = 0.712*np.sin(2*np.pi*1.5*t-np.pi*1.1) + 0.23525*np.sin(2*np.pi*3.0*t+np.pi*0.2)#+0.129*(np.sin(2*np.pi*1.3*t-np.pi))
plt.plot(t,y,'r',label='sin')
plt.plot(t,y_temp,'b',label='target')
# [ < matplotlib.lines.Line2D object at... >]
# plt.plot(t, s.imag, '--', label='imaginary')
# [ < matplotlib.lines.Line2D object at... >]
plt.legend()
# < matplotlib.legend.Legend object at... >
plt.show()
