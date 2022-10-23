import wave
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math

def func1(x, a): #x is k
    return a*x

omega = np.array([5.955625884,7.383296483,8.787671758,13.03565416])

#k = 2pi/lambda
omega_sqare = np.power(omega,2)
waveLength = np.array([1.1, 0.725,0.635,0.513])
k = (2*np.pi)/waveLength

k_square = np.power(k,2)
xs = np.array([1,2,3,4,5,6,7])
ys = np.array([2,4,6,8,10,12,14])
popt, pcov = curve_fit(func1, xs, ys)


plt.plot(k_square, omega_sqare)
# plt.plot(k, func1(k, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
# curve = 
# plt.plot(xs,func1(xs, *popt), 'g--', label='fit: a=%5.3f' % tuple(popt))
plt.legend()
plt.show()


