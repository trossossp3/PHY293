import wave
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math


# omega = sqrt(omega0^2 + c0^2k^2
def func1(x, a, b): #x is k
    return (a**2+b**2*x)
    # return ((a*np.exp(2))+(x*np.exp(2))*(b*np.exp(2)))*np.exp(1/2)

omega = np.array([5.955625884,7.383296483,8.787671758,13.03565416])

#k = 2pi/lambda
omega_sqare = np.power(omega,2)
waveLength = np.array([1.1, 0.725,0.635,0.513])
k = (2*np.pi)/waveLength
k_unc = np.array([0.003, 0.006,0.008,0.01])*np.exp(2) #for erorr on square graph do u square uncertainity too
omega_unc = np.array([0.6,0.7,0.9,1]) *np.exp(2)

k_square = np.power(k,2)
popt, pcov = curve_fit(func1, omega_sqare, k_square)

print(func1(1,1,1))
# plt.plot(k_square, omega_sqare)
# plt.plot(k_square,omega_sqare, label ='measured')
plt.errorbar(k_square,omega_sqare, xerr=k_unc,yerr=omega_unc, label='measured')
plt.plot(k_square, func1(k_square, *popt), 'r-', label='fit: \u03C90=%5.3f, c=%5.3f' % tuple(popt))
# plt.errorbar(k_square, func1(k_square, *popt), yerr=10)
# # curve = func1(k,pcov[0,0]**(0.5),pcov[1,1]**(0.5))
# plt.plot(k,func1(k, *popt))
plt.legend()
plt.show()


