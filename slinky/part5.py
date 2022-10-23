
'''
Amplitude measured when driving force less than natty frequency
'''
import wave
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math


def expo(x, a): #x is k
    # y=y0*a(e^(kx))
    return a*(np.e**(x))
    # return a*x
    # return ((a*np.exp(2))+(x*np.exp(2))*(b*np.exp(2)))*np.exp(1/2)
def twoExpo(x,a):
    return a*(np.e**x-np.e**(-1*x))

amp = np.array([0.052,0.034,0.02,0.009,0.005,0.002,0.002,0.001,0])*100
distance = np.arange(160,-20,-20) #0-180 with steps of 20
amp = np.array([2*np.e**0,2*np.e**1,2*np.e**2])
distance = np.array([0,1,2])
y0 = 0.052 #this is the driving amplitude

print(expo(2,1))
# popt, pcov = curve_fit(expo, distance, amp);
# plt.plot(distance, expo(distance, *popt), label='fit: a=%5.10f' %tuple(popt))
popt, pcov = curve_fit(expo, distance, amp);
plt.plot(distance, expo(distance, *popt), label='fit: a=%5.10f' %tuple(popt))
plt.plot(distance,amp)
# plt.gca().invert_xaxis()
plt.legend()
plt.show()

# #k = 2pi/lambda
# omega_sqare = np.power(omega,2)
# waveLength = np.array([1.1, 0.725,0.635,0.513])
# k = (2*np.pi)/waveLength
# k_unc = np.array([0.003, 0.006,0.008,0.01])*np.exp(2) #for erorr on square graph do u square uncertainity too
# omega_unc = np.array([0.6,0.7,0.9,1]) *np.exp(2)

# k_square = np.power(k,2)
# popt, pcov = curve_fit(func1, omega_sqare, k_square)

# print(func1(1,1,1))
# # plt.plot(k_square, omega_sqare)
# # plt.plot(k_square,omega_sqare, label ='measured')
# plt.errorbar(k_square,omega_sqare, xerr=k_unc,yerr=omega_unc, label='measured')
# plt.plot(k_square, func1(k_square, *popt), 'r-', label='fit: \u03C90=%5.3f, c=%5.3f' % tuple(popt))
# # plt.errorbar(k_square, func1(k_square, *popt), yerr=10)
# # # curve = func1(k,pcov[0,0]**(0.5),pcov[1,1]**(0.5))
# # plt.plot(k,func1(k, *popt))
# plt.legend()
# plt.show()


