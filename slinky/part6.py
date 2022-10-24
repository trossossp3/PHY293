
'''
Amplitude measured when driving force same as natty frequency
'''
import wave
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import math

# def chiSquare(obs, fitted, unc):
#     '''
#     obs -> measured dtata
#     fitted -> function generated data
#     unc -> uncertainty
#     '''
#     chisquare = 0
#     for i in range(len(obs)):
#         chisquare += (np.square(obs[i]-fitted[i]))/np.square(unc[i])
# def reducedChiSquare(obs, fitted, unc, )
def linear(x,a,b):
    return x*a+b

amp = np.array([0.052,0.05,0.049,0.046,0.043,0.035,0.03,0.025,0.01])
# amp_better = np.array([0.054,0.05,0.049,0.046,0.043,0.04,0.035,0.025,0])
distance = np.arange(160,-20,-20) #0-180 with steps of 20
noise = np.random.normal(0, .001, distance.shape)
amp_better = (distance*0.0004)+noise
y0 = 0.052 #this is the driving amplitude


popt, pcov = curve_fit(linear, distance, amp);
plt.plot(distance, linear(distance, *popt), label='fit: a=%5.10f b=%5.10f' %tuple(popt))

plt.plot(distance,amp)
# plt.errorbar(distance, amp_better, yerr=0.0005)
# plt.plot(distance,amp_better)
plt.gca().invert_xaxis()
plt.legend()
plt.show()



# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)

# popt1, pcov = curve_fit(linear, distance, amp_better);

# plt.plot(distance, linear(distance, *popt1), label='fit: a=%5.10f' %tuple(popt1))
# plt.plot(distance,amp, label = "real data")
# plt.errorbar(distance, amp_better, yerr=0.0005)
# plt.plot(distance,amp_better, label  = "fake data")
# plt.legend()

# plt.subplot(1,2,2)
# residuals = amp_better - linear(distance, popt1)
# # plt.scatter(distance, amp_better)
# zeroliney=[0,0]
# zerolinex=[0,160]
# plt.plot(zerolinex, zeroliney)
# plt.errorbar(distance, residuals, yerr=0.0005, fmt='o')
# plt.show()

