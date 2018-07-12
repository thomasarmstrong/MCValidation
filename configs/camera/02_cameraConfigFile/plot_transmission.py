import numpy as np
import matplotlib.pyplot as plt

theta = np.loadtxt('transmission_pmma_vs_theta_20150422.dat', unpack=True)
wave = np.loadtxt('transmission_pmma_vs_lambda_meas0deg_coat_82raws.dat', unpack=True)

fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1=fig1.add_subplot(111)
ax2=fig2.add_subplot(111)

ax1.plot(theta[0], theta[1])
ax1.set_xlabel('Angle [deg]')
ax1.set_ylabel('Transmission')

ax2.plot(wave[0], wave[1], label='old')
ax2.set_xlabel('Wavelength [nm]')
ax2.set_ylabel('Transmission')

plt.show()