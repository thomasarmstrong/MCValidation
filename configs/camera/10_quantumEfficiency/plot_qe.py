import numpy as np
import matplotlib.pyplot as plt

u25 = np.loadtxt('S10362-11-025U.txt', delimiter=',', unpack=True)
u50 = np.loadtxt('S10362-11-050U.txt', delimiter=',', unpack=True)
u100 = np.loadtxt('S10362-11-100U.txt', delimiter=',', unpack=True)
old = np.loadtxt('PDE_ASTRI_LCT5_75um_OV_4V_meas.dat', unpack=True)

fig1 = plt.figure(1)
ax1=fig1.add_subplot(111)
ax1.plot(u25[0], u25[1], label='S10362-11-025U')
ax1.plot(u50[0],u50[1], label='S10362-11-050U')
ax1.plot(u100[0],u100[1], label='S10362-11-100U')
ax1.legend()
ax1.set_xlabel('wavelength [nm]')
ax1.set_ylabel('PDE')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(u50[0],0.305*u50[1]/u50[1][867], label='OV=3V')
ax2.plot(u50[0],0.325*u50[1]/u50[1][867], label='OV=3.5V')
ax2.plot(u50[0],0.34*u50[1]/u50[1][867], label='OV=4V')
ax2.plot(old[0], old[1], label='Prod3')
ax2.legend()
ax2.set_xlabel('wavelength [nm]')
ax2.set_ylabel('PDE')
ax2.set_title('S10362-11-050U')

out_file3v = open('S10362-11-050U-3V.txt','w')
# out_file35v = open('S10362-11-050U-35V.txt','w')
# out_file4v = open('S10362-11-050U-4V.txt','w')
for i in range(len(u50[1])):
    out_file3v.write('%s\t%s\n' % (u50[0][i], 0.305*u50[1][i]/u50[1][867]))
    # out_file35v.write('%s\t%s\n' % (u50[1][i], 0.325*u50[1]/u50[1][867]))
    # out_file4v.write('%s\t%s\n' % (u50[1][i], 0.34*u50[1]/u50[1][867]))

out_file3v.close()
# out_file35v.close()
# out_file4v.close()

plt.show()