import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Get Trigger Efficiency Rate scan plot')
parser.add_argument('--lightFile', default=None, help='File containing the trigger fraction for signal data')
parser.add_argument('--noiseFile', default=None, help='File containint the trigger fraction for noise data')
parser.add_argument('--reference', default=None, help='File containint the trigger fraction for reference data')
parser.add_argument('--flashHz', default=10000, help='Freqency of laser pulses (Hz)')
parser.add_argument('--noiseMHz', default=0.005, help='Assumed Noise frequency')
parser.add_argument('--fadcMHz', default=1000, help='FADC sampling frequency (MHz)')
parser.add_argument('--fadcBins', default=96, help='number of readout bins')
parser.add_argument('--pe2mv', default=2.33, help='conversion between p.e. and mV')
args = parser.parse_args()

signal = np.loadtxt(args.lightFile, unpack=True)
noise = np.loadtxt(args.noiseFile, unpack=True)

signal_rate = signal[1]*args.flashHz
noise_rate = 1000*args.fadcMHz*noise[1]/args.fadcBins

fig=plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.plot(signal[0], signal_rate, color='b', ls='--', label='Light Pulse')
ax1.plot(noise[0], noise_rate, color='r', ls='--', label='non pulsed background')
total = signal_rate[0:len(noise[0])]+noise_rate
total = np.append(total, signal_rate[len(noise[0]):])
ax1.plot(signal[0], total, color='k', label='total rate')
ax1.set_ylim([0,20000])
ax1.set_xlim([0,50])

ax1.set_xlabel('Discriminator Threshold [p.e.]')
ax1.set_ylabel('Rate [Hz]')

ax2 = ax1.twiny()
ax2.set_xlim([0,50*(float(args.pe2mv))])
ax2.set_xlabel('~Discriminator Threshold [mV]')

if args.reference is not None:
    lab = np.loadtxt(args.reference, unpack=True, delimiter=',')
    plt.plot(lab[0],lab[1], color='g', label='lab measurement')
    plt.plot(-10,-10, color='b', ls='--', label='laser pulse')
    plt.plot(-10,-10, color='r', ls='--', label='non pulsed background')
    plt.plot(-10,-10, color='k', label='total rate')
    ax2.set_ylim([0,20000])

plt.legend()

plt.show()
