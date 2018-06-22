from ctapipe.calib import CameraCalibrator
import matplotlib.pyplot as plt
from tqdm import tqdm
from ctapipe.io import event_source
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Run LightEmission and simtel')
parser.add_argument('--mcfile', help='Path to mc data file')
parser.add_argument('--labfile', help='Path to lab data file')
parser.add_argument('--maxevents', default=1, help='maximum number of events to include')
parser.add_argument('--mc_pix', default=None, type=int, help='specify pixel number to plot in MC data file, otherwise loop over 64')
parser.add_argument('--lab_pix', default=None, type=int, help='specify pixel number to plot in LAB data file, otherwise loop over 64')
parser.add_argument('--mc_thresh', default=None, type=float, help='minimum y value required to plot waveform')
parser.add_argument('--lab_thresh', default=None, type=float, help='minimum y value required to plot waveform')
parser.add_argument('--mv2pe', default=1, type=float, help='input for simple conversion from mV to pe for lab data')
args = parser.parse_args()

fig0 = plt.figure(0)
# fig1 = plt.figure(1)
ax1 = fig0.add_subplot(121)
ax2 = fig0.add_subplot(122)

ax1.set_title('Lab data')
ax1.set_ylabel('amplitude')
ax1.set_xlabel('time [ns]')

ax2.set_title('MC data')
ax2.set_ylabel('amplitude')
ax2.set_xlabel('time [ns]')

######### CAMERA DATA p.e. ###########
cal = CameraCalibrator()
try:
    for event_r1 in tqdm(event_source(args.labfile, max_events=args.maxevents), total = args.maxevents):
        cal.calibrate(event_r1) # Not needed as it is already R1 data?
        print(event_r1.r1.tels_with_data)
        print('\n\nCamera Event\n\n')
        for tel in event_r1.r1.tels_with_data:
            if args.lab_pix == None:
                for i in range(0,64):
                    if args.lab_thresh==None:
                        ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])), event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe,
                                 color='C0')
                    else:
                        if max(event_r1.r1.tel[tel].waveform[0][i])/args.mv2pe > args.lab_thresh:
                            print('plotting only lab pixel: ', i)
                            ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])), event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe,color='C0')


except FileNotFoundError:
    print('LAB file_not_found')

######### MC DATA p.e. ###########
cal2 = CameraCalibrator()
try:
    for event_mc2 in tqdm(event_source(args.mcfile, max_events=args.maxevents), total=args.maxevents):
        cal2.calibrate(event_mc2)
        print('\n\nMonte Carlo Event\n\n')
        for tel in event_mc2.r1.tels_with_data:
            if args.mc_pix == None:
                for i in range(0,64):
                    if args.mc_thresh == None:
                        ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][i],color='C1')
                    else:
                        if max(event_mc2.r1.tel[tel].waveform[0][i])>args.mc_thresh:
                            print('plotting only mc pixel: ', i)
                            ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][i],color='C1')


except FileNotFoundError:
    print('MC file not found')

plt.show()
