from ctapipe.calib import CameraCalibrator
import matplotlib.pyplot as plt
from tqdm import tqdm
from ctapipe.io import event_source
import argparse
from ctapipe.io.eventsourcefactory import EventSourceFactory
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import UnivariateSpline
import numpy as np

parser = argparse.ArgumentParser(description='Run LightEmission and simtel')
parser.add_argument('--mcfile', default=None, help='Path to mc data file')
parser.add_argument('--labfile', default=None, help='Path to lab data file')
parser.add_argument('--maxevents', default=1, type=int, help='maximum number of events to include')
parser.add_argument('--mc_pix', default=None, type=int, help='specify pixel number to plot in MC data file, otherwise loop over 64')
parser.add_argument('--lab_pix', default=None, type=int, help='specify pixel number to plot in LAB data file, otherwise loop over 64')
parser.add_argument('--max_pix', default=None, type=int, help='maximum number of pixels, default from geom')
parser.add_argument('--mc_thresh', default=None, type=float, help='minimum y value required to plot waveform')
parser.add_argument('--lab_thresh', default=None, type=float, help='minimum y value required to plot waveform')
parser.add_argument('--mv2pe', default=1, type=float, help='input for simple conversion from mV to pe for lab data')
args = parser.parse_args()

fig0 = plt.figure(0)
# fig1 = plt.figure(1)
if args.mcfile is None:
    ax1 = fig0.add_subplot(111)
    ax1.set_title('Lab data')
    ax1.set_ylabel('amplitude')
    ax1.set_xlabel('time [ns]')
if args.labfile is None:
    ax2 = fig0.add_subplot(111)
    ax2.set_title('MC data')
    ax2.set_ylabel('amplitude')
    ax2.set_xlabel('time [ns]')
if args.labfile is not None and args.mcfile is not None:
    ax1 = fig0.add_subplot(121)
    ax2 = fig0.add_subplot(122)

    ax1.set_title('Lab data')
    ax1.set_ylabel('amplitude')
    ax1.set_xlabel('time [ns]')

    ax2.set_title('MC data')
    ax2.set_ylabel('amplitude')
    ax2.set_xlabel('time [ns]')

def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))


######### CAMERA DATA p.e. ###########


try:
    if args.labfile is not None:
        cal = CameraCalibrator(eventsource=EventSourceFactory.produce(input_url=args.labfile, max_events=1))
        for event_r1 in tqdm(event_source(args.labfile, max_events=args.maxevents), total = args.maxevents):
            cal.calibrate(event_r1) # Not needed as it is already R1 data?
            print('\n\nCamera Event\n\n')
            for tel in event_r1.r1.tels_with_data:
                geom = event_r1.inst.subarray.tel[tel].camera
                if args.max_pix==None:
                    args.max_pix = len(geom.pix_x)
                if args.lab_pix == None:
                    for i in range(0,args.max_pix):
                        if args.lab_thresh==None:
                            ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])), event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe,
                                     color='C0')
                            n=len(event_r1.r1.tel[tel].waveform[0][0])
                            mean = sum(range(len(event_r1.r1.tel[tel].waveform[0][0])) * (event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe))/n
                            sigma = sum((event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe)* (range(len(event_r1.r1.tel[tel].waveform[0][0]))-mean)**2 )/n
                            popt, pcov = curve_fit(gaus, range(len(event_r1.r1.tel[tel].waveform[0][0])), event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe, p0=[1, mean, sigma])
                            ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])), gaus(range(len(event_r1.r1.tel[tel].waveform[0][0])), *popt))
                            spline = UnivariateSpline(range(len(event_r1.r1.tel[tel].waveform[0][0])), gaus(range(len(event_r1.r1.tel[tel].waveform[0][0])), *popt) -np.max(gaus(range(len(event_r1.r1.tel[tel].waveform[0][0])), *popt))/2, s=0)
                            r1, r2 = spline.roots()
                            ax1.axvspan(r1,r2, facecolor='g', alpha=0.5)
                        else:
                            if max(event_r1.r1.tel[tel].waveform[0][i])/args.mv2pe > args.lab_thresh:
                                print('plotting only lab pixel: ', i)
                                ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])), event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe,color='C0')
                                n=len(event_r1.r1.tel[tel].waveform[0][0])
                                mean = sum(range(len(event_r1.r1.tel[tel].waveform[0][0])) * (event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe))/n
                                sigma = sum((event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe)* (range(len(event_r1.r1.tel[tel].waveform[0][0]))-mean)**2 )/n
       	       	       	        popt, pcov = curve_fit(gaus, range(len(event_r1.r1.tel[tel].waveform[0][0])), event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe, p0=[1, mean, sigma])
       	       	       	        ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])), gaus(range(len(event_r1.r1.tel[tel].waveform[0][0])), *popt))
                                spline = UnivariateSpline(range(len(event_r1.r1.tel[tel].waveform[0][0])), gaus(range(len(event_r1.r1.tel[tel].waveform[0][0])), *popt) -np.max(gaus(range(len(event_r1.r1.tel[tel].waveform[0][0])), *popt))/2, s=0)
                                r1, r2 = spline.roots()
                                ax1.axvspan(r1,r2, facecolor='g', alpha=0.5)
                else:
                    ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][0])),
                             event_r1.r1.tel[tel].waveform[0][args.lab_pix] / args.mv2pe,
                             color='C0')
                    n=len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])
                    mean = sum(range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])) * (event_r1.r1.tel[tel].waveform[0][args.lab_pix]/args.mv2pe))/n
                    sigma = sum((event_r1.r1.tel[tel].waveform[0][i]/args.mv2pe)* (range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix]))-mean)**2 )/n
                    popt, pcov = curve_fit(gaus, range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])), event_r1.r1.tel[tel].waveform[0][args.lab_pix]/args.mv2pe, p0=[1, mean, sigma])
                    ax1.plot(range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])), gaus(range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])), *popt))
       	       	    spline = UnivariateSpline(range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])), gaus(range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])), *popt) -np.max(gaus(range(len(event_r1.r1.tel[tel].waveform[0][args.lab_pix])), *popt))/2, s=0)
       	       	    r1, r2 = spline.roots()
       	       	    ax1.axvspan(r1,r2, facecolor='g', alpha=0.5)

except FileNotFoundError:
    print('LAB file_not_found')

######### MC DATA p.e. ###########

try:
    if args.mcfile is not None:
        cal2 = CameraCalibrator(eventsource=EventSourceFactory.produce(input_url=args.mcfile, max_events=1))

        for event_mc2 in tqdm(event_source(args.mcfile, max_events=args.maxevents), total=args.maxevents):
            cal2.calibrate(event_mc2)

            print('\n\nMonte Carlo Event\n\n')
            for tel in event_mc2.r1.tels_with_data:
                geom = event_mc2.inst.subarray.tel[tel].camera
                if args.max_pix==None:
                    args.max_pix = len(geom.pix_x)
                if args.mc_pix == None:
                    for i in range(0,args.max_pix ):
                        if args.mc_thresh == None:
                            ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][i],color='C1')
                            n=len(event_mc2.r1.tel[tel].waveform[0][0])
                            mean = sum(range(len(event_mc2.r1.tel[tel].waveform[0][0])) * (event_mc2.r1.tel[tel].waveform[0][i]))/n
                            sigma = sum((event_mc2.r1.tel[tel].waveform[0][i])* (range(len(event_mc2.r1.tel[tel].waveform[0][0]))-mean)**2 )/n
                            #popt, pcov = curve_fit(gaus, range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][i], p0=[1, mean, sigma])
                            #ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), gaus(range(len(event_mc2.r1.tel[tel].waveform[0][0])), *popt))
                            #spline = UnivariateSpline(range(len(event_mc2.r1.tel[tel].waveform[0][0])), gaus(range(len(event_mc2.r1.tel[tel].waveform[0][0])), *popt) -np.max(gaus(range(len(event_mc2.r1.tel[tel].waveform[0][0])), *popt))/2, s=0)
                            #r1, r2 = spline.roots()
                            #ax2.axvspan(r1,r2, facecolor='g', alpha=0.5)
                        else:
                            if max(event_mc2.r1.tel[tel].waveform[0][i])>args.mc_thresh:
                                print('plotting only mc pixel: ', i)
                                ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][i],color='C1')
                                n=len(event_mc2.r1.tel[tel].waveform[0][0])
                                mean = sum(range(len(event_mc2.r1.tel[tel].waveform[0][0])) * (event_mc2.r1.tel[tel].waveform[0][i]))/n
                                sigma = sum((event_mc2.r1.tel[tel].waveform[0][i])* (range(len(event_mc2.r1.tel[tel].waveform[0][0]))-mean)**2 )/n
                                #popt, pcov = curve_fit(gaus, range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][i], p0=[1, mean, sigma])
                                #ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), gaus(range(len(event_mc2.r1.tel[tel].waveform[0][0])), *popt))
                                #spline = UnivariateSpline(range(len(event_mc2.r1.tel[tel].waveform[0][0])), gaus(range(len(event_mc2.r1.tel[tel].waveform[0][0])), *popt) -np.max(gaus(range(len(event_mc2.r1.tel[tel].waveform[0][0])), *popt))/2, s=0)
                                #r1, r2 = spline.roots()
                                #ax2.axvspan(r1,r2, facecolor='g', alpha=0.5)
                else:
                    ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][0])), event_mc2.r1.tel[tel].waveform[0][args.mc_pix],
                             color='C1')
                    n=len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])
                    mean = sum(range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])) * (event_mc2.r1.tel[tel].waveform[0][args.mc_pix]))/n
                    sigma = sum((event_mc2.r1.tel[tel].waveform[0][args.mc_pix])* (range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix]))-mean)**2 )/n
                    #popt, pcov = curve_fit(gaus, range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])), event_mc2.r1.tel[tel].waveform[0][args.mc_pix], p0=[1, mean, sigma])
                    #ax2.plot(range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])), gaus(range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])), *popt))
                    #spline = UnivariateSpline(range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])), gaus(range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])), *popt) -np.max(gaus(range(len(event_mc2.r1.tel[tel].waveform[0][args.mc_pix])), *popt))/2, s=0)
                    #r1, r2 = spline.roots()
                    #ax2.axvspan(r1,r2, facecolor='g', alpha=0.5)

except FileNotFoundError:
    print('MC file not found')

plt.show()
