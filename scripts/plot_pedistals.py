import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.targetioeventsource import TargetIOEventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.calib import CameraCalibrator
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from tqdm import tqdm
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.core import Tool
from traitlets import Dict, List, Int, Unicode, Bool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from os import listdir, path, makedirs
import argparse


def sort_array(a):
    a = a.T
    a = sorted(a,key=lambda  x : x[0])
    a = np.asarray(a)
    return a.T

def main():


    parser = argparse.ArgumentParser(description='Run LightEmission and simtel')
    parser.add_argument('--input_path', default='.', help='File containing run number, Npe and Nphotons')
    parser.add_argument('--out_file', default=None, help='name of output data file')
    parser.add_argument('--max_events', default=None, help='max number of events to include [NO FUNCTION]')
    parser.add_argument('--pe2mv', default=1, type=float, help='value to convert from p.e. to mV')
    parser.add_argument('--extra_pe2mv', default=1, type=float, help='value to convert from p.e. to mV')
    parser.add_argument('--extra', default=None, help='extra results file to read in and plot')
    parser.add_argument('--plothist', default=False, action='store_true')
    args = parser.parse_args()

    file_list = listdir('%s' % args.input_path)

    # if args.extra is not None:
    #     fig1 = plt.figure(1)
    #     ax1 = fig1.add_subplot(211)
    #     ax12 = fig1.add_subplot(212)
    #     ax1.set_title('baselineStartMean')
    #     ax1.set_xlabel('NSB Rate [MHz]')
    #     ax12.set_xlabel('NSB Rate [MHz]')
    #     fig2 = plt.figure(2)
    #     ax2 = fig2.add_subplot(211)
    #     ax22 = fig2.add_subplot(212)
    #     ax2.set_title('baselineStartRMS')
    #     ax2.set_xlabel('NSB Rate [MHz]')
    #     ax2.set_ylabel('Mean (1300) pulse baseline_rms/mV')
    #     ax22.set_xlabel('NSB Rate [MHz]')
    #     ax22.set_ylabel('Raitio Mean (1300) pulse baseline_rms/mV')
    #     fig3 = plt.figure(3)
    #     ax3 = fig3.add_subplot(211)
    #     ax32 = fig3.add_subplot(212)
    #     ax3.set_title('baselineEndMean')
    #     ax3.set_xlabel('NSB Rate [MHz]')
    #     ax32.set_xlabel('NSB Rate [MHz]')
    #     fig4 = plt.figure(4)
    #     ax4 = fig4.add_subplot(211)
    #     ax42 = fig4.add_subplot(212)
    #     ax4.set_title('baselineEndRMS')
    #     ax4.set_xlabel('NSB Rate [MHz]')
    #     ax42.set_xlabel('NSB Rate [MHz]')
    #     fig5 = plt.figure(5)
    #     ax5 = fig5.add_subplot(211)
    #     ax52 = fig5.add_subplot(212)
    #     ax5.set_title('baselineWaveformMean')
    #     ax5.set_xlabel('NSB Rate [MHz]')
    #     ax52.set_xlabel('NSB Rate [MHz]')
    #     fig6 = plt.figure(6)
    #     ax6 = fig6.add_subplot(211)
    #     ax62 = fig6.add_subplot(212)
    #     ax6.set_title('baselineWaveformRMS')
    #     ax6.set_xlabel('NSB Rate [MHz]')
    #     ax62.set_xlabel('NSB Rate [MHz]')
    # else:
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_title('baselineStartMean')
    ax1.set_xlabel('NSB Rate [MHz]')
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.set_title('baselineStartRMS')
    ax2.set_xlabel('NSB Rate [MHz]')
    ax2.set_ylabel('Mean (1300) pulse baseline_rms/mV')
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.set_title('baselineEndMean')
    ax3.set_xlabel('NSB Rate [MHz]')
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)
    ax4.set_title('baselineEndRMS')
    ax4.set_xlabel('NSB Rate [MHz]')
    fig5 = plt.figure(5)
    ax5 = fig5.add_subplot(111)
    ax5.set_title('baselineWaveformMean')
    ax5.set_xlabel('NSB Rate [MHz]')
    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(111)
    ax6.set_title('baselineWaveformRMS')
    ax6.set_xlabel('NSB Rate [MHz]')

    if args.out_file is not None:
        outfile = open(args.out_file, 'w')
        outfile.write('#NSBRate\tBSMean_mean\tBSMean_err\tBSRMS_mean\tBSRMS_err\t'
                      'BEMean_mean\tBEmean_err\tBERMS_mean\tBERMS_err\t'
                      'BWMean_mean\tBWMean_err\tBWRMS_mean\tBWRMS_err\n')

    results = np.zeros((7,len(file_list)))
    nsb = []
    bSMean =[]
    bSRMS =[]
    bMean =[]
    bERMS =[]
    bWMean =[]
    bWRMS =[]

    for n, fl in enumerate(file_list):
        print(fl)
        print(fl.split('_')[2][:-6])
        # exit()
        data = pd.read_hdf('%s/%s' % (args.input_path, fl), 'table')

        if args.plothist:
            if float(fl.split('_')[2][:-6]) not in [10.0,50.0,100.0,200.0,300.0,500.0]:
                continue

            ax1.hist(data['baselineStartMean'].values, bins=50, alpha=0.9, histtype='stepfilled',
                                 label='%s MHz' % fl.split('_')[2][:-3])
            ax2.hist(data['baselineStartRMS'].values, bins=50, alpha=0.9, histtype='stepfilled',
                                 label='%s MHz' % fl.split('_')[2][:-3])
            ax3.hist(data['baselineEndMean'].values, bins=50, alpha=0.9, histtype='stepfilled',
                                 label='%s MHz' % fl.split('_')[2][:-3])
            ax4.hist(data['baselineEndRMS'].values, bins=50, alpha=0.9, histtype='stepfilled',
                                 label='%s MHz' % fl.split('_')[2][:-3])
            ax5.hist(data['baselineWaveformMean'].values, bins=50, alpha=0.9, histtype='stepfilled',
                                 label='%s MHz' % fl.split('_')[2][:-3])
            ax6.hist(data['baselineWaveformRMS'].values, bins=50, alpha=0.9, histtype='stepfilled',
                                 label='%s MHz' % fl.split('_')[2][:-3])
        else:
            ax1.errorbar(float(fl.split('_')[2][:-6]), args.pe2mv*np.mean(data['baselineStartMean'].values),
                         yerr=np.std(args.pe2mv*data['baselineStartMean'].values),color='k', marker='.')
            ax2.errorbar(float(fl.split('_')[2][:-6]), args.pe2mv*np.mean(data['baselineStartRMS'].values),
                         yerr=np.std(args.pe2mv*data['baselineStartRMS'].values),color='k', marker='.')
            ax3.errorbar(float(fl.split('_')[2][:-6]), args.pe2mv*np.mean(data['baselineEndMean'].values),
                         yerr=np.std(args.pe2mv*data['baselineEndMean'].values),color='k', marker='.')
            ax4.errorbar(float(fl.split('_')[2][:-6]), args.pe2mv*np.mean(data['baselineEndRMS'].values),
                         yerr=np.std(args.pe2mv*data['baselineEndRMS'].values),color='k', marker='.')
            ax5.errorbar(float(fl.split('_')[2][:-6]), args.pe2mv*np.mean(data['baselineWaveformMean'].values),
                         yerr=np.std(args.pe2mv*data['baselineWaveformMean'].values),color='k', marker='.')
            ax6.errorbar(float(fl.split('_')[2][:-6]), args.pe2mv*np.mean(data['baselineWaveformRMS'].values),
                         yerr=np.std(args.pe2mv*data['baselineWaveformRMS'].values),color='k', marker='.')

            results[0][n] = float(fl.split('_')[2][:-6])
            results[1][n] = args.pe2mv*np.mean(data['baselineStartMean'].values)
            results[2][n] = args.pe2mv*np.mean(data['baselineStartRMS'].values)
            results[3][n] = args.pe2mv*np.mean(data['baselineEndMean'].values)
            results[4][n] = args.pe2mv*np.mean(data['baselineEndRMS'].values)
            results[5][n] = args.pe2mv*np.mean(data['baselineWaveformMean'].values)
            results[6][n] = args.pe2mv*np.mean(data['baselineWaveformRMS'].values)


            # nsb.append(float(fl.split('_')[2][:-6]))
            # bSMean.append(args.pe2mv*np.mean(data['baselineStartMean'].values))
            # bSRMS.append(args.pe2mv*np.mean(data['baselineStartRMS'].values))
            # bMean.append(args.pe2mv*np.mean(data['baselineEndMean'].values))
            # bERMS.append(args.pe2mv*np.mean(data['baselineEndRMS'].values))
            # bWMean.append(args.pe2mv*np.mean(data['baselineWaveformMean'].values))
            # bWRMS.append(args.pe2mv*np.mean(data['baselineWaveformRMS'].values))

        if args.out_file is not None:
            outfile.write('%s\t%s\t%s\t%s\t%s\t%s\t%s'
                          '\t%s\t%s\t%s\t%s\t%s\t%s\n' % (float(fl.split('_')[2][:-6]),
                                                          args.pe2mv * np.mean(data['baselineStartMean'].values),
                                                          np.std(args.pe2mv * data['baselineStartMean'].values),
                                                          args.pe2mv * np.mean(data['baselineStartRMS'].values),
                                                          np.std(args.pe2mv * data['baselineStartRMS'].values),
                                                          args.pe2mv * np.mean(data['baselineEndMean'].values),
                                                          np.std(args.pe2mv * data['baselineEndMean'].values),
                                                          np.std(args.pe2mv * data['baselineEndRMS'].values),
                                                          np.std(args.pe2mv * data['baselineEndRMS'].values),
                                                          args.pe2mv * np.mean(data['baselineWaveformMean'].values),
                                                          np.std(args.pe2mv * data['baselineWaveformMean'].values),
                                                          args.pe2mv * np.mean(data['baselineWaveformRMS'].values),
                                                          np.std(args.pe2mv * data['baselineWaveformRMS'].values)))
    if args.out_file is not None:
        outfile.close()

    if args.extra is not None:
        fl2 = np.loadtxt(args.extra,unpack=True)
        ax1.errorbar(fl2[0]+2,args.extra_pe2mv*fl2[1],yerr=args.extra_pe2mv*fl2[2],color='r', marker='.', label=args.extra, ls=' ')
        ax2.errorbar(fl2[0]+2,args.extra_pe2mv*fl2[3],yerr=args.extra_pe2mv*fl2[4],color='r', marker='.', label=args.extra, ls=' ')
        ax3.errorbar(fl2[0]+2,args.extra_pe2mv*fl2[5],yerr=args.extra_pe2mv*fl2[6],color='r', marker='.', label=args.extra, ls=' ')
        ax4.errorbar(fl2[0]+2,args.extra_pe2mv*fl2[7],yerr=args.extra_pe2mv*fl2[8],color='r', marker='.', label=args.extra, ls=' ')
        ax5.errorbar(fl2[0]+2,args.extra_pe2mv*fl2[9],yerr=args.extra_pe2mv*fl2[10],color='r', marker='.', label=args.extra, ls=' ')
        ax6.errorbar(fl2[0]+2,args.extra_pe2mv*fl2[11],yerr=args.extra_pe2mv*fl2[12],color='r', marker='.', label=args.extra, ls=' ')

        # results = sort_array(results)
        # fl2 = sort_array(fl2)
        #
        # ax12.scatter(fl2[0],(args.extra_pe2mv*fl2[1]-results[1])/results[1],color='r', marker='.', label=args.extra)
        # ax22.scatter(fl2[0],(args.extra_pe2mv*fl2[3]-results[2])/results[2],color='r', marker='.', label=args.extra)
        # ax32.scatter(fl2[0],(args.extra_pe2mv*fl2[5]-results[3])/results[3],color='r', marker='.', label=args.extra)
        # ax42.scatter(fl2[0],(args.extra_pe2mv*fl2[7]-results[4])/results[4],color='r', marker='.', label=args.extra)
        # ax52.scatter(fl2[0],(args.extra_pe2mv*fl2[9]-results[5])/results[5],color='r', marker='.', label=args.extra)
        # ax62.scatter(fl2[0],(args.extra_pe2mv*fl2[11]-results[6])/results[6],color='r', marker='.', label=args.extra)
        #
        # ax12.scatter(results[0],(results[1]-results[1])/results[1], color='k', marker='.')
        # ax22.scatter(results[0],(results[2]-results[2])/results[2], color='k', marker='.')
        # ax32.scatter(results[0],(results[3]-results[3])/results[3], color='k', marker='.')
        # ax42.scatter(results[0],(results[4]-results[4])/results[4], color='k', marker='.')
        # ax52.scatter(results[0],(results[5]-results[5])/results[5], color='k', marker='.')
        # ax62.scatter(results[0],(results[6]-results[6])/results[6], color='k', marker='.')

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax6.legend()
    plt.show()

if __name__ == '__main__':
    main()
