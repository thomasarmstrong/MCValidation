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


def main():


    parser = argparse.ArgumentParser(description='Run LightEmission and simtel')
    parser.add_argument('--input_path', default='.', help='File containing run number, Npe and Nphotons')
    parser.add_argument('--outdir', default='/scratch/armstrongt/Workspace/CTA/MCValidation/data')
    parser.add_argument('--max_events', default='/scratch/armstrongt/Workspace/CTA/MCValidation/data')
    args = parser.parse_args()

    file_list = listdir('%s' % args.input_path)

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

    for fl in file_list:
        print(fl)
        print(fl.split('_')[2][:-6])
        # exit()
        data = pd.read_hdf('%s/%s' % (args.input_path, fl), 'table')

        # ax1.hist(data['baselineStartMean'].values, bins=50, alpha=0.9, histtype='stepfilled',
        #                      label='%s MHz' % fl.split('_')[2][:-3])
        # ax2.hist(data['baselineStartRMS'].values, bins=50, alpha=0.9, histtype='stepfilled',
        #                      label='%s MHz' % fl.split('_')[2][:-3])
        # ax3.hist(data['baselineEndMean'].values, bins=50, alpha=0.9, histtype='stepfilled',
        #                      label='%s MHz' % fl.split('_')[2][:-3])
        # ax4.hist(data['baselineEndRMS'].values, bins=50, alpha=0.9, histtype='stepfilled',
        #                      label='%s MHz' % fl.split('_')[2][:-3])
        # ax5.hist(data['baselineWaveformMean'].values, bins=50, alpha=0.9, histtype='stepfilled',
        #                      label='%s MHz' % fl.split('_')[2][:-3])
        # ax6.hist(data['baselineWaveformRMS'].values, bins=50, alpha=0.9, histtype='stepfilled',
        #                      label='%s MHz' % fl.split('_')[2][:-3])


        ax1.errorbar(float(fl.split('_')[2][:-6]), np.mean(data['baselineStartMean'].values), yerr=np.std(data['baselineStartMean'].values),color='k', marker='.')
        ax2.errorbar(float(fl.split('_')[2][:-6]), np.mean(data['baselineStartRMS'].values), yerr=np.std(data['baselineStartRMS'].values),color='k', marker='.')
        ax3.errorbar(float(fl.split('_')[2][:-6]), np.mean(data['baselineEndMean'].values), yerr=np.std(data['baselineEndMean'].values),color='k', marker='.')
        ax4.errorbar(float(fl.split('_')[2][:-6]), np.mean(data['baselineEndRMS'].values), yerr=np.std(data['baselineEndRMS'].values),color='k', marker='.')
        ax5.errorbar(float(fl.split('_')[2][:-6]), np.mean(data['baselineWaveformMean'].values), yerr=np.std(data['baselineWaveformMean'].values),color='k', marker='.')
        ax6.errorbar(float(fl.split('_')[2][:-6]), np.mean(data['baselineWaveformRMS'].values), yerr=np.std(data['baselineWaveformRMS'].values),color='k', marker='.')


    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    plt.show()

if __name__ == '__main__':
    main()
