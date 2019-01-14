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
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.interpolate import UnivariateSpline
import numpy as np

###### ToDo Define inputs via some sort of argparser, for now hardcode

def gaus(x, a, x0, sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))



class PedestalGenerator(Tool):
    name = "PedestalGenerator"
    description = "Generate the a pickle file of Pedestals for " \
                  "either MC or data files."

    telescopes = Int(1,help='Telescopes to include from the event file. '
                           'Default = 1').tag(config=True)
    pixel = Int(None, allow_none=True, help='Which pixel to use, defaul = all').tag(config=True)
    output_name = Unicode('extracted_pedestals',
                          help='path where to store the output extracted pedestal hdf5 '
                               'file').tag(config=True)
    input_path = Unicode(help='Path to directory containing data').tag(config=True)

    max_events = Int(1, help='Maximum number of events to use').tag(config=True)

    t0 = Int(0, help='Timeslice to start pedestal').tag(config=True)

    window_width = Int(10, help='length of window within which to determine pedestal').tag(config=True)

    debug = Bool(False, "plot resulting histograms").tag(config=True)

    aliases = Dict(dict(input_path='PedestalGenerator.input_path',
                        max_events='PedestalGenerator.max_events',
                        window_width='PedestalGenerator.window_width',
                        t0='PedestalGenerator.t0',
                        T='PedestalGenerator.telescopes',
                        p='PedestalGenerator.pixel',
                        o='PedestalGenerator.output_name',
                        dd='PedestalGenerator.debug'
                        ))
    classes = List([EventSourceFactory,
                    HESSIOEventSource,
                    TargetIOEventSource,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    CameraCalibrator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.cal = None
        self.run_list = None
        self.file_list = None
        self.baseline_bins = np.arange(0,7, 7/17.)
        self.baseline_start_rms = []
        self.baseline_start_mean =[]
        self.baseline_end_rms = []
        self.baseline_end_mean =[]
        self.waveform_rms = []
        self.waveform_mean =[]
        self.pulse_max  = []
        self.pulse_fwhm = []
        self.pltnsb = [0.01,0.05,0.100,0.200,0.300,0.500]

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.run_list = np.loadtxt('%s/../runlist.txt' % self.input_path, unpack=True)
        self.file_list = listdir('%s' % self.input_path)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(**kwargs)

        self.cal = CameraCalibrator(eventsource=EventSourceFactory.produce(input_url ="%s/%s" % (self.input_path, self.file_list[0]),max_events=1))

    def start(self):
        for n, run in enumerate(self.run_list[0]):
            self.baseline_start_rms.append([])
            self.baseline_start_mean.append([])
            self.baseline_end_rms.append([])
            self.baseline_end_mean.append([])
            self.waveform_rms.append([])
            self.waveform_mean.append([])
            self.pulse_max.append([])
            self.pulse_fwhm.append([])

            # if self.run_list[6][n] not in self.pltnsb:
            #     print('lets save some time!')
            #     continue
            #check
            if str(int(run)) not in self.file_list[n]:
                print(str(int(run)), self.file_list[n])
                print('check runlist.txt order, needs to be sorted?')
                self.file_list.sort()
                if str(int(run)) not in self.file_list[n]:
                    print('Sorting didn\'t seem to help, giving up.')
                    exit()
                else:
                    print('Sorted and sorted.')
            file_name = "%s/%s" % (self.input_path, self.file_list[n])
            print(file_name)




            try:
                source = EventSourceFactory.produce(input_url =file_name, max_events=self.max_events)

                for event in tqdm(source):
                    self.cal.calibrate(event)
                    self.dl0.reduce(event)
                    self.dl1.calibrate(event)

                    teldata = event.r1.tel[self.telescopes].waveform[0]

                    if self.pixel is None:

                        self.baseline_start_mean[n].append(np.mean(np.mean(teldata[:, 0:20], axis=1)))
                        self.baseline_start_rms[n].append(np.mean(np.std(teldata[:, 0:20], axis=1)))

                        self.baseline_end_mean[n].append(np.mean(np.mean(teldata[:, -20:], axis=1)))
                        self.baseline_end_rms[n].append(np.mean(np.std(teldata[:, -20:], axis=1)))

                        self.waveform_mean[n].append(np.mean(np.mean(teldata, axis=1)))
                        self.waveform_rms[n].append(np.mean(np.std(teldata, axis=1)))
                       
                        pls_max = []
                        pls_fwhm  = []

                        nm=len(teldata[0])
                        nmrange = np.arange(nm)
                        mean = np.mean(np.multiply(teldata, np.arange(nm)), axis=1)
                        #sigma = np.mean(teldata*(range(nm) - mean)**2, axis=1)
                        

                        for i in range(0,2048):
                            popt, pcov = curve_fit(gaus, nmrange, teldata[i], p0=[1, mean[i], 3])
                            spline = UnivariateSpline(nmrange, gaus(nmrange, *popt) -np.max(gaus(nmrange, *popt))/2, s=0)
                            #print(nm, mean, sigma, popt)
                            #print(spline)
                            #exit()
                            try:
                                r1,	r2 = spline.roots()
                            except ValueError:
                                r1=0
                                r2=0
                            fwhm = r2-r1 
                            mx = max(gaus(range(nm), *popt))
                            pls_max.append(mx)
                            pls_fwhm.append(fwhm)
                        self.pulse_fwhm[n].append(np.mean(pls_fwhm))
                        self.pulse_max[n].append(np.mean(pls_max))

                    else:

                        self.baseline_start_mean[n].append(np.mean(teldata[self.pixel, 0:20]))
                        self.baseline_start_rms[n].append(np.std(teldata[self.pixel, 0:20]))

                        self.baseline_end_mean[n].append(np.mean(teldata[self.pixel, -20:]))
                        self.baseline_end_rms[n].append(np.std(teldata[self.pixel, -20:]))

                        self.waveform_mean[n].append(np.mean(teldata[self.pixel, :]))
                        self.waveform_rms[n].append(np.std(teldata[self.pixel, :]))


                        n=len(teldata[0])
       	       	       	mean = sum(range(len(teldata[0])) * (teldata[self.pixel]))/n
       	       	        sigma = sum((teldata[self.pixel])* (range(len(teldata[0]))-mean)**2 )/n
                        popt, pcov = curve_fit(gaus, range(len(teldata[0])), teldata[self.pixel], p0=[1, mean, sigma])
       	       	       	spline = UnivariateSpline(range(len(teldata[0])), gaus(range(len(teldata[0])), *popt) -np.max(gaus(range(len(teldata[0])), *popt))/2, s=0)
       	       	       	r1, r2 = spline.roots()
       	       	       	fwhm = r2-r1 
       	       	       	mx = max(gaus(range(len(teldata[0])), *popt))
                        self.pulse_max[n].append(mx)
                        self.pulse_fwhm[n].append(fwhm)


            except FileNotFoundError:
                stop=0
                print('file_not_found')
    def finish(self):
        if self.debug:

            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(111)
            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            fig3 = plt.figure(3)
            ax3 = fig3.add_subplot(111)
            fig4 = plt.figure(4)
            ax4 = fig4.add_subplot(111)
            fig5 = plt.figure(5)
            ax5 = fig5.add_subplot(111)
            fig6 = plt.figure(6)
            ax6 = fig6.add_subplot(111)
            fig7 = plt.figure(7)
            ax7 = fig7.add_subplot(111)
            fig8 = plt.figure(8)
            ax8 = fig8.add_subplot(111)

            for n in range(len(self.baseline_start_rms)):
                if len(self.baseline_start_mean[n])>0:
                    ax1.hist(self.baseline_start_mean[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000 * self.run_list[6][n]))
                    ax2.hist(self.baseline_start_rms[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000 * self.run_list[6][n]))
                    ax3.hist(self.baseline_end_mean[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000 * self.run_list[6][n]))
                    ax4.hist(self.baseline_end_rms[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000 * self.run_list[6][n]))
                    ax5.hist(self.waveform_mean[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000*self.run_list[6][n]))
                    ax6.hist(self.waveform_rms[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000*self.run_list[6][n]))
                    ax7.hist(self.pulse_max[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000*self.run_list[6][n]))
                    ax8.hist(self.pulse_fwhm[n], bins=50, alpha=0.9, histtype='stepfilled',
                             label='%s MHz' % str(1000*self.run_list[6][n]))

            ax1.set_title('baseline_start_mean')
            ax2.set_title('baseline_start_rms')
            ax3.set_title('baseline_end_mean')
            ax4.set_title('baseline_end_rms')
            ax5.set_title('waveform_mean')
            ax6.set_title('waveform_rms')
            ax7.set_title('pulse_max')
            ax8.set_title('pulse_fwhm')


            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
            ax5.legend()
            ax6.legend()
            ax7.legend()
            ax8.legend()
            plt.show()

        if not path.isdir(self.output_name):
            makedirs(self.output_name)

        columns_str = ['baselineStartMean', 'baselineStartRMS', 'baselineEndMean',
                       'baselineEndRMS','baselineWaveformMean', 'baselineWaveformRMS', 'pulseMAX', 'pulseFWHM']
        for n in range(len(self.baseline_start_rms)):
            if len(self.baseline_start_mean[n]) > 0:
                print(self.baseline_start_mean[n])
                out_array = np.array([self.baseline_start_mean[n], self.baseline_start_rms[n],
                                            self.baseline_end_mean[n], self.baseline_end_rms[n],
                                            self.waveform_mean[n], self.waveform_rms[n], self.pulse_max[n], self.pulse_fwhm[n]])
                print(out_array, columns_str)
                data = pd.DataFrame(out_array.T, columns=columns_str)
                data.to_hdf('%s/extracted_pedestals_%sMHz.h5' % (self.output_name,str(1000*self.run_list[6][n])), 'table', append=True)
                # columns_str.append('%sMHz_baselineStartMean' % str(1000*self.run_list[6][n]))
                # columns_str.append('%sMHz_baselineStartRMS' % str(1000*self.run_list[6][n]))
                # columns_str.append('%sMHz_baselineEndMean' % str(1000*self.run_list[6][n]))
                # columns_str.append('%sMHz_baselineEndRMS' % str(1000*self.run_list[6][n]))
                # columns_str.append('%sMHz_baselineWaveformMean' % str(1000*self.run_list[6][n]))
                # columns_str.append('%sMHz_baselineWaveformRMS' % str(1000*self.run_list[6][n]))

        # out_array = np.concatenate((self.baseline_start_mean,self.baseline_start_rms,
        #                             self.baseline_end_mean, self.baseline_end_rms,
        #                             self.waveform_mean, self.waveform_rms), axis=0)


        # data = pd.DataFrame(out_array.T, columns=columns_str)
        # data.columns = data.columns.str.split('_', expand=True)
        # print(data)
        # data.to_hdf(self.output_name, 'table', append=True)
        print('Done!')

def main():
    exe = PedestalGenerator()
    exe.run()

if __name__ == '__main__':
    main()
