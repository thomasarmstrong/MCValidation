from ctapipe.calib.camera.r1 import (
    CameraR1CalibratorFactory,
    HESSIOR1Calibrator,
    TargetIOR1Calibrator,
    NullR1Calibrator)
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
from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
from ctapipe.core import Tool
from traitlets import Dict, List, Int, Unicode, Bool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from glob import glob
from os import listdir

debug = True

###### ToDo Define inputs via some sort of argparser, for now hardcode

class PedestalGenerator(Tool):
    name = "PedestalGenerator"
    description = "Generate the a pickle file of Pedestals for " \
                  "either MC or data files."

    telescopes = Int(1,help='Telescopes to include from the event file. '
                           'Default = 1').tag(config=True)
    output_name = Unicode('extracted_pedestals',
                          help='Name of the output extracted pedestal hdf5 '
                               'file').tag(config=True)
    input_path = Unicode(help='Path to directory containing data').tag(config=True)

    max_events = Int(1, help='Maximum number of events to use').tag(config=True)

    plot_cam = Bool(False, "enable plotting of individual camera").tag(config=True)

    t0 = Int(0, help='Timeslice to start pedestal').tag(config=True)

    window_width = Int(10, help='length of window within which to determine pedestal').tag(config=True)

    aliases = Dict(dict(input_path='PedestalGenerator.input_path',
                        max_events='PedestalGenerator.max_events',
                        window_width='PedestalGenerator.window_width',
                        t0='PedestalGenerator.t0',
                        T='PedestalGenerator.telescopes',
                        o='PedestalGenerator.output_name',
                        plot_cam='PedestalGenerator.plot_cam'
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

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(**kwargs)

        self.cal = CameraCalibrator()


    def start(self):
        run_list = np.loadtxt('%s/../runlist.txt' % self.input_path, unpack=True)
        file_list = listdir('%s' % self.input_path)
        plot_cam = False
        plot_delay = 0.5
        disp = None

        pltnsb = [0.01, 0.05,0.100,0.200,0.300,0.500]
        if debug:
            fig=plt.figure(1)
            ax=fig.add_subplot(111)
            fig2 = plt.figure(2)

        for n, run in enumerate(run_list[0]):
            if debug:
                ax2 = fig2.add_subplot(7,8,n+1)
            #check
            if str(int(run)) not in file_list[n]:
                print(str(int(run)), file_list[n])
                print('check runlist.txt order, needs to be sorted?')
                exit()
            file_name = "%s/%s" % (self.input_path, file_list[n])
            print(file_name)


            try:
                source = EventSourceFactory.produce(input_url =file_name, max_events=self.max_events)
                true_pe = []
                # lab_pe = []
                peds_sdev = []
                peds_all = []
                peds_mean = []
                maxpeak=[]

                baseline_start_mean = []
                baseline_start_rms = []

                baseline_end_mean = []
                baseline_end_rms = []

                waveform_mean = []
                waveform_rms = []

                for event in tqdm(source):
                    self.cal.calibrate(event)
                    self.dl0.reduce(event)
                    self.dl1.calibrate(event)
                    input_pe = run_list[3][n]
                    input_nsb = run_list[6][n]

                    if self.plot_cam == True:
                        if disp is None:
                            geom = event.inst.subarray.tel[self.telescopes].camera
                            disp = CameraDisplay(geom)
                            disp.add_colorbar()
                            plt.show(block=False)
                        im = event.dl1.tel[self.telescopes].image[0]
                        disp.image = im
                        plt.pause(plot_delay)
                    # print(event)
                    teldata = event.r1.tel[self.telescopes].waveform[0]
                    print(teldata[100].shape)
                    peds = teldata[:, self.t0:self.t0+self.window_width].mean(axis=1)
                    peds2 = teldata[:, self.t0:self.t0+self.window_width].std(axis=1)
                    peds_sdev.append(peds2)
                    peds_mean.append(peds)
                    peds_all.append(teldata[:, self.t0:self.t0+self.window_width])
                    for i in range(2048):
                        maxpeak.append(max(teldata[i]))
                    if debug:
                        ax2.plot(range(128), teldata[100])
                        ax2.fill_between([self.t0, self.t0+self.window_width], 0,60, color='r', alpha=0.3)

                    baseline_start_mean.append(np.mean(teldata[:, 0:20], axis=1))
                    baseline_start_rms.append(np.std(teldata[:, 0:20], axis=1))

                    baseline_end_mean.append(np.mean(teldata[:, -20:], axis=1))
                    baseline_end_rms.append(np.std(teldata[:, -20:], axis=1))

                    waveform_mean.append(np.mean(teldata, axis=1))
                    waveform_rms.append(np.std(teldata, axis=1))


                    # plt.hist(peds,bins=50, alpha=0.4)
                    # plt.show()
                    # print(teldata)
                    # plt.plot(range(len(teldata[100])), teldata[100])
                    # plt.show()
                    # exit()
                # print(np.mean(peds_all), np.std(peds_all))
                # exit()
                    # true_charge_mc = event.mc.tel[self.telescopes].photo_electron_image
                    # measured_charge = event.dl1.tel[self.telescopes].image[0]
                    # true_charge_lab = np.asarray([input_pe]*len(measured_charge))
                    # true_pe.append(true_charge_mc)
                    # if self.use_true_pe:
                    #     true_charge=true_charge_mc
                    # else:
                    #     true_charge=true_charge_lab.astype(int)
                    #
                    # self.calculator.add_charges(true_charge, measured_charge)

                if debug and run_list[6][n] in pltnsb:
                    ax.hist(peds_sdev, bins=50, alpha=0.4)
                    # plt.errorbar(input_nsb, np.mean(peds_all), np.std(peds_all), marker ='x',color='k')
                    # plt.scatter(input_nsb, np.std(peds_all), marker ='x',color='k')
                    ax.set_xlabel('Non pulsed background light [GHz]')
                    ax.set_ylabel('Pedistal mean')
                    fig3 = plt.figure(3)
                    plt.hist(maxpeak, bins=50, alpha=0.4)
                    plt.title('maxpeak')
                    fig4 = plt.figure(4)
                    plt.hist(peds_mean, bins=50, alpha=0.4)
                    plt.title('peds_mean')
                    fig5 = plt.figure(5)
                    plt.hist(baseline_start_mean, bins=50, alpha=0.4)
                    plt.title('baseline_start_mean')
                    fig6 = plt.figure(6)
                    plt.hist(baseline_start_rms, bins=50, alpha=0.4)
                    plt.title('baseline_start_rms')
                    fig7 = plt.figure(7)
                    plt.hist(baseline_end_mean, bins=50, alpha=0.4)
                    plt.title('baseline_end_mean')
                    fig8 = plt.figure(8)
                    plt.hist(baseline_end_rms, bins=50, alpha=0.4)
                    plt.title('baseline_end_rms')
                    fig9 = plt.figure(9)
                    plt.hist(waveform_mean, bins=50, alpha=0.4)
                    plt.title('waveform_mean')
                    fig10 = plt.figure(10)
                    plt.hist(waveform_rms, bins=50, alpha=0.4)
                    plt.title('waveform_rms')
            except FileNotFoundError:
                stop=0
                print('file_not_found')
        plt.show()
        # if debug:
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.plot([0,1000],[0,1000], 'k:')
        #     plt.xlabel('Input p.e.')
        #     plt.ylabel('True mc p.e.')
        #     plt.show()
    def finish(self):
        # out_file = '%s/charge_resolution_test.h5' % self.input_path
        # self.calculator.save(self.output_name)
        print('Done!')

def main():
    exe = PedestalGenerator()
    exe.run()

if __name__ == '__main__':
    main()
