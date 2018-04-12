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

debug = True

###### ToDo Define inputs via some sort of argparser, for now hardcode

class ChargeResolutionGenerator(Tool):
    name = "ChargeResolutionGenerator"
    description = "Generate the a pickle file of ChargeResolutionFile for " \
                  "either MC or data files."

    telescopes = Int(1,help='Telescopes to include from the event file. '
                           'Default = 1').tag(config=True)
    output_name = Unicode('charge_resolution',
                          help='Name of the output charge resolution hdf5 '
                               'file').tag(config=True)
    input_path = Unicode(help='Path to directory containing data').tag(config=True)

    max_events = Int(1, help='Maximum number of events to use').tag(config=True)

    plot_cam = Bool(False, "enable plotting of individual camera").tag(config=True)

    use_true_pe = Bool(False, "Use true mc p.e.").tag(config=True)

    calibrator = Unicode('HESSIOR1Calibrator', help='which calibrator to use, default = HESSIOR1Calibrator').tag(config=True)

    aliases = Dict(dict(input_path='ChargeResolutionGenerator.input_path',
                        calibrator='ChargeResolutionGenerator.calibrator',
                        max_events='ChargeResolutionGenerator.max_events',
                        extractor='ChargeExtractorFactory.product',
                        window_width='ChargeExtractorFactory.window_width',
                        t0='ChargeExtractorFactory.t0',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        max_pe='ChargeResolutionCalculator.max_pe',
                        T='ChargeResolutionGenerator.telescopes',
                        o='ChargeResolutionGenerator.output_name',
                        plot_cam='ChargeResolutionGenerator.plot_cam',
                        use_true_pe='ChargeResolutionGenerator.use_true_pe'
                        ))
    classes = List([EventSourceFactory,
                    HESSIOEventSource,
                    TargetIOEventSource,
                    ChargeExtractorFactory,
                    CameraDL1Calibrator,
                    ChargeResolutionCalculator,
                    CameraCalibrator
                    ])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eventsource = None
        self.r1 = None
        self.dl0 = None
        self.dl1 = None
        self.calculator = None
        self.cal = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(**kwargs)

        self.cal = CameraCalibrator(r1_product=self.calibrator)

        self.calculator = ChargeResolutionCalculator(**kwargs)


    def start(self):
        run_list = np.loadtxt('%s/runlist.txt' % self.input_path, unpack=True)
        plot_cam = False
        plot_delay = 0.5
        disp = None

        if debug:
            fig=plt.figure(1)
            ax=fig.add_subplot(111)
        for n, run in enumerate(run_list[0]):
            # TODO remove need for hardcoded file name
            if self.calibrator == "TargetIOR1Calibrator":
                file_name = "%s/Run%05d_r1.tio" % (self.input_path, int(run))
                print(file_name)
            elif self.calibrator == "HESSIOR1Calibrator":
                file_name = "%s/Run%05d_mc.simtel.gz" % (self.input_path, int(run))
                print(file_name)

            try:
                source = EventSourceFactory.produce(input_url =file_name, max_events=self.max_events)
                true_pe = []
                # lab_pe = []
                peds_all = []
                for event in tqdm(source):
                    self.cal.calibrate(event)
                    self.dl0.reduce(event)
                    self.dl1.calibrate(event)
                    input_pe = run_list[2][n]
                    try:
                        input_nsb = run_list[5][n]
                    except IndexError:
                        print('File has no column for NSB, setting to 0')
                        input_nsb = 0
                    if self.plot_cam == True:
                        if disp is None:
                            geom = event.inst.subarray.tel[self.telescopes].camera
                            disp = CameraDisplay(geom)
                            disp.add_colorbar()
                            plt.show(block=False)
                        im = event.dl1.tel[self.telescopes].image[0]
                        disp.image = im
                        plt.pause(plot_delay)

                    teldata = event.r0.tel[self.telescopes].waveform[0]
                    peds = teldata[:, 0:10].mean(axis=1)
                    peds2 = teldata[:, 0:10].std(axis=1)
                    peds_all.append(teldata[:, 0:90])
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

                if debug:
                    # plt.errorbar(input_nsb, np.mean(peds_all), np.std(peds_all),color='k')
                    plt.scatter(input_nsb, np.std(peds_all), marker ='x',color='k')
            except FileNotFoundError:
                stop=0
                print('file_not_found')
        plt.show()
        if debug:
            plt.xscale('log')
            plt.yscale('log')
            plt.plot([0,1000],[0,1000], 'k:')
            plt.xlabel('Input p.e.')
            plt.ylabel('True mc p.e.')
            plt.show()
    def finish(self):
        out_file = '%s/charge_resolution_test.h5' % self.input_path
        self.calculator.save(self.output_name)

def main():
    exe = ChargeResolutionGenerator()
    exe.run()

if __name__ == '__main__':
    main()
