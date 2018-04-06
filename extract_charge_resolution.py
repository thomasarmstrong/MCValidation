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
                        plot_cam='ChargeResolutionGenerator.plot_cam'
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

        for n, run in enumerate(run_list[0]):
            # TODO remove need for hardcoded file name
            if self.calibrator == "TargetIOR1Calibrator":
                file_name = "%s/Run%s_r1.tio" % (self.input_path, int(run))
                print(file_name)
            elif self.calibrator == "HESSIOR1Calibrator":
                file_name = "%s/Run%s_mc.simtel.gz" % (self.input_path, int(run))
                print(file_name)

            try:
                source = EventSourceFactory.produce(input_url =file_name, max_events=self.max_events)
                for event in tqdm(source):
                    self.cal.calibrate(event)
                    self.dl0.reduce(event)
                    self.dl1.calibrate(event)
                    input_pe = run_list[2][n]

                    if self.plot_cam == True:
                        if disp is None:
                            geom = event.inst.subarray.tel[self.telescopes].camera
                            disp = CameraDisplay(geom)
                            disp.add_colorbar()
                            plt.show(block=False)
                        im = event.dl1.tel[self.telescopes].image[0]
                        disp.image = im
                        plt.pause(plot_delay)

                    true_charge_mc = event.mc.tel[self.telescopes].photo_electron_image
                    measured_charge = event.dl1.tel[self.telescopes].image[0]
                    true_charge_lab = np.asarray([input_pe]*len(measured_charge))

                    self.calculator.add_charges(true_charge_lab.astype(int), measured_charge)
            except FileNotFoundError:
                stop=0
                print('file_not_found')

    def finish(self):
        out_file = '%s/charge_resolution_test.h5' % self.input_path
        self.calculator.save(self.output_name)

def main():
    exe = ChargeResolutionGenerator()
    exe.run()

if __name__ == '__main__':
    main()
