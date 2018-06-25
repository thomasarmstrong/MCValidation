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
from ctapipe.core import Tool
from traitlets import Dict, List, Int, Unicode, Bool
from ctapipe.image.charge_extractors import ChargeExtractorFactory
from glob import glob

debug = True

###### ToDo Define inputs via some sort of argparser, for now hardcode

class TriggerEffiencyGenerator(Tool):
    name = "TriggerEffiencyGenerator"
    description = "Generate the a pickle file of TriggerEffiency for " \
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

    aliases = Dict(dict(input_path='TriggerEffiencyGenerator.input_path',
                        calibrator='TriggerEffiencyGenerator.calibrator',
                        max_events='TriggerEffiencyGenerator.max_events',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        max_pe='TriggerEffiencyGenerator.max_pe',
                        T='TriggerEffiencyGenerator.telescopes',
                        o='TriggerEffiencyGenerator.output_name',
                        plot_cam='TriggerEffiencyGenerator.plot_cam',
                        use_true_pe='TriggerEffiencyGenerator.use_true_pe'
                        ))
    classes = List([EventSourceFactory,
                    HESSIOEventSource,
                    TargetIOEventSource,
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

        self.cal = CameraCalibrator(r1_product=self.calibrator)



    def start(self):
        run_list = np.loadtxt('%s/runlist.txt' % self.input_path, unpack=True)
        plot_cam = False
        plot_delay = 0.5
        disp = None
        n_events=[]
        trig_eff=[]

        n_pe = []
        if debug:
            fig=plt.figure(1)
            ax=fig.add_subplot(111)
        for n, run in enumerate(run_list[0]):
            n_events.append(run_list[4][n])
            n_pe.append(run_list[3][n])
            # TODO remove need for hardcoded file name
            file_name = None
            if self.calibrator == "TargetIOR1Calibrator":
                file_name = "%s/Run%05d_r1.tio" % (self.input_path, int(run))
                print(file_name)
            elif self.calibrator == "HESSIOR1Calibrator":
                file_name = "%s/sim_tel/run%05d.simtel.gz" % (self.input_path, int(run))
                print(file_name)
            n_trig = 0
            try:
                print('trying to open file')
                source = EventSourceFactory.produce(input_url=file_name, max_events=self.max_events)
                true_pe = []
                # lab_pe = []
                peds_all = []

                for event in tqdm(source):
                    n_trig = n_trig + 1

            except FileNotFoundError:
                stop=0
                print('file_not_found')
            print(n_trig, run_list[4][n], n_trig/run_list[4][n] )
            trig_eff.append(n_trig/run_list[4][n])
            # exit()
        plt.plot(n_pe, trig_eff )
        plt.show()
        if debug:
            plt.xscale('log')
            plt.yscale('log')
            plt.plot([0,1000],[0,1000], 'k:')
            plt.xlabel('Input p.e.')
            plt.ylabel('True mc p.e.')
            plt.show()
    def finish(self):
        #out_file = '%s/charge_resolution_test.h5' % self.input_path
        #self.calculator.save(self.output_name)
        print('done')

def main():
    exe = TriggerEffiencyGenerator()
    exe.run()

if __name__ == '__main__':
    main()
