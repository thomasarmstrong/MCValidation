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

class FlatFieldGenerator(Tool):
    name = "FlatFieldGenerator"
    description = "Generate the a pickle file of FlatField for " \
                  "either MC or data files."

    telescopes = Int(1,help='Telescopes to include from the event file. '
                           'Default = 1').tag(config=True)
    output_name = Unicode('extracted_flatfield',
                          help='Name of the output extracted flat field hdf5 '
                               'file').tag(config=True)
    infile = Unicode(help='Path to file containing data').tag(config=True)

    max_events = Int(1, help='Maximum number of events to use').tag(config=True)

    plot_cam = Bool(False, "enable plotting of individual camera").tag(config=True)

    use_true_pe = Bool(False, "Use true mc p.e.").tag(config=True)

    calibrator = Unicode('HESSIOR1Calibrator', help='which calibrator to use, default = HESSIOR1Calibrator').tag(
        config=True)
    debug = Bool(False, "plot resulting histograms").tag(config=True)

    aliases = Dict(dict(infile='FlatFieldGenerator.infile',
                        calibrator='FlatFieldGenerator.calibrator',
                        max_events='FlatFieldGenerator.max_events',
                        extractor='ChargeExtractorFactory.product',
                        window_width='ChargeExtractorFactory.window_width',
                        t0='ChargeExtractorFactory.t0',
                        window_shift='ChargeExtractorFactory.window_shift',
                        sig_amp_cut_HG='ChargeExtractorFactory.sig_amp_cut_HG',
                        sig_amp_cut_LG='ChargeExtractorFactory.sig_amp_cut_LG',
                        lwt='ChargeExtractorFactory.lwt',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        T='FlatFieldGenerator.telescopes',
                        o='FlatFieldGenerator.output_name',
                        plot_cam='FlatFieldGenerator.plot_cam',
                        use_true_pe='FlatFieldGenerator.use_true_pe',
                        dd='FlatFieldGenerator.debug'
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
        self.calculator = None
        self.cal = None
        self.reconstructed_image_array =[]
        self.geom = None
        self.disp = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(**kwargs)
        self.cal = CameraCalibrator(r1_product=self.calibrator)

    def start(self):
        try:
            source = EventSourceFactory.produce(input_url=self.infile, max_events=self.max_events)

            for event in tqdm(source):
                self.cal.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                if self.debug:
                    if self.disp is None:
                        self.geom = event.inst.subarray.tel[self.telescopes].camera
                        self.disp = CameraDisplay(self.geom)
                        self.disp.add_colorbar()

                self.reconstructed_image_array.append(event.dl1.tel[self.telescopes].image[0])

        except FileNotFoundError:
            print('file_not_found')

    def finish(self):
        out_file = open(self.output_name)
        out_file.write('#PixID meanIllum\n')
        for i in range(len(self.reconstructed_image_array)):
            out_file.write('%s\t%s\n' % (self.geom.pix_id, self.reconstructed_image_array))
        out_file.close()
        if self.debug:
            mean_image = np.mean(self.reconstructed_image_array,axis=0)/np.mean(np.mean(self.reconstructed_image_array))
            self.disp.image = mean_image
            plt.show()

def main():
    exe = FlatFieldGenerator()
    exe.run()


if __name__ == '__main__':
    main()