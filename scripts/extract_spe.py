from ctapipe.calib import CameraCalibrator
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.r1 import CameraR1Calibrator
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from tqdm import tqdm
from ctapipe.io import event_source
from ctapipe.image.charge_extractors import GlobalPeakIntegrator, NeighbourPeakIntegrator
from matplotlib import cm
from ctapipe.image.charge_extractors import ChargeExtractorFactory
import numpy as np
from scipy import interpolate
from scipy.ndimage import correlate1d
from inspect import isabstract
from ctapipe.image.charge_extractors import Integrator
from traitlets import Dict, List, Int, Unicode, Bool
from ctapipe.core import Tool

from ctapipe.io.hessioeventsource import HESSIOEventSource
from ctapipe.io.targetioeventsource import TargetIOEventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
from ctapipe.core import Tool
from glob import glob
from ctapipe.calib.camera.r1 import HESSIOR1Calibrator

from ctapipe.image.charge_extractors import GlobalPeakIntegrator, LocalPeakIntegrator, NeighbourPeakIntegrator, AverageWfPeakIntegrator, ChargeExtractorFactory


class CrossCorrelation():
    """
    Extractor which uses the result of the cross correlation of the waveforms
    with a reference pulse. The cross correlation results acts as a sliding
    integration window that is weighted according to the pulse shape. The
    maximum of the cross correlation result is the point at which the
    reference pulse best matches the waveform. To choose an unbiased
    extraction time I average the cross correlation result across all pixels
    and take the maximum as the peak time.
    """

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        path =  "/Users/armstrongt/Software/CTA/CHECsoft/lab/data/checs_reference_pulse_lei.txt"
        file = np.loadtxt(path, delimiter=', ')
        refx = file[:, 0]
        refy = file[:, 1] - file[:, 1][0]
        f = interpolate.interp1d(refx, refy, kind=3)
        x = np.linspace(0, 77e-9, 76)
        y = f(x)

        # Put pulse in center so result peak time matches with input peak
        pad = y.size - 2 * np.argmax(y)
        if pad > 0:
            y = np.pad(y, (pad, 0), mode='constant')
        else:
            y = np.pad(y, (0, -pad), mode='constant')

        # Make maximum of cc result == 1
        y = y / correlate1d(y, y).max()

        self.reference_pulse = y
        self.cc = None

    def _apply_cc(self, waveforms):
        cc = correlate1d(waveforms, self.reference_pulse)
        return cc



class SPEGenerator(Tool):
    name = "SPEGenerator"
    description = "Generate the a pickle file of single photo electron spectrum for " \
                  "either MC or data files."

    telescopes = Int(1,help='Telescopes to include from the event file. '
                           'Default = 1').tag(config=True)
    output_name = Unicode('single_photoelectron_spectum',
                          help='Name of the output SPE hdf5 file '
                               'file').tag(config=True)
    input_path = Unicode(help='Path to input file containing data').tag(config=True)

    max_events = Int(1000, help='Maximum number of events to use').tag(config=True)

    pixel = Int(1000, allow_none=True, help='Pixel to generate SPE data for').tag(config=True)

    aliases = Dict(dict(input_path='SPEGenerator.input_path',
                        output_name='SPEGenerator.output_name',
                        max_events='SPEGenerator.max_events',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        T='SPEGenerator.telescopes',
                        p='SPEGenerator.pixel'
                        ))
    classes = List([EventSourceFactory,
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
        self.recope12 = np.array([])
        self.recope22 = np.array([])
        self.recope32 = np.array([])
        self.recope42 = np.array([])
        self.recope52 = np.array([])

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.r1 = HESSIOR1Calibrator(**kwargs)

        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(**kwargs)

        self.cal = CameraCalibrator()

        self.cross = CrossCorrelation()

        self.glob_peak = GlobalPeakIntegrator()

        self.local_peak = LocalPeakIntegrator()

        self.neighbour = NeighbourPeakIntegrator()

        self.aver = AverageWfPeakIntegrator()

    def start(self):

        file_name = "%s" % (self.input_path)
        print('opening ', file_name)

        try:
            print('trying to open file')
            source = EventSourceFactory.produce(input_url=file_name, max_events=self.max_events)
            for event in tqdm(source):
                self.r1.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                # measured_charge = event.dl1.tel[self.telescopes].image[0][self.pixel]
                # print(self.glob_peak.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0])
                # print(self.glob_peak.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0].shape)
                # exit()
                # self.recope12 = np.append(self.recope12, max(self.cross._apply_cc(event.r1.tel[self.telescopes].waveform[0][self.pixel])))
                # self.recope22 = np.append(self.recope22, self.glob_peak.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0][self.pixel])
                self.recope32 = np.append(self.recope32, self.local_peak.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0][self.pixel])
                # self.recope42 = np.append(self.recope32, self.aver.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0][self.pixel])
                # self.recope52 = np.append(self.recope52, self.neighbour.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0][self.pixel])


        except FileNotFoundError:
            print('file_not_found')
            exit()


    def finish(self):
        bins2 = np.arange(-0.5, 4, 0.055)
        bins = np.arange(-0.5, 4, 0.055)
        center = (bins[:-1] + bins[1:]) / 2
        fig0 = plt.figure(0)
        # ax1 = fig0.add_subplot(141)
        # ax2 = fig0.add_subplot(142)
        ax3 = fig0.add_subplot(111)
        # ax4 = fig0.add_subplot(144)
        # ax1.hist(self.recope12, bins=bins2, color='C0', alpha =0.6, label='cross correlation')
        # ax1.set_title('cross correlation')
        # ax2.hist(self.recope22, bins=bins, color='C1', alpha =0.6, label='global peak')
        # ax2.set_title('global peak')
        histout = ax3.hist(self.recope32, bins=bins, color='C2', alpha =0.6, label='local peak')
        ax3.set_title('local peak')
        out_file = open(self.output_name, 'w')
        for i in range(len(center)):
            out_file.write('%s\t%s\n' % (center[i], histout[0][i]))
        out_file.close()
        # ax4.hist(self.recope42, bins=bins, color='C3', alpha =0.6, label='average waveform')
        # ax4.set_title('average waveform')
        plt.show()
        # out_file = open(self.output_name, 'w')
        # for n,i in enumerate(self.trig_eff_array):
        #     out_file.write('%s\t%s\n' % (self.disc_array[n], i))
        # out_file.close()
        print('done')

def main():
    exe = SPEGenerator()
    exe.run()

if __name__ == '__main__':
    main()
