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
from scipy import interpolate
from scipy.ndimage import correlate1d
from ctapipe.image.charge_extractors import GlobalPeakIntegrator, LocalPeakIntegrator, NeighbourPeakIntegrator, AverageWfPeakIntegrator, ChargeExtractorFactory
from CHECLabPy.core.base_reducer import WaveformReducer


def load_reference_pulse(path):
    file = np.loadtxt(path, delimiter=',')
    print("Loaded reference pulse: {}".format(path))
    time_slice = 1E-9
    refx = file[:, 0]
    refy = file[:, 1]
    f = interpolate.interp1d(refx, refy, kind=3)
    max_sample = int(refx[-1] / time_slice)
    x = np.linspace(0, max_sample * time_slice, max_sample + 1)
    y = f(x)

    # Put pulse in center so result peak time matches with input peak
    pad = y.size - 2 * np.argmax(y)
    if pad > 0:
        y = np.pad(y, (pad, 0), mode='constant')
    else:
        y = np.pad(y, (0, -pad), mode='constant')

    # Create 1p.e. pulse shape
    y_1pe = y / np.trapz(y)

    # Make maximum of cc result == 1
    y = y / correlate1d(y_1pe, y).max()

    return y, y_1pe


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

        ref = load_reference_pulse(path)
        self.reference_pulse, self.y_1pe = ref
        self.cc = None
        self.window_shift = 4
        self.window_size = 8
        self.t_event = None
        self.window_start = None
        self.window_end = None

    def get_pulse_height(self, charge):
        return charge * self.y_1pe.max()

    def get_charge(self, waveforms):
        self.cc = correlate1d(waveforms, self.reference_pulse)
        avg = np.mean(self.cc, axis=0)
        t_event = np.argmax(avg)
        n_samples = self.cc.shape[1]
        if t_event < 10:
            t_event = 10
        elif t_event > n_samples - 10:
            t_event = n_samples - 10
        self.t_event = t_event
        self.window_start = self.t_event - self.window_shift
        self.window_end = self.window_start + self.window_size
        charge = self.cc[:, self.t_event]
        cc_height = self.get_pulse_height(charge)

        params = dict(
            charge=charge,
            cc_height=cc_height,
        )
        return params


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
        self.aver = None
        self.cross = None
        self.glob_peak = None
        self.local_peak = None
        self.neighbour = None
        self.reconstructed_image_array =[]
        self.mean_reconstructed_image_array = None
        self.event_count=0
        self.geom = None
        self.disp = None

    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.dl0 = CameraDL0Reducer(**kwargs)
        self.dl1 = CameraDL1Calibrator(**kwargs)
        self.cal = CameraCalibrator(r1_product=self.calibrator)
        self.cross = CrossCorrelation()
        self.glob_peak = GlobalPeakIntegrator()
        self.local_peak = LocalPeakIntegrator()
        self.neighbour = NeighbourPeakIntegrator()
        self.aver = AverageWfPeakIntegrator()

    def start(self):
        try:
            source = EventSourceFactory.produce(input_url=self.infile, max_events=self.max_events)

            for event in tqdm(source):
                self.cal.calibrate(event)
                self.dl0.reduce(event)
                self.dl1.calibrate(event)

                if self.disp is None:
                    self.geom = event.inst.subarray.tel[self.telescopes].camera
                    self.mean_reconstructed_image_array = np.zeros(len(self.geom.pix_id))
                    if self.debug:
                        self.disp = CameraDisplay(self.geom)
                        self.disp.add_colorbar()

                # reco_array = self.cross.get_charge(event.r1.tel[self.telescopes].waveform[0])['charge']
                # reco_array = self.glob_peak.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0]
                reco_array =  self.local_peak.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0]
                # reco_array = self.aver.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0]
                # reco_array = self.neighbour.extract_charge(event.r1.tel[self.telescopes].waveform)[0][0]
                # print(reco_array)
                # exit()
                # print(event.mc.tel[self.telescopes].photo_electron_image)

                self.mean_reconstructed_image_array = np.add(self.mean_reconstructed_image_array,
                                                             # event.mc.tel[self.telescopes].photo_electron_image)
                                                             reco_array)
                self.reconstructed_image_array.append(event.dl1.tel[self.telescopes].image[0])
                self.event_count += 1
        except FileNotFoundError:
            print('file_not_found')

    def finish(self):
        # out_file = open(self.output_name)
        # out_file.write('#PixID meanIllum\n')
        # for i in range(len(self.reconstructed_image_array)):
        #     out_file.write('%s\t%s\n' % (self.geom.pix_id, self.reconstructed_image_array))
        # out_file.close()
        self.mean_reconstructed_image_array = self.mean_reconstructed_image_array/self.event_count
        if self.debug:
            # mean_image = np.mean(self.reconstructed_image_array,axis=0)/np.mean(np.mean(self.reconstructed_image_array))
            mean_image = self.mean_reconstructed_image_array
            self.disp.image = mean_image
            # self.disp.norm = 'log'

            plt.show()

def main():
    exe = FlatFieldGenerator()
    exe.run()


if __name__ == '__main__':
    main()


