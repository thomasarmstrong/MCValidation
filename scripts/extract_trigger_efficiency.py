from ctapipe.calib import CameraCalibrator
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.calib.camera.dl0 import CameraDL0Reducer
from ctapipe.calib.camera.dl1 import CameraDL1Calibrator
from tqdm import tqdm
from ctapipe.io.eventsourcefactory import EventSourceFactory
from ctapipe.core import Tool
from traitlets import Dict, List, Int, Unicode, Bool
from os import listdir


debug = True

###### ToDo Define inputs via some sort of argparser, for now hardcode

class TriggerEffiencyGenerator(Tool):
    name = "TriggerEffiencyGenerator"
    description = "Generate the a pickle file of TriggerEffiency for " \
                  "either MC or data files."

    telescopes = Int(1,help='Telescopes to include from the event file. '
                           'Default = 1').tag(config=True)
    output_name = Unicode('trigger_efficiency',
                          help='Name of the output trigger efficiency hdf5 '
                               'file').tag(config=True)
    input_path = Unicode(help='Path to directory containing data').tag(config=True)

    max_events = Int(1000, help='Maximum number of events to use').tag(config=True)

    plot_cam = Bool(False, "enable plotting of individual camera").tag(config=True)

    use_true_pe = Bool(False, "Use true mc p.e.").tag(config=True)


    aliases = Dict(dict(input_path='TriggerEffiencyGenerator.input_path',
                        output_name='TriggerEffiencyGenerator.output_name',
                        max_events='TriggerEffiencyGenerator.max_events',
                        clip_amplitude='CameraDL1Calibrator.clip_amplitude',
                        radius='CameraDL1Calibrator.radius',
                        max_pe='TriggerEffiencyGenerator.max_pe',
                        T='TriggerEffiencyGenerator.telescopes',
                        plot_cam='TriggerEffiencyGenerator.plot_cam',
                        use_true_pe='TriggerEffiencyGenerator.use_true_pe'
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
        self.trig_eff_array = []
        self.disc_array = []
    def setup(self):
        kwargs = dict(config=self.config, tool=self)
        self.dl0 = CameraDL0Reducer(**kwargs)

        self.dl1 = CameraDL1Calibrator(**kwargs)

        self.cal = CameraCalibrator()



    def start(self):
        run_list = np.loadtxt('%s/../runlist.txt' % self.input_path, unpack=True)
        file_list = listdir('%s' % self.input_path)
        file_list.sort()
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
            n_events.append(run_list[5][n])
            n_pe.append(run_list[3][n])

            if str(int(run)) not in file_list[n]:
                print(str(int(run)), file_list[n])
                print('check runlist.txt order, needs to be sorted?')
                exit()
            file_name = "%s/%s" % (self.input_path, file_list[n])
            print(file_name)

            n_trig = 0
            try:
                print('trying to open file')
                source = EventSourceFactory.produce(input_url=file_name, max_events=self.max_events)
                for event in tqdm(source):
                    n_trig = n_trig + 1

            except FileNotFoundError:
                print('file_not_found')
            print(n_trig, run_list[5][n], n_trig/run_list[5][n] )
            trig_eff.append(n_trig/run_list[5][n])
            self.trig_eff_array.append(n_trig/run_list[5][n])
            self.disc_array.append(run_list[7][n])
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
        out_file = open(self.output_name, 'w')
        for n,i in enumerate(self.trig_eff_array):
            out_file.write('%s\t%s\n' % (self.disc_array[n], i))
        out_file.close()
        print('done')

def main():
    exe = TriggerEffiencyGenerator()
    exe.run()

if __name__ == '__main__':
    main()
