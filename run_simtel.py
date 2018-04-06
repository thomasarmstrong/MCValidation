import os

def make_mask(on_pixels):
    cam_file = open('/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/sim_telarray/cfg/CTA/camera_CHEC-S_GATE.dat', 'r')
    out_file = open('/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/sim_telarray/cfg/CTA/camera_CHEC-S_GATE_masked.dat', 'w')
    for line in cam_file:
        if line.startswith('Pixel'):
            data = line.strip().split(' ')
            if int(data[0].split('\t')[1]) in on_pixels:
                out_file.write(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4] + ' ' + '1\n')
            else:
                out_file.write(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4] + ' ' + '0\n')
        else:
            out_file.write(line)

def run_simtel(outfile ='../data/bypass2_enoise.simtel.gz', nsb = 0.02, infile = '/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_test.dat.gz'):
	os.system('/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/sim_telarray/bin/sim_telarray '
			  '-c /scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/sim_telarray/cfg/CTA/CTA-ULTRA6-SST-GCT-S.cfg '
			  '-o %s '
			  '-C BYPASS_OPTICS=2 '
			  '-C NIGHTSKY_BACKGROUND=all:%s '
              '-C discriminator_threshold=230 '
              '-C trigger_pixels=2 '
              '-C trigger_telescopes=1 '
              '-C MIN_PHOTONS=0 '
              '-C MIN_PHOTOELECTRONS=0 '
			  '-I/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/sim_telarray/cfg/CTA/ '
			  '-I/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/corsika-6990/run '
			  '%s' % (outfile, nsb, infile))

def run_lightemission(events = 3, photons = 10946249, distance = 100, cam_radius = 30, ang_dist = '/scratch/armstrongt/Workspace/CTA/MCValidation/data/ang_dist_2.dat', out_file = '/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_test.dat.gz'):
    os.system('/scratch/armstrongt/Workspace/CTA/MCValidation/src/LightEmission-pkg/ff-1m '
              '--events %s '
              '--photons %s ' 
              '--distance %s '
              '--camera-radius %s '
              '--angular-distribution %s '
              '-o %s' % (events, photons, distance, cam_radius, ang_dist, out_file))
def main():
    pe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 25, 29, 33, 39, 44, 51, 59, 68, 79, 91, 104, 120, 138,
          159, 184, 212, 244, 281, 323, 372, 429, 494, 568, 655, 754, 868, 1000]
    photons = [10025.760117977465, 20051.520235954929, 30077.280353932398, 40103.040471909859, 50128.800589887331,
               60154.560707864795, 70180.320825842267, 80206.080943819718, 90231.841061797197, 100257.60117977466,
               120309.12141572959, 140360.64165168453, 160412.16188763944, 190489.44224157184, 220566.72259550422,
               250644.00294943663, 290747.04342134646, 330850.08389325638, 391004.64460112114, 441133.44519100845,
               511313.76601685071, 591519.84696067055, 681751.68802246766, 792035.04932021978, 912344.1707359493,
               1042679.0522696566, 1203091.2141572959, 1383554.8962808903, 1594095.858758417, 1844739.8617078539,
               2125461.1450112229, 2446285.4687865013, 2817238.5931516676, 3238320.5181067213, 3729582.7638876173,
               4301051.0906123333, 4952725.4982808679, 5694631.7470112005, 6566872.8772752397, 7559423.1289550085,
               8702359.7824044414, 10025760.117977465]
    for p in range(len(pe)):
        infl = '/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_events1000_pe%s_dist100_camr30_angdistfullflat.dat.gz' % pe[p]
        outfl = '/scratch/armstrongt/Workspace/CTA/MCValidation/data/bypass2_enoise_pe%s_masked.simtel.gz' % pe[p]
        print("@@@@ Running LightEmission Package\n\n")
        run_lightemission(events = 1000, photons = photons[p], out_file = infl)
        # print("@@@@ Running Simtelarray\n\n")
        # run_simtel(infile = infl,outfile = outfl)

if __name__ == '__main__':
    main()