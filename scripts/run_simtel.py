import os
import argparse
import numpy as np
import multiprocessing

## Hardcoded parameters:
simtel_path = '/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/sim_telarray'
corsika_path = '/scratch/armstrongt/Software/CTA/CorsikaSimtel/2017-12-08_testing/corsika-6990/run'
lightEmission_path = '/scratch/armstrongt/Workspace/CTA/MCValidation/src/LightEmission-pkg'


def make_mask(on_pixels, cam_file, out_file):
    for line in cam_file:
        if line.startswith('Pixel'):
            data = line.strip().split(' ')
            if int(data[0].split('\t')[1]) in on_pixels:
                out_file.write(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4] + ' ' + '1\n')
            else:
                out_file.write(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + data[4] + ' ' + '0\n')
        else:
            out_file.write(line)


def run_simtel(outfile='../data/bypass2_enoise.simtel.gz', nsb=0.02, disc_thresh=230, extra_opts=' ',
               infile='/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_test.dat.gz'):
    os.system('%s/sim_telarray '
              '-c /%s/cfg/CTA/CTA-ULTRA6-SST-GCT-S.cfg '
              '-o %s '
              '-C BYPASS_OPTICS=2 '
              '-C NIGHTSKY_BACKGROUND=all:%s '
              '-C discriminator_threshold=%s '
              '-C trigger_pixels=2 '
              '-C trigger_telescopes=1 '
              '-C MIN_PHOTONS=0 '
              '-C MIN_PHOTOELECTRONS=0 '
              '%s '
              '-I%s/cfg/CTA/ '
              '-I%s '
              '%s' % (simtel_path, simtel_path, outfile, nsb, disc_thresh, extra_opts, simtel_path, corsika_path, infile))


def run_lightemission(events=3, photons=10946249, distance=100, cam_radius=30, xdisp=0, ydisp=0, spectrum=405,
                      ang_dist='/scratch/armstrongt/Workspace/CTA/MCValidation/data/ang_dist_2.dat',
                      out_file='/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_test.dat.gz'):
    os.system('%s/ff-1m '
              '--events %s '
              '--photons %s '
              '--distance %s '
              '--camera-radius %s '
              '--angular-distribution %s '
              '--spectrum %s '
              '--xy %s,%s '
              '-o %s' % (lightEmission_path, events, photons, distance, cam_radius, ang_dist, spectrum, xdisp, ydisp, out_file))


def run_corsika_simtel(args, infile, n, p):
    if not args.fixCorsika:
        # infl = '%s/corsika/run%04d.corsika.gz' % (args.outdir, int(infile[0][0]))
        # else:
        infl = '%s/corsika/run%04d.corsika.gz' % (args.outdir, int(infile[0][n]))

        if args.runLightEmission:
            print("@@@@ Running LightEmission Package\n\n")
            if args.nevents == "File":
                run_lightemission(events=infile[5][n], photons=infile[4][n],
                                  out_file=infl, ang_dist=args.angdist,
                                  distance=args.distance, cam_radius=args.camradius, xdisp=args.xdisp,
                                  ydisp=args.ydisp)
            else:
                run_lightemission(events=args.nevents, photons=infile[4][n],
                                  out_file=infl, ang_dist=args.angdist,
                                  distance=args.distance, cam_radius=args.camradius, xdisp=args.xdisp,
                                  ydisp=args.ydisp)

    outfl = '%s/sim_tel/run%04d.simtel.gz' % (args.outdir, int(infile[0][n]))
    if args.runSimTelarray:
        print("@@@@ Running Simtelarray\n\n")
        if args.discthresh == "File":
            if args.nsb == "File":
                run_simtel(infile=infl, outfile=outfl, nsb=infile[6][n], disc_thresh=infile[7][n],
                           extra_opts=args.extra_opts)
            else:
                run_simtel(infile=infl, outfile=outfl, nsb=args.nsb, disc_thresh=infile[7][n],
                           extra_opts=args.extra_opts)
        else:
            if args.nsb == "File":
                run_simtel(infile=infl, outfile=outfl, nsb=infile[6][n], disc_thresh=args.discthresh,
                           extra_opts=args.extra_opts)
            else:
                run_simtel(infile=infl, outfile=outfl, nsb=args.nsb, disc_thresh=args.discthresh,
                           extra_opts=args.extra_opts)

    return ('run%04d.simtel.gz' % int(infile[0][n]), 0 )

def run_corsika_simtel_noloop(args, infile):
    infl = '%s/corsika/run%04d.corsika.gz' % (args.outdir, int(infile[0]))
    outfl = '%s/sim_tel/run%04d.simtel.gz' % (args.outdir, int(infile[0]))
    if args.runLightEmission:
        print("@@@@ Running LightEmission Package\n\n")
        if args.nevents == "File":
            run_lightemission(events=infile[5], photons=infile[4],
                              out_file=infl, ang_dist=args.angdist,
                              distance=args.distance, cam_radius=args.camradius, xdisp=args.xdisp, ydisp=args.ydisp)
        else:
            run_lightemission(events=args.nevents, photons=infile[4],
                              out_file=infl, ang_dist=args.angdist,
                              distance=args.distance, cam_radius=args.camradius, xdisp=args.xdisp, ydisp=args.ydisp)
    if args.runSimTelarray:
        print("@@@@ Running Simtelarray\n\n")
        if args.discthresh == "File":
            if args.nsb == "File":
                run_simtel(infile=infl, outfile=outfl, nsb=infile[6], disc_thresh=infile[7],
                           extra_opts=args.extra_opts)
            else:
                run_simtel(infile=infl, outfile=outfl, nsb=args.nsb, disc_thresh=infile[7],
                           extra_opts=args.extra_opts)
        else:
            if args.nsb == "File":
                run_simtel(infile=infl, outfile=outfl, nsb=infile[6], disc_thresh=args.discthresh,
                           extra_opts=args.extra_opts)
            else:
                run_simtel(infile=infl, outfile=outfl, nsb=args.nsb, disc_thresh=args.discthresh,
                           extra_opts=args.extra_opts)
    return 0

def main():
    if not os.path.isdir(simtel_path) or not os.path.isdir(simtel_path) or not os.path.isdir(lightEmission_path):
        print('Need to set paths (hardcoded!)')

    parser = argparse.ArgumentParser(description='Run LightEmission and simtel')
    parser.add_argument('--infile', default='./runlist.txt', help='File containing run number, Npe and Nphotons')
    parser.add_argument('--outdir', default='/scratch/armstrongt/Workspace/CTA/MCValidation/data')
    parser.add_argument('--nevents', default=1, help='Number of events to run at each illumination')
    parser.add_argument('--runLightEmission', action='store_true', default=False, help='Run Light Emission Package')
    parser.add_argument('--angdist', default='/scratch/armstrongt/Workspace/CTA/MCValidation/data/ang_dist_2.dat',
                        help='file containing the angular distribution of the light source')
    parser.add_argument('--spec', default=405, help='spectrum of the light source (filename or value)')
    parser.add_argument('--distance', default='100', help='distance of lightsource from detector [cm]')
    parser.add_argument('--camradius', default='30',
                        help='radius of the fiducial sphere that contains the detector [cm]')
    parser.add_argument('--xdisp', default=0, help='displacement of the light source in the x direction')
    parser.add_argument('--ydisp', default=0, help='displacement of the light source in the y direction')
    parser.add_argument('--runSimTelarray', action='store_true', default=False, help='Run Simtelarray')
    parser.add_argument('--cfg', default='/%s/cfg/CTA/CTA-ULTRA6-SST-GCT-S.cfg' % simtel_path,
                        help='sim_telarray configuration file')
    parser.add_argument('--nsb', default=0, help='level of non-pulsed background light [MHz]')
    parser.add_argument('--discthresh', default=0, help='level of discriminator threshold, 0 for external trigger')
    parser.add_argument('--extra_opts', default=' ', help='extra options for simtelarray, each must have -C proceeding')
    parser.add_argument('--only_pixels', default=None, help='list of pixels to have turned on')
    parser.add_argument('--fixCorsika', default=False, action='store_true')
    parser.add_argument('--cores', default=None, help='Multiprocessing, how many cores to use')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.isdir('%s/corsika/' % (args.outdir)):
        os.makedirs('%s/corsika/' % (args.outdir))
    if not os.path.isdir('%s/sim_tel/' % (args.outdir)):
        os.makedirs('%s/sim_tel/' % (args.outdir))

    infile=None

    try:
        infile = np.loadtxt(args.infile, unpack=True)
    except FileNotFoundError:
        print('No such input file, please specify one with --infile FILE')

    try:
        if args.fixCorsika:
            print('Only running CORSIKA once')
            infl = '%s/corsika/run%04d.corsika.gz' % (args.outdir, int(infile[0][0]))
            if args.runLightEmission:
                print("@@@@ Running LightEmission Package\n\n")
                if args.nevents == "File":
                    run_lightemission(events=infile[5][0], photons=infile[4][0],
                                      out_file=infl, ang_dist=args.angdist,
                                      distance=args.distance, cam_radius=args.camradius, xdisp=args.xdisp,
                                      ydisp=args.ydisp)
                else:
                    run_lightemission(events=args.nevents, photons=infile[4][0],
                                      out_file=infl, ang_dist=args.angdist,
                                      distance=args.distance, cam_radius=args.camradius, xdisp=args.xdisp,
                                      ydisp=args.ydisp)


        if args.cores is None:
            for n, p in enumerate(infile[3]):
                run_corsika_simtel(args, infile, n, p)
        else:
            tasks = []
            pool = multiprocessing.Pool(int(args.cores))
            print('using %s out of %s cores' % (args.cores, multiprocessing.cpu_count()))
            for n, p in enumerate(infile[3]):
                tasks.append((args, infile, n, p))

            results = [pool.map_async(run_corsika_simtel, t) for t in tasks]

            for result in results:
                (fl, res) = result.get()
                print('File %s run, with exit status %s' % (fl,res))

    except TypeError as e:
        print(e, ' runfile appears to only have one line')
        exit()
        run_corsika_simtel_noloop(args, infile)


if __name__ == '__main__':
    main()
