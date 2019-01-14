import os
import argparse
import numpy as np
import multiprocessing

## Hardcoded parameters:
simtel_path = '/scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray'
corsika_path = '/scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/corsika-6990/run'
lightEmission_path = '/scratch/armstrongt/Workspace/CTA/MCValidation/src/LightEmission-pkg'



def run_simtel(outfile='../data/bypass2_enoise.simtel.gz', nsb=0.02, disc_thresh=230, extra_opts=' ',
               infile='/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_test.dat.gz'):
    """
    Simple helper script to launch sim_telarray (as with command line)
    :param outfile: Name of output simtelarray file
    :param nsb: Value of NSB to use [GHz]
    :param disc_thresh: Value of discriminator threshold to use in triggering [units are nominally p.e. but depends on 
    definition of discriminator amplitude
    :param extra_opts: Any extra options to pass to sim_telarray, should be in format -C PARAMETER=VALUE
    :param infile: name of corsika file to use
    :return: None
    """
    os.system('%s/sim_telarray '
              '-c /scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray/cfg/CTA/CTA-ULTRA6-SST-GCT-S-test.cfg '
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
              '-I/home/armstrongt/Software/CTA/MCValidation/configs/config-prod4-v1.0/ '
              '-I%s '
              '%s' % (simtel_path, outfile, nsb, disc_thresh, extra_opts, simtel_path, corsika_path, infile))


def run_lightemission(events=3, photons=10946249, distance=100, cam_radius=30, xdisp=0, ydisp=0, spectrum=405,
                      ang_dist='/scratch/armstrongt/Workspace/CTA/MCValidation/data/ang_dist_2.dat',
                      out_file='/scratch/armstrongt/Workspace/CTA/MCValidation/data/beamed_test.dat.gz'):
    """
    Simple helper script to launch the ff-1m script part of the LightEmission package
    :param events: Number of events to simulate
    :param photons: Number of photons to produce (this is the total number and should be calculated based on the desired p.e./pixel)
    :param distance: distance of light source to camera focal plane [cm]
    :param cam_radius: radius of fiducial sphere that contains the camera focal plane [cm]
    :param xdisp: displacement of light source in the x direction [cm]
    :param ydisp: displacement of light source in the y dierection [cm]
    :param spectrum: wavelength of light source to use if singular value, or distribution if file is provided
    :param ang_dist: angular distribution of light source
    :param out_file: name of the output file.
    :return: None
    """
    os.system('%s/ff-1m '
              '--events %s '
              '--photons %s '
              '--distance %s '
              '--camera-radius %s '
              '--angular-distribution %s '
              '--spectrum %s '
              '--xy %s,%s '
              '-o %s' % (lightEmission_path, events, photons, distance, cam_radius, ang_dist, spectrum, xdisp, ydisp, out_file))


def run_corsika_simtel(params):
    """
    function for running the helper function, separated out for multiprocessing reasons
    :param params: input command line options
    :return: returns name of output file
    """
    #################### CORSIKA STEP ###################

    args,infile,n,p = params
    if not args.fixCorsika:
        # Continue with normal loop, corsika file generated for each line in runlist
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
    else:
        # Keep corsika name fixed to first line of runlist
        infl = '%s/corsika/run%04d.corsika.gz' % (args.outdir, int(infile[0][0]))


    #################### SIMTEL STEP ###################

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
    """
    function for running the helper function, only called if runlist only has a single line.
    :param args: input command line options
    :param infile: name of the input CORSIKA file
    :return: 0
    """
    infl = '%s/corsika/run%04d.corsika.gz' % (args.outdir, int(infile[0]))
    outfl = '%s/sim_tel/run%04d.simtel.gz' % (args.outdir, int(infile[0]))

    #################### CORSIKA STEP ###################
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

    #################### SIMTEL STEP ###################
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

    #Check if you need to set hardcoded path
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

    # if path to output data does not exist, create it.
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
        # Try and loop over runlist file.
        if args.fixCorsika:
            ## Needs to be outside multiprocessing loop as only needs to be run once.
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
                run_corsika_simtel([args, infile, n, p])
        else:
            tasks = []
            pool = multiprocessing.Pool(int(args.cores))
            print('using %s out of %s cores' % (args.cores, multiprocessing.cpu_count()))
            for n, p in enumerate(infile[3]):
                tasks.append((args, infile, n, p))

            print(pool.map(run_corsika_simtel, tasks))

    except TypeError as e:
        print(e, ' runfile appears to only have one line')
        run_corsika_simtel_noloop(args, infile)


if __name__ == '__main__':
    main()
