# MCValidation

The goal of this repository is to create some scripts and examples of how to produce and analise the MC data for the
MCValidation process, focusing on using the LightEmission package to replicate lab measurements. See [mcmeas.pdf](https://forge.in2p3.fr/projects/cta_analysis-and-simulations/repository/changes/Simulations/MCMeasurements/mcmeas.pdf)
which lays out some of the test measurements that will be used, and are generally grouped in the following cases: Pedestal and noise measurements,
basic photo-sensor response, pulsed light measurements with external trigger, pulsed light measurements with camera trigger and
electronic test pulses instead of light pulses. Before looking at each one of these it is important to evaluate the tools
that we need for this process.

# LightEmission Package

In order to simulate the lasers used in lab measurements and which can be read in by sim_telarray, we utalise the LightEmission
 package created by K Bernloehr (now part of the latest corsika_simtel distribution) which enables a generic light source to be
 defined by the number of emitted photons, light temporal, spectral and angular distribution (or single values).


![lab setup](Figures/LightEmission_setup.png)

One initial difficulty is defining the desired number of photo-electrons emitted as part of the light source. In lab measurements
the different light levels are generally created using a filterwheel, and the absolute illumination is calibrated by measuring the
detected number of photo-electrons and calibrate it using the single p.e. measurements [Needs confirmation and better description].
For now a simple approach has been adapted that takes the input requested geometry and the photon detection efficiency (along with
any other transmission factors) to calculate the required number of photons needed to produce a desired (average) number of photo-electrons.


See the script:
get_photons.py reads in a run list and provides the required number of photons emitted in LightEmission package to obtain the desired p.e. level

![input pe](Figures/inputTruepe.png)

although this doesn't currently take into account full distributions of wavelength or angle. The light source can then simply be
simulated by running the following

```
./ff-1m --events <Nevents> --photons <Nphotons> --distance <z> --camera-radius <Rcam> --angular-distribution <File/isotropic>
--spectrum <File/value> <(x, y)> -o <File>
```

run_simtel.py - just a script that runs the light emission package and sim_telarray. Also can generate a pixel mask file

example run:

python run_simtel.py --infile runlist.txt --outdir ~/Data/test_run --nevents 1 --runLightEmission --angdist ./angular_distribution.dat
--distance 100 --camradius 30 --runSimTelarray --nsb 0 --discthresh 0

compare_waveforms.py - simple script to view waveforms in parallel for lab and MC

![example waveform](Figures/compare_waveform.png)

exctract_charge_resolution.py 

python extract_charge_resolution.py --input_path ./d2018-02-09_DynRange_NSB0_mc --max_events 10 -o ./charge_resolution_test.h5 --calibrator HESSIOR1Calibrator -T 1 --plot_cam False --use_true_pe True

result can be plotted using 

python .../ctapipe/ctapipe/tools/plot_charge_resolution.py -f="['./charge_resolution_true_pe.h5', './charge_resolution_labpe.h5']"  -O ./comp_charge_res.png

![comp charge res](Figures/compare_charge_res.png)
