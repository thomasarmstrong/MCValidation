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


See the script get_photons.py which reads in a run list and provides the required number of photons emitted in LightEmission package to obtain the desired p.e. level.
Looking at the distribution of true mc p.e. recorded from the simulation compared to the requested input (see following image), we see that they agree reasonably well.
There is, as is expected a slight bias due to the geometry and, in this case the curved focal plane, where we expect to see fewer photo-electrons
towards the edge of the camera.

![input pe](Figures/inputTruepe.png)

Note that script doesnt doesn't currently take into account full distributions of wavelength or angle (issue #1). The light source can then simply be
simulated by running the following

```
./ff-1m --events <Nevents> --photons <Nphotons> --distance <z> --camera-radius <Rcam> --angular-distribution <File/isotropic> --spectrum <File/value> <(x, y)> -o <File>
```

which will produce a data file which is in the same format as a corsika file. This is put into sim_telarray using what ever configuration
that represents the camera you are simulating. Note that the additional flag BYPASS_OPTICS needs to be set in order to prevent the telescope structure
and mirrors. A helper script has been created, run_simtel.py which runs the light emission package and sim_telarray. Also can generate a pixel mask file

example run:

python run_simtel.py --infile runlist.txt --outdir ~/Data/test_run --nevents 1 --runLightEmission --angdist ./angular_distribution.dat
--distance 100 --camradius 30 --runSimTelarray --nsb 0 --discthresh 0

Additionally, there are some helper scripts for the helper script... (the example...py). But these are mainly here to keep track
of some different runs performed and will be replaced with config files shortly (see issue #2).


# Data Exploration


compare_waveforms.py - simple script to view waveforms in parallel for lab and MC

![example waveform](Figures/compare_waveform.png)

exctract_charge_resolution.py 

python extract_charge_resolution.py --input_path ./d2018-02-09_DynRange_NSB0_mc --max_events 10 -o ./charge_resolution_test.h5 --calibrator HESSIOR1Calibrator -T 1 --plot_cam False --use_true_pe True

result can be plotted using 

python .../ctapipe/ctapipe/tools/plot_charge_resolution.py -f="['./charge_resolution_true_pe.h5', './charge_resolution_labpe.h5']"  -O ./comp_charge_res.png

![comp charge res](Figures/compare_charge_res.png)


# Pedestal and Noise Measurements

> Baseline (pedestal) measurements without any light source and with increasing levels of nonpulsed
> background light (emulating NSB over the expected range in observations, e.g. from zero
> to maximum operational rate in the requirements). The documentation must include how the rates
> were determined. Where possible, such measurements should be available with both the readout
> window length as intended for later operation and with a window length as long as possible to
> extract also lower-frequency noise contributions. (Single sensors or camera parts. Relevant for
> MC parameters and algorithms related to pedestals, electronics noise, and dark count rates.)

# Basic Photo-Sensor Response

> For normal incidence the PDE (photon detection efficiency) for SiPM at the bias voltage as used
> in the camera. QE and CE (quantum efficiency / collection efficiency) for PMTs, over the light
> collector exit area, using a mask in front of the photocathode, at nominal gain. (Single sensors
> or camera parts. Not all of these measurements will have a direct comparison in simulations, e.g.
> when measuring the photocathode currents for the quantum efficiency of PMTs.)

> Light collector plus photosensor efficiency versus angle of incidence, either in absolute units
> as for normal incidence or relative to the efficiency of the masked PMT (and scaled of light collector
> exit area over entry area).

> Afterpulsing measurements for PMTs (between 4 and 40 p.e. or to amplitudes beyond which the
> afterpulse probability is < 10−9), revealing integral afterpulse fractions and spectrum of afterpulses.
> Either with non-pulsed light source or cumulative over all relevant afterpulsing delays. In
> order to have a reproducible p.e.-scale, raw data taken with camera electronics is recommended
> for this purpose, together with matching calibration measurement data. Other ways to measure
> afterpulsing are acceptable as long as the p.e. scale is well-defined.

> SiPM dark count measurements at the bias voltage as used in the camera, revealing dark count
> rate and pure optical-cross talk probability. In addition to measurements with the camera electronics,
> additional measurements with low-noise electronics are appreciated for this purpose, in
> particular for the width of the individual peaks in the amplitude distribution.

# Pulsed Light Measurements with External Trigger

Pulsed light measurements at low illumination levels (“single-p.e.”) together with off-pulse (“dark”)
control measurements. For disentangling the single-p.e. response from the superposition of multiple
photoelectrons, measurements at multiple illumination levels – best with known attenuation
ratios – from well below one p.e. up to a few p.e. are recommended but knowledge of absolute
levels is not necessary. At operational gain (PMT) or bias voltage (SiPM). Measurements with
the readout system of the camera are suitable for gain calibration (matching pedestal measurements
also needed). Measurements with separate low-noise readout suitable for r.m.s. width of
the individual peaks in the SiPM amplitude distribution and, for PMTs, for following the single-p.e.
amplitude distribution down to lower amplitudes than available with the camera readout.

Pulsed light measurements at moderate illumination levels (flat-fielding style) with external trigger,
for pulse shapes and timing accuracy. Proper documentation of light pulse intrinsic shape is
necessary. Also documentation on the stability of the external trigger with respect to the actual
light pulse. Changing illumination levels via known optical attenuation, also down to low levels for
corresponding single-p.e. calibration, is preferred but other types of attenuation are acceptable
as long as attenuation ratios, light pulse shape, and trigger delay are under control. These measurements
are most likely available as part of the charge resolution validation, B-xST-1010 and
the time resolution validation, B-xST-1030. They may include different levels of emulated NSB.
(Requires a significant part of a camera.)

Pulse shapes at different pulse illumination levels (available with previous set of measurements
for all camera types with trace readout). In addition to readout of normal trace length it may also
be useful to obtain measurements with longer readout covering the tails of the pulses.

Instead of the previous measurement, for cameras without trace readout it may be possible to
measure the (integral or peak) signal at different delays of the external trigger.

Pulsed light measurements at high illumination levels: changing pulse shapes, high-gain to lowgain
ratio, and non-linearity (at least up to required levels, typically 1000 p.e., optionally well
beyond).

# Pulsed Light Measurements with Camera Trigger

# Electronic Test Pulses Instead of Light Pulses