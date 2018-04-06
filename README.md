# MCValidation
Files:
test_geometry3D.py - main code that contained the relavent code to perform lab tests. Or at least to get the required number of emitted photons for a desired number of p.e. in the camera

![input pe](Figures/inputTruepe.png)



run_simtel.py - just a script that runs the light emission package and sim_telarray. Also can generate a pixel mask file

exctract_charge_resolution.py 

python extract_charge_resolution.py --input_path ./d2018-02-09_DynRange_NSB0_mc --max_events 10 -o ./charge_resolution_test.h5 --calibrator HESSIOR1Calibrator -T 1 --plot_cam False --use_true_pe True

result can be plotted using 

python .../ctapipe/ctapipe/tools/plot_charge_resolution.py -f="['./charge_resolution_true_pe.h5', './charge_resolution_labpe.h5']"  -O ./comp_charge_res.png

![comp charge res](Figures/compare_charge_res.png)
