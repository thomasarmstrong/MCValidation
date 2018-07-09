import os

# Origional with no noise
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/origional_no_noise "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")
#
# # No Noise, No gain var
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/no_gainvar "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.0 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")
#
# # No QE var
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/no_qevar "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.0 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")
#
#
# # No FADC Noise
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/no_fadcnoise "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.0 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")
#
# # # No FADC Var pedestal
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/no_fadcvarped "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.0 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

# Prod3 SPE
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/prod3SPE "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

# SPE Not normalised
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/no_normalise "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

# # Low optical cross talk
# #
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/low_oxt "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Software/CTA/MCValidation/configs/checs_spe_spectrum_low_OXT_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

#Low Illumination
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/lowillum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/lowillum "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Software/CTA/MCValidation/configs/checs_spe_spectrum_low_OXT_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

# Origional with no noise
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/new_spec "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/home/armstrongt/Software/CTA/MCValidation/configs/checs_spe_spectrum_long_norm.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.1 "
#           " -C QE_variation=0.1 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.5 "
#           " -C FADC_Noise=0.69 "
#           " -C FADC_var_pedestal=0.1 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

# New Parameters
#
# os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
#           " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/new_params2 "
#           " --nevents File "
#           " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#           " --distance 155.2 "
#           " --camradius 30 "
#           " --runSimTelarray "
#           " --nsb 0.0 "
#           " --discthresh 0 "
#           " --extra_opts \" "
#           " -C CAMERA_CONFIG_FILE=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/02_cameraConfigFile/checs_pixel_mapping_v2.dat "
#           " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#           " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#           " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#           " -C pm_photoelectron_spectrum=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_v2_normalised.txt "
#           " -C pm_voltage_variation=0 "
#           " -C gain_variation=0.074 "
#           " -C QE_variation=0.09 "
#           " -C FADC_PEDESTAL=10000 "
#           " -C FADC_AMPLITUDE=2.77 "
#           " -C FADC_Noise=1.4 "
#           " -C FADC_var_pedestal=0.07 "
#           " -C FADC_SUM_OFFSET=15 "
#           " -C FADC_Sum_Bins=96 "
#           " -C FADC_BINS=128 "
#           " -C discriminator_amplitude=1 "
#           " -C TRIGGER_PIXELs=2 "
#           " -C teltrig_min_sigsum=0 "
#           " -C discriminator_sigsum_over_threshold=0 "
#           " -C discriminator_var_sigsum_over_threshold=0 "
#           " -C discriminator_gate_length=-8 "
#           " -C discriminator_rise_time=0 "
#           " -C discriminator_fall_time=0 "
#           " -C discriminator_output_amplitude=1 "
#           " -C discriminator_output_var_percent "
#           " -C discriminator_var_threshold=0.2 \"  "
#           "--xdisp 0.0 --ydisp 0.0 --spec 398 ")


# test fixed gain_var and qe_var
os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/test_spe_spectrum/runlist.txt "
          " --outdir /home/armstrongt/Data/MCValidation/test_spe_spectrum/test_fixgain "
          " --nevents File "
          " --angdist configs/ang_dist_mpik_flat_20deg.dat "
          " --distance 155.2 "
          " --camradius 30 "
          " --runSimTelarray "
          " --nsb 0.0 "
          " --discthresh 0 "
          " --extra_opts \" "
          " -C CAMERA_CONFIG_FILE=/home/armstrongt/Software/CTA/MCValidation/configs/checs_pixel_mapping_fixedgainvar.dat  "
          " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
          " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
          " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
          " -C pm_photoelectron_spectrum=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_v2_normalised.txt "
          " -C pm_voltage_variation=0 "
          " -C gain_variation=0.0 "
          " -C QE_variation=0.0 "
          " -C FADC_PEDESTAL=10000 "
          " -C FADC_AMPLITUDE=2.77 "
          " -C FADC_Noise=1.4 "
          " -C FADC_var_pedestal=0.07 "
          " -C FADC_SUM_OFFSET=15 "
          " -C FADC_Sum_Bins=96 "
          " -C FADC_BINS=128 "
          " -C FADC_VAR_SENSITIVITY=0 "
          " -C discriminator_amplitude=1 "
          " -C TRIGGER_PIXELs=2 "
          " -C teltrig_min_sigsum=0 "
          " -C discriminator_sigsum_over_threshold=0 "
          " -C discriminator_var_sigsum_over_threshold=0 "
          " -C discriminator_gate_length=-8 "
          " -C discriminator_rise_time=0 "
          " -C discriminator_fall_time=0 "
          " -C discriminator_output_amplitude=1 "
          " -C discriminator_output_var_percent=0 "
          " -C FLATFIELDING=0 "
          " -C discriminator_var_threshold=0.2 \"  "
          "--xdisp 0.0 --ydisp 0.0 --spec 398 ")
