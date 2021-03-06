import os


def make_mask(on_pixels, cam_file_str, out_file_str):
    cam_file = open(cam_file_str, 'r')
    out_file = open(out_file_str, 'w')
    for line in cam_file:
        if line.startswith('Pixel'):
            data = line.strip().split(' ')
            if int(data[0].split('\t')[1]) in on_pixels:
                out_file.write(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + '1\n')
            else:
                out_file.write(data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3] + ' ' + '0\n')
        else:
            out_file.write(line)
    cam_file.close()
    out_file.close()


#print('making mask')
#make_mask([1385,1386,1337,1338,1339,1290,1291,1292,1243,1244],'/home/armstrongt/Software/CTA/MCValidation/configs/checs_pixel_mapping_fixedgainvar.dat','/home/armstrongt/Software/CTA/MCValidation/configs/checs_pixel_mapping_fixedgainvar_masked.dat')

#os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /store/armstrongt/MCValidation/Trigger_Efficiency/signal/runlist.txt "
#          " --outdir /store/armstrongt/MCValidation/Trigger_Efficiency/signal "
#          " --nevents File "
#          " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#          " --distance 155.2 "
#          " --camradius 30 "
#          " --runSimTelarray "
#          #" --runLightEmission "
#          " --fixCorsika "
#          " --cores 5 "
#          " --nsb 0.005 "
#          " --discthresh File "
#          " --extra_opts \" "
#          " -C CAMERA_CONFIG_FILE=/home/armstrongt/Software/CTA/MCValidation/configs/checs_pixel_mapping_fixedgainvar.dat  "
#          " -C quantum_efficiency=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/10_quantumEfficiency/S10362-11-050U-3V-format.dat "
#          " -C discriminator_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/disc_shape_CHEC-S_27042018.dat "
#          " -C fadc_pulse_shape=/home/armstrongt/Workspace/CTA/MCValidation/data/config/camera/15_pulseShape/pulse_CHEC-S_FADC_27042018.dat "
#          " -C pm_photoelectron_spectrum=/scratch/armstrongt/Workspace/CTA/MCValidation/data/config/camera/04_SPE/checs_spe_spectrum_v2_normalised.txt "
#          " -C pm_voltage_variation=0 "
#          " -C gain_variation=0.0 "
#          " -C QE_variation=0.0 "
#          " -C FADC_PEDESTAL=10000 "
#          " -C FADC_AMPLITUDE=2.77 "
#          " -C FADC_Noise=1.4 "
#          " -C FADC_var_pedestal=0.07 "
#          " -C FADC_SUM_OFFSET=15 "
#          " -C FADC_Sum_Bins=96 "
#          " -C FADC_BINS=128 "
#          " -C FADC_VAR_SENSITIVITY=0 "
#          " -C discriminator_amplitude=1 "
#          " -C TRIGGER_PIXELs=2 "
#          " -C TRIGGER_CURRENT_LIMIT=1000000 "
#          " -C teltrig_min_sigsum=0 "
#          " -C discriminator_sigsum_over_threshold=0 "
#          " -C discriminator_var_sigsum_over_threshold=0 "
#          " -C discriminator_gate_length=-8 "
#          " -C discriminator_rise_time=0 "
#          " -C discriminator_fall_time=0 "
#          " -C discriminator_output_amplitude=1 "
#          " -C discriminator_output_var_percent=0 "
#          " -C discriminator_var_threshold=0.2 \"  "
#          "--xdisp 0.0 --ydisp 0.0 --spec 398 ")


os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /store/armstrongt/MCValidation/Trigger_Efficiency/noise/runlist.txt "
          " --outdir /store/armstrongt/MCValidation/Trigger_Efficiency/noise "
          " --nevents File "
          " --angdist configs/ang_dist_mpik_flat_20deg.dat "
          " --distance 155.2 "
          " --camradius 30 "
          " --runSimTelarray "
          #" --runLightEmission "
          " --fixCorsika "
          " --cores 5 "
          " --nsb 0.005 "
          " --discthresh File "
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
          " -C TRIGGER_CURRENT_LIMIT=1000000 "
          " -C teltrig_min_sigsum=0 "
          " -C discriminator_sigsum_over_threshold=0 "
          " -C discriminator_var_sigsum_over_threshold=0 "
          " -C discriminator_gate_length=-8 "
          " -C discriminator_rise_time=0 "
          " -C discriminator_fall_time=0 "
          " -C discriminator_output_amplitude=1 "
          " -C discriminator_output_var_percent=0 "
          " -C discriminator_var_threshold=0.2 \"  "
          "--xdisp 0.0 --ydisp 0.0 --spec 398 ")



#os.system('rm -f /home/armstrongt/Software/CTA/MCValidation/configs/checs_pixel_mapping_fixedgainvar_masked.dat')
