import os


def make_mask(on_pixels, cam_file_str, out_file_str):
    cam_file = open(cam_file_str, 'r')
    out_file = open(out_file_str, 'w')
    for line in cam_file:
        if line.startswith('Pixel'):
            data = line.strip().split('\t')
            #print(data)
            if int(data[1]) in on_pixels:
                out_file.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\t' + data[3] + '\t' + data[4] + '\t' + data[5] + '\t' + data[6] + '\t' + data[7] + '\t' + data[8] + '\t' + '1\n')
            else:
                out_file.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\t' + data[3] + '\t' + data[4] + '\t' + data[5] + '\t' + data[6] + '\t' + data[7] + '\t' + data[8] + '\t' + '0\n')
        else:
            out_file.write(line)
    cam_file.close()
    out_file.close()


#print('making mask')
#make_mask([1385,1386,1337,1338,1339,1290,1291,1292,1243,1244],'/home/armstrongt/Software/CTA/MCValidation/configs/checs_pixel_mapping_fixedgainvar.dat','/home$



#os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/d2018-05-14_100mV_fw50pe/mcdata/runlist.txt "
#          " --outdir /home/armstrongt/Data/MCValidation/d2018-05-14_100mV_fw50pe/mcdata3_5MHz "
#          " --nevents 100 "
#          " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#          " --distance 155.2 "
#          " --camradius 30 "
#          " --runSimTelarray "
#          # " --runLightEmission "
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
#          " -C teltrig_min_sigsum=0 "
#          " -C discriminator_sigsum_over_threshold=0 "
#          " -C discriminator_var_sigsum_over_threshold=0 "
#          " -C discriminator_gate_length=-35 "
#          " -C discriminator_rise_time=0 "
#          " -C discriminator_fall_time=0 "
#          " -C discriminator_output_amplitude=1 "
#          " -C discriminator_output_var_percent=0 "
#          " -C discriminator_var_threshold=0.2 \"  "
#          "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

#os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/d2018-05-14_100mV_fw50pe/mcdata_dummy/runlist.txt "
#          " --outdir /home/armstrongt/Data/MCValidation/d2018-05-14_100mV_fw50pe/mcdata_dummy3_5MHz "
#          " --nevents 100 "
#          " --angdist configs/ang_dist_mpik_flat_20deg.dat "
#          " --distance 155.2 "
#          " --camradius 30 "
#          " --runSimTelarray "
#          # " --runLightEmission "
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
#          " -C teltrig_min_sigsum=0 "
#          " -C discriminator_sigsum_over_threshold=0 "
#          " -C discriminator_var_sigsum_over_threshold=0 "
#          " -C discriminator_gate_length=-35 "
#          " -C discriminator_rise_time=0 "
#          " -C discriminator_fall_time=0 "
#          " -C discriminator_output_amplitude=1 "
#          " -C discriminator_output_var_percent=0 "
#          " -C discriminator_var_threshold=0.2 \"  "
#          "--xdisp 0.0 --ydisp 0.0 --spec 398 ")

# test new config

pix1 = [793,793,793,793,793,793,792,794]
pix2 = [1201,881,1241,792,794,795,795,795]

pixids = [[792,793,794,795,880,881,882,883],[792,793,794,795,1240,1241,1242,1243],[792,793,794,795,1200,1201,1202,1203],[792,793,794,795,784,785,786,787],[792,793,794,795,796,797,798,799],[792,793,794,795,788,789,790,791],[796,797,798,799,788,789,790,791], [784,785,786,787,788,789,790,791]]

for i in range(8):
    print('making mask')
    make_mask(pixids[i],'/scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray/cfg/CTA/checs_pixel_mapping_v2.dat', '/scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray/cfg/CTA/checs_pixel_mapping_v2_masked.dat')

    os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel_new.py --infile /store/armstrongt/MCValidation/Rate_Scan/light/runlist.txt "
              " --outdir /store/armstrongt/MCValidation/Rate_Scan32MHz/light/pix%s_%s/ "
              " --nevents File "
              " --angdist configs/ang_dist_mpik_flat_20deg.dat "
              " --distance 155.2 "
              " --camradius 30 "
              " --runSimTelarray "
              " --fixCorsika "
              " --cores 5 "
              " --nsb 0.032 "
              " --discthresh File "
              " --extra_opts \" "
              " -C camera_config_file=/scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray/cfg/CTA/checs_pixel_mapping_v2_masked.dat \" "
              "--xdisp 0.0 --ydisp 0.0 --spec 398 " % (pixids[i][0], pixids[i][4]))

    os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel_new.py --infile /store/armstrongt/MCValidation/Rate_Scan/dummy/runlist.txt "
              " --outdir /store/armstrongt/MCValidation/Rate_Scan32MHz/dummy/pix%s_%s/ "
              " --nevents File "
              " --angdist configs/ang_dist_mpik_flat_20deg.dat "
              " --distance 155.2 "
              " --camradius 30 "
              " --runSimTelarray "
              " --fixCorsika "
              " --cores 5 "
              " --nsb 0.032 "
              " --discthresh File "
              " --extra_opts \" "
              " -C camera_config_file=/scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray/cfg/CTA/checs_pixel_mapping_v2_masked.dat \" "
              "--xdisp 0.0 --ydisp 0.0 --spec 398 " % (pixids[i][0], pixids[i][4]))

    os.system('rm -rf /scratch/armstrongt/Software/CTA/CorsikaSimtel/2018-09-28_testing/sim_telarray/cfg/CTA/checs_pixel_mapping_v2_masked.dat')
