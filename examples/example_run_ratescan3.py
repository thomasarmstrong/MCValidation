import os

os.system("python /home/armstrongt/Software/CTA/MCValidation/scripts/run_simtel.py --infile /home/armstrongt/Data/MCValidation/d2018-05-14_100mV_fw50pe/mcdata_dummy/runlist.txt "
          " --outdir /home/armstrongt/Data/MCValidation/d2018-05-14_100mV_fw50pe/mcdata_dummy3_5MHz "
          " --cfg /home/armstrongt/CHECMC/CHEConASTRI/SimTelarray/astri_chec_v002/CTA-PROD4-SST-ACHEC200mV.cfg "
          " --nevents 1000 "
          " --angdist configs/ang_dist_mpik_flat_20deg.dat "
          " --distance 155.2 "
          " --camradius 30 "
          " --runSimTelarray "
          # " --runLightEmission "
          " --fixCorsika "
          " --cores 5 "
          " --nsb 0.00 "
          " --discthresh File "
          " --xdisp 0.0 --ydisp 0.0 --spec 398 ")