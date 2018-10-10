import numpy as np
import matplotlib.pyplot
import argparse



airlightspeed = 29.9792458/1.0002256 # /* [cm/ns] at H=2200 m */

# /* At least air-glow emission component of NSB light increases with airmass. */
# /* Star light and zodiacal light are independent of airmass. */
# /* Stray light from cities may rapidly increase with airmass but is ignored. */

def nsb_za_scale(airmass):
    airglow_frac = 0.6 # /* Assumed wavelength independent */
    return (1.0 - airglow_frac) + airglow_frac * airmass

# def read_table3 (fname, maxrow, *col1, *col2, *col3):
#    FILE *f;
#    double x, y, z;
#    char line[1024];
#    int iline = 0, n=0;
#
#    if ( (f = fileopen(fname,"r")) == NULL )
#    {
#       perror(fname);
#       return -1;
#    }
#
#    while ( fgets(line,sizeof(line)-1,f) != NULL )
#    {
#       iline++;
#       strip_comments(line);
#
#       if ( line[0] == '\0' )
#          continue;
#
#       if ( sscanf(line,"%lf %lf %lf",&x,&y,&z) != 3 )
#       {
#        	 fprintf(stderr,"Error in line %d of file %s\n",iline,fname);
#          fileclose(f);
#          return -1;
#       }
#       if ( n >= maxrow )
#       {
#        	 fprintf(stderr,"Too many entries in file %s (max=%d)\n",fname,maxrow);
#          fileclose(f);
#          return -1;
#       }
#       col1[n] = x;
#       col2[n] = y;
#       col3[n] = z;
#       n++;
#    }
#
#    printf("Table with %d rows has been read from file %s\n",n,fname)
#    fileclose(f)
#    return n
#
# int read_table5 (const char *fname, int maxrow, double *col1, double *col2, double *col3, double *col4, double *col5)
# {
#    FILE *f;
#    double x, y, z, u, v;
#    char line[1024];
#    int iline = 0, n=0;
#
#    if ( (f = fileopen(fname,"r")) == NULL )
#    {
#       perror(fname);
#       return -1;
#    }
#
#    while ( fgets(line,sizeof(line)-1,f) != NULL )
#    {
#       iline++;
#       strip_comments(line);
#
#       if ( line[0] == '\0' )
#          continue;
#
#       if ( sscanf(line,"%lf %lf %lf %lf %lf",&x,&y,&z,&u,&v) != 5 )
#       {
#        	 fprintf(stderr,"Error in line %d of file %s\n",iline,fname);
#          fileclose(f);
#          return -1;
#       }
#       if ( n >= maxrow )
#       {
#        	 fprintf(stderr,"Too many entries in file %s (max=%d)\n",fname,maxrow);
#          fileclose(f);
#          return -1;
#       }
#       col1[n] = x;
#       col2[n] = y;
#       col3[n] = z;
#       col4[n] = u;
#       col5[n] = v;
#       n++;
#    }
#
#    printf("Table with %d rows has been read from file %s\n",n,fname);
#    fileclose(f);
#
#    return n;
# }
#
# int read_spe_check(const char *fname)
# {
# #define MAX_SPE 2000
#    double xspe[MAX_SPE], xspe_ap[MAX_SPE], fspe_prompt[MAX_SPE], fspe_with_ap[MAX_SPE];
#    int n3 = read_table3(fname,MAX_SPE,xspe_ap,fspe_prompt,fspe_with_ap);
#    int n2 = read_table(fname,MAX_SPE,xspe,fspe_prompt);
#    int with_ap = 0;
#    double s=0., ss=0., ss2=0., sap=0., sap4=0.;
#    double amp, smean = 0., min_amp=0., max_amp=10.;
#    if ( n2 < 2 )
#    {
#       printf("Single-p.e. response file is not usable.\n");
#       return -1;
#    }
#    if ( n3 != n2 )
#       printf("Single p.e. response file is without afterpulsing distribution\n"
#              "or afterpulses are missing for some entries.\n");
#    else
#       with_ap = 1;
#    min_amp = xspe[0];
#    max_amp = xspe[n2-1];
#    for (amp=0.005; amp<max_amp; amp+=0.01)
#    {
#       double fpe = rpol(xspe,fspe_prompt,n2,amp);
#       if ( amp < min_amp )
#          continue;
#       s += fpe;
#       ss += fpe*amp;
#       ss2 += fpe*amp*amp;
#    }
#    if ( s > 0. )
#    {
#       double sm = ss / s;
#       double rmsom = sqrt((ss2/s)-(ss/s)*(ss/s)) / sm;
#       double xsf = sqrt(1.+rmsom*rmsom);
#       smean = sm;
#       printf("Single p.e. response file %s has \n"
#              "  - normalization = %f\n"
#              "  - mean amplitude = %f (without aftperpulsing)\n"
#              "  - r.m.s. of ampl. = %f (again without a.p.)\n"
#              "  - excess noise factor = %f\n",
#          fname, s * 0.01, smean, rmsom, xsf);
#    }
#    if ( with_ap )
#    {
#       for (amp=0.005*smean; amp<max_amp; amp+=0.01)
#       {
#        	 double fpe_ap = rpol(xspe_ap,fspe_with_ap,n3,amp);
#          if ( amp < min_amp )
#             continue;
#          sap += fpe_ap;
#          if ( amp > 4.*smean )
#             sap4 += fpe_ap;
#       }
#       if ( sap > 0. )
#          printf("  - afterpulsing fraction (> 4 mean p.e.) = %f\n", sap4/sap);
#    }
#    return n2;
# }


def main():
    parser = argparse.ArgumentParser(description='Get NSB rate from config')
    parser.add_argument('-fatm', help='Atmospheric transmission file', default='/Users/armstrongt/Software/CTA/Corsika_Simtel/2018-03-01_testing/sim_telarray/cfg/common/atm_trans_2150_1_0_0_0_2150.dat') #atm_trans_2150_1_10_0_0_2150.dat
    parser.add_argument('-alt', help='Observation level a.s.l.', default=2150.0)
    parser.add_argument('-teltrans', help='Telescope transmission (shadowing factor)', default=0.881)
    parser.add_argument('-camtrans', help='Camera transmission (w.l. independent)', default=1.00)
    parser.add_argument('-fqe', help='Quantum efficiency file', default='/Users/armstrongt/Documents/CTA/MonteCarlo/MCVarification/DateForNewModel/GCT/camera/config/SPE_Gentile_oxt0d08_spe0d05_d2018-10-04m.txt')
    parser.add_argument('-fref',  help='Mirror reflectivity file', default='/Users/armstrongt/Software/CTA/Corsika_Simtel/2018-03-01_testing/sim_telarray/cfg/CTA/inaf_m1_refl.dat')
    parser.add_argument('-fref2', help='Mirror reflectivity file (for secondary in dual mirror tel.)', default='/Users/armstrongt/Software/CTA/Corsika_Simtel/2018-03-01_testing/sim_telarray/cfg/CTA/inaf_m2_refl_alsio2_32deg_sim.dat')
    parser.add_argument('-fang',  help='Funnel angular acceptance file', default='/Users/armstrongt/Software/CTA/Corsika_Simtel/2018-03-01_testing/sim_telarray/cfg/common/funnel_perfect.dat')
    parser.add_argument('-fwl', help='Funnel wavelength acceptance file', )
    parser.add_argument('-fmir', help='Mirror geometry configuration file')
    parser.add_argument('-flen', help='Focal length (imaging) [m]')
    parser.add_argument('-fcur', help='Radius of curvature of dish (DC:r=f, par:r=2*f) [m]')
    parser.add_argument('-fflt', help='Additional filter file (at funnel entry or so)')
    parser.add_argument('-fspe', help='Also check single-p.e. response distribution', default = '/home/armstrongt/Software/CTA/MCValidation/configs/config-prod4-v1.0')
    parser.add_argument('-hpix', help='Diameter of hexagonal pixel [cm]')
    parser.add_argument('-spix', help='Diameter of square pixel [cm]')
    parser.add_argument('-m2', help='Secondary mirror optics (two reflections)')
    parser.add_argument('-nm', help='Use 1 nm steps rather than 10 nm steps')
    parser.add_argument('-nsb-extra', help='Show more columns with NSB information')
    parser.add_argument('-lambda_min', help='Lower limit of wavelength range [nm]', default=200)
    parser.add_argument('-lambda_max', help='Upper limit of wavelength range [nm]', default=700)
    parser.add_argument('-Xmax', help='Atmospheric depth of assumed light emission')
    parser.add_argument('-iatmo', help='No. of atmospheric density profile (Xmax->Hmax)')
    parser.add_argument('-za', help='Zenith angle [deg]')
    args = parser.parse_args()

    iatmo = 10
    start_thick=300.0










if __name__ == '__main__':
    main()