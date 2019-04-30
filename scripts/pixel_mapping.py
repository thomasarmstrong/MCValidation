#Author J Watson

import os
from target_calib import CameraConfiguration


def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(file_dir, "outputs")
    output_path = os.path.join(output_dir, "checs_pixel_mapping.txt")

    config = CameraConfiguration("1.1.0")
    mapping = config.GetMapping()
    mappingsp = mapping.GetMappingSP()

    with open(output_path, 'w') as f:
        pixtype = """# PixType format:
# Par.  1: pixel type (here always 1)
#       2: PMT type (must be 0)
#       3: cathode shape type
#       4: visible cathode diameter [cm]
#       5: funnel shape type (see above)
#       6: funnel diameter (flat-to-flat for hex.) [cm]
#       7: depth of funnel [cm]
#       8: a) funnel efficiency "filename",   b) funnel plate transparency
#       9: a) optional wavelength "filename", b) funnel wall reflectivity
# In case a) in column 8, columns 3+7 are not used. If in case a) the
# optional file name for the wavelength dependence is provided, the
# overall scale in the file provided as parameter 8 is ignored because
# it is rescaled such that the average over all mirrors is equal to
# the given wavelength dependent value.
#
# Shape types: 0: circ., 1: hex(flat x), 2: sq., 3: hex(flat y)
#
#Angular Dep currently all set to one for checks

# Note that pixel size is scale from the actual 6.125 mm at the focal length
# planned with GATE telescopes (228.3 cm) to the focal length of ASTRI (215 cm).
# Similarly scaled are the pixel positions.
PixType    1 0 2 0.623   2 0.623 0.0   "transmission_pmma_vs_theta_20150422.dat" "transmission_pmma_vs_lambda_meas0deg_coat_82raws.dat"

# Pixel format:
# Par.  1: pixel number (starting at 0)
#       2: pixel type (must be 1)
#       3: x position [cm]
#       4: y position [cm]
#       5: drawer/module number
#       6: board number in module
#       7: channel number n board
#       8: board Id number ('0x....')
#       9: pixel on (is on if parameter is missing)

"""

        f.write(pixtype)

        for i in range(mapping.GetNPixels()):
            ipix = mapping.GetPixel(i)
            xpix = mapping.GetXPix(i) * 10**2
            ypix = mapping.GetYPix(i) * 10**2
            imod = mapping.GetSlot(i)
            ichan = mapping.GetTMPixel(i)
            l = "Pixel\t{}\t1\t{:.2f}\t{:.2f}\t{}\t0\t{}\t0x00\t1\n"
            lf = l.format(ipix, xpix, ypix, imod, ichan)
            f.write(lf)

        f.write('\n')

        for i in range(mappingsp.GetNSuperPixels()):
            nei = mappingsp.GetNeighbours(i, True)
            f.write("MajorityTrigger * of ")
            for isp in [i, *nei]:
                con = list(mappingsp.GetContainedPixels(isp))
                f.write(str(isp))
                f.write("[{}] ".format(','.join(str(x) for x in con)))
            f.write('\n')


if __name__ == '__main__':
    main()
