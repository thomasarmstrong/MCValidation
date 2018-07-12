import numpy as np
import matplotlib.pyplot as plt
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

def focal_plane(x):
	# x=x/10
	p2 = -5.0e-3
	p4 = -1.25e-7
	p6 = -6.25e-12
	p8 = -3.90625e-16
	p10 = -2.734375e-20
	return (p2 * x ** 2 + p4 * x ** 4 + p6 * x ** 6 + p8 * x ** 8 + p10 * x ** 10)

pixel_dat = np.loadtxt('CHEC-S_camera_full_19-02-2018-1.dat', unpack=True)

z_focalplane = []
z_diff = []
for n, i in enumerate(pixel_dat[0]):
    xi = i/10
    yi = pixel_dat[1][n]/10
    zi = focal_plane(np.sqrt(xi ** 2 + yi ** 2))
    z_focalplane.append(zi)
    z_diff.append(zi-pixel_dat[2][n]/10)

print(max(z_diff))

geom = CameraGeometry.from_name("CHEC")
disp = CameraDisplay(geom)
# disp.set_limits_minmax(0, 300)
cb = disp.add_colorbar()
print(cb)
disp.image = z_diff

fig2= plt.figure(2)
plt.hist(z_diff)

plt.show()