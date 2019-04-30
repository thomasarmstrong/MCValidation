import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math as m
from scipy.spatial import cKDTree as KDTree
from matplotlib.collections import PatchCollection
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches

import astropy.units as u
mc_cam = open('/Users/armstrongt/Workspace/CTA/CHECMC/Workspace/CHEConASTRI/SimTelarray/astri_chec_v005/camera_CHEC-S_x9r.dat', 'r')
cad_cam = open('/Users/armstrongt/Documents/CTA/MonteCarlo/MCVarification/DateForNewModel/GCT/camera/02_cameraConfigFile/CHEC-S_camera_full_19-02-2018-1.dat', 'r')

cad_cam2 = open('/Users/armstrongt/Documents/CTA/MonteCarlo/MCVarification/DateForNewModel/GCT/camera/02_cameraConfigFile/camera_CHEC-S_prototype_26042018.dat','r')
fig = plt.figure(1)
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ps = 0.623/2


def plotOneAxisDef(plt, **axesPars):
	# Check that all necessary axes parameters were passed
	if not ('xTitle' in axesPars and 'yTitle' in axesPars and 'x' in axesPars and 'y' in axesPars
			and 'rotateAngle' in axesPars and 'fc' in axesPars and 'ec' in axesPars and 'invertYaxis' in axesPars):
		print('Input parameters missing when calling plotOneAxisDef!\n')
		return

	xTitle = axesPars['xTitle']
	yTitle = axesPars['yTitle']
	x, y = (axesPars['x'], axesPars['y'])

	r = 0.1  # size of arrow
	sign = 1.
	if axesPars['invertYaxis']:
		sign *= -1.
	xText1 = x + sign * r * np.cos(axesPars['rotateAngle'])
	yText1 = y + r * np.sin(0 + axesPars['rotateAngle'])
	xText2 = x + sign * r * np.cos(np.pi / 2. + axesPars['rotateAngle'])
	yText2 = y + r * np.sin(np.pi / 2. + axesPars['rotateAngle'])

	plt.gca().annotate(xTitle, xy=(x, y), xytext=(xText1, yText1), xycoords='axes fraction',
					   ha='center', va='center', size='xx-large',
					   arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0, fc=axesPars['fc'], ec=axesPars['ec']))

	plt.gca().annotate(yTitle, xy=(x, y), xytext=(xText2, yText2), xycoords='axes fraction',
					   ha='center', va='center', size='xx-large',
					   arrowprops=dict(arrowstyle='<|-', shrinkA=0, shrinkB=0, fc=axesPars['fc'], ec=axesPars['ec']))

	return

def plotAxesDef(plt, rotateAngle):
	invertYaxis = False
	xLeft = 0.7  # Position of the left most axis
	# if not telNow in self.twoMirrorTels:
	# 	if not self.cameraInSkyCoor:
	invertYaxis = False
	xLeft = 0.8

	xTitle = r'$x_{\!pix}$'
	yTitle = r'$y_{\!pix}$'
	x, y = (xLeft, 0.12)
	# The rotation of LST (above 100 degrees) raises the axes. In this case, lower the starting point.
	if np.rad2deg(rotateAngle) > 100:
		y -= 0.09
		x -= 0.05
	axesPars = {'xTitle': xTitle, 'yTitle': yTitle, 'x': x, 'y': y,
				'rotateAngle': rotateAngle - (1 / 2.) * np.pi, 'fc': 'black', 'ec': 'black', 'invertYaxis': invertYaxis}
	plotOneAxisDef(plt, **axesPars)

	xTitle = r'$x_{\!cam}$'
	yTitle = r'$y_{\!cam}$'
	x, y = (xLeft + 0.15, 0.12)
	axesPars = {'xTitle': xTitle, 'yTitle': yTitle, 'x': x, 'y': y,
				'rotateAngle': (3 / 2.) * np.pi, 'fc': 'blue', 'ec': 'blue', 'invertYaxis': invertYaxis}
	plotOneAxisDef(plt, **axesPars)

	xTitle = 'Alt'
	yTitle = 'Az'
	x, y = (xLeft + 0.15, 0.25)
	axesPars = {'xTitle': xTitle, 'yTitle': yTitle, 'x': x, 'y': y,
				'rotateAngle': (3 / 2.) * np.pi, 'fc': 'red', 'ec': 'red', 'invertYaxis': invertYaxis}
	plotOneAxisDef(plt, **axesPars)

	return


def find_neighbour_pixels(pix_x, pix_y, rad):
	"""use a KD-Tree to quickly find nearest neighbours of the pixels in a
    camera. 

    Parameters
    ----------
    pix_x : array_like
        x position of each pixel
    pix_y : array_like
        y position of each pixels
    rad : float
        radius to consider neighbour it should be slightly larger
        than the pixel diameter.

    Returns
    -------
    array of neighbour indices in a list for each pixel

    """

	points = np.array([pix_x, pix_y]).T
	indices = np.arange(len(pix_x))
	kdtree = KDTree(points)
	neighbours = [kdtree.query_ball_point(p, r=rad) for p in points]
	for nn, ii in zip(neighbours, indices):
		nn.remove(ii)  # get rid of the pixel itself
	return neighbours


def cart2sph(x,y,z):
    l = x**2 + y**2
    r = np.sqrt(l + z**2)
    elev = m.atan2(z,np.sqrt(l))
    az = m.atan2(y,x)
    return r, elev, az


def sph2cart(r, elev, az):
    x = r * np.cos(elev) * np.cos(az)
    y = r * np.cos(elev) * np.sin(az)
    z = r * np.sin(elev)
    return x, y, z
def focal_plane(x):
	# x=x/10
	p2 = -5.0e-3
	p4 = -1.25e-7
	p6 = -6.25e-12
	p8 = -3.90625e-16
	p10 = -2.734375e-20
	return (p2 * x ** 2 + p4 * x ** 4 + p6 * x ** 6 + p8 * x ** 8 + p10 * x ** 10)

x=[]
y=[]
z=[]
z3= []
arr=[]
pixn=0
pixn_list=[]
r=[]
r_error=[]
dz=[]
dz_error=[]


for line in mc_cam:
	#continue
	if line.startswith('Pixel'):
		# if pixn > 4*8*8 + 6*8*8 and pixn < 4*8*8 + 7*8*8:
		# 	continue
		data = line.strip().split()
		# print(len(data))
		# if float(data[2]) > 0 and float(data[2]) < 5 and float(data[3]) >0 and float(data[3]) < 5:
		xi = float(data[3])
		yi = float(data[4])
		module = float(data[5])
		r_dist = np.sqrt(xi**2+yi**2)
		zi = focal_plane(r_dist)
		r.append(r_dist)
		dz.append(zi)

		# ax.scatter(xi, yi, zi, marker='x', color='C%s' % int(module))
		x.append(xi)
		y.append(yi)
		z.append(zi)
		one_face_y = np.array([yi - ps, yi - ps, yi + ps, yi + ps,
							   yi - ps])
		one_face_x = np.array([xi - ps, xi + ps, xi + ps, xi - ps, xi - ps, ])
		one_face_z = np.array([zi, zi, zi, zi, zi])
		# one_face = np.array([[xi-ps, yi-ps, zi], [xi+ps, yi-ps, zi], [xi+ps, yi+ps, zi], [xi-ps, yi+ps, zi], [xi-ps, yi-ps, zi]])
		# plt.Polygon(one_face)
		arr.append([float(data[2]),float(data[3])])
		# mc_cam_one.write("%s\t%s 1\t%.3f\t%.3f\t%s\t%s\n" % (data[0],pixn,float(data[2])-2.6715625,float(data[3])-2.6715625,data[4],data[5]))
		pixn_list.append(str(pixn))
		# plt.text(float(data[2]),float(data[3]), s=str(pixn))
		pixn=pixn+1
	elif line.startswith('MajorityTrigger'):
		continue
	else:
		# mc_cam_one.write(line)
		continue

x2=[]
y2=[]
z2=[]
module2=[]
# arr=[]
# pixn=0
# pixn_list=[]
n1=0
n2=1
# cmap = plt.cm.viridis
pixID=[]
r2=[]
dz2=[]



for line in cad_cam2:
	# print(line)
	if line.startswith('Pixel'):
		# if pixn > 4*8*8 + 6*8*8 and pixn < 4*8*8 + 7*8*8:
		# 	continue
		data = line.strip().split()
		# print(data)
		# if float(data[2]) > 0 and float(data[2]) < 5 and float(data[3]) >0 and float(data[3]) < 5:
		xi = float(data[3])
		yi = float(data[4])
		module = float(data[4])
		module2.append(module)
		r_dist=np.sqrt(xi**2+yi**2)
		zi = focal_plane(r_dist)
		r2.append(r_dist)
		dz2.append(zi)
		# ax.scatter(xi, yi, zi, marker='x', color='C%s' % int(module))
		x2.append(xi)
		y2.append(yi)
		# z2.append(zi)
		# one_face_y = np.array([yi - ps, yi - ps, yi + ps, yi + ps,
		# 					   yi - ps])
		# one_face_x = np.array([xi - ps, xi + ps, xi + ps, xi - ps, xi - ps, ])
		# one_face_z = np.array([zi, zi, zi, zi, zi])
		# one_face = np.array([[xi-ps, yi-ps, zi], [xi+ps, yi-ps, zi], [xi+ps, yi+ps, zi], [xi-ps, yi+ps, zi], [xi-ps, yi-ps, zi]])
		# plt.Polygon(one_face)
		# arr.append([float(data[2]),float(data[3])])
		# mc_cam_one.write("%s\t%s 1\t%.3f\t%.3f\t%s\t%s\n" % (data[0],pixn,float(data[2])-2.6715625,float(data[3])-2.6715625,data[4],data[5]))
		pixn_list.append(str(pixn))
		# plt.text(float(data[2]),float(data[3]), s=str(pixn))
		pixn=pixn+1
	elif line.startswith('MajorityTrigger'):
		continue
	else:
		# mc_cam_one.write(line)
		continue


r3=[]
dz3=[]

for n, line in enumerate(cad_cam):
	# if line.startswith('Pixel'):
	pixID.append(n)
	n1+=1
	if n1 == 64:
		print(n2)
		n2+=1
		n1=0
	# print(n)
	data = line.strip().split('\t')
	# print(len(data))
	# if float(data[2]) > 0 and float(data[2]) < 5 and float(data[3]) >0 and float(data[3]) < 5:
	xi = float(data[0])/10
	yi = float(data[1])/10
	zi = float(data[2])/10
	r3.append(np.sqrt((float(data[0])/10)**2+(float(data[1])/10)**2))
	dz3.append(zi)
	one_face_y = np.array([yi - ps, yi - ps, yi + ps, yi + ps,
						 yi - ps])
	one_face_x = np.array([xi - ps,xi + ps, xi + ps,xi - ps,  xi - ps, ])
	one_face_z = np.array([zi,zi,zi,zi,zi])
	zi3 = focal_plane(np.sqrt(xi ** 2 + yi ** 2))
	# ax.plot(one_face_x, one_face_y, one_face_z, color='C0')
	# ax.scatter(xi, yi, zi, marker='x', color='C%s' %int(n2-1))
	# x2.append(xi)
	# y2.append(yi)
	z2.append(zi)
	z3.append(zi3)
		# one_face = np.array([[xi-ps, yi-ps, zi], [xi+ps, yi-ps, zi], [xi+ps, yi+ps, zi], [xi-ps, yi+ps, zi], [xi-ps, yi-ps, zi]])
		# plt.Polygon(one_face)
		# arr.append([float(data[2]),float(data[3])])
		# mc_cam_one.write("%s\t%s 1\t%.3f\t%.3f\t%s\t%s\n" % (data[0],pixn,float(data[2])-2.6715625,float(data[3])-2.6715625,data[4],data[5]))
		# pixn_list.append(str(pixn))
		# plt.text(float(data[2]),float(data[3]), s=str(pixn))
		# pixn=pixn+1
	# elif line.startswith('MajorityTrigger'):
	# 	continue
	# else:
		# mc_cam_one.write(line)
		# continue


start = 4*8*8+6*8*8+ 6*8*8 + 16
end = 4*8*8+6*8*8+ 7*8*8 -16
# # print(np.mean(x), np.mean(y))
ax.scatter(x2[start:end],z3[start:end],marker='s', label='focal plane parameter')
ax.scatter(x2[start:end],z2[start:end],marker='x', label='from Duncan')
ax.scatter(x[start:end],z[start:end],marker='.', label='Old mc values', color='r')
ax.legend()

for i in range(len(z3[start:end])):
	ax2.scatter(x2[start:end][i], z3[start:end][i]-z2[start:end][i], marker='x',color='k')

ax2.set_xlabel('x [cm]')
ax2.set_ylabel(r'$z_{meas}- z_{focal}$')
ax.set_ylabel('z [cm]')
plt.legend()

fig2 = plt.figure(2)
ax3 =fig2.add_subplot(111)
pixelDiameter = 0.62


xyPixPos = np.column_stack((x,y))
neighbours = find_neighbour_pixels(xyPixPos[:,0], xyPixPos[:,1], 1.4*pixelDiameter)
gaps = []
pixels, edgePixels, offPixels = list(), list(), list()
pixels2, edgePixels2, offPixels2 = list(), list(), list()
edgePixelIndices = list()
for i_pix, pixel in enumerate(xyPixPos):
	square = mpatches.Rectangle((pixel[0], pixel[1]), width=pixelDiameter, height=pixelDiameter)
	# if len(neighbours[i_pix]) == 4:
	# 	edgePixelIndices.append(i_pix)
	# 	edgePixels.append(square)
	# else:
	# if pixel[0] < 0:
	pixels.append(square)
	# else:
	# 	pixels2.append(square)
	for i in neighbours[i_pix]:
		d = np.sqrt((x[i_pix] - x[i]) ** 2)
		if m.isnan(d):
			continue
		else:
			gaps.append(d-pixelDiameter)
	for i in neighbours[i_pix]:
		d = np.sqrt((y[i_pix] - y[i]) ** 2)
		if m.isnan(d):
			continue
		else:
			gaps.append(d - pixelDiameter)


					#
	# if pixID[i_pix] < 51:
	# 	posX = 0.5
	# 	posY = 0.5
	# 	plt.text(pixel[0] + pixelDiameter * posX,
	# 			 pixel[1] + pixelDiameter * posY,
	# 			 pixID[i_pix],
	# 			 horizontalalignment='center',
	# 			 verticalalignment='center',
	# 			 fontsize=4)

#ax3.add_collection(PatchCollection(pixels, facecolor='none', edgecolor='r'))
# ax3.add_collection(PatchCollection(edgePixels, facecolor='brown', edgecolor='C1'))



xyPixPos = np.column_stack((y2,x2))
neighbours = find_neighbour_pixels(xyPixPos[:,0], xyPixPos[:,1], 1.458*pixelDiameter)
print(neighbours)
gaps2 =[]
# print(neighbours)
# exit()
pixels, edgePixels, offPixels = list(), list(), list()
pixels2, edgePixels2, offPixels2 = list(), list(), list()
edgePixelIndices = list()
for i_pix, pixel in enumerate(xyPixPos):
	square = mpatches.Rectangle((pixel[0], pixel[1]), width=pixelDiameter, height=pixelDiameter)
	if len(neighbours[i_pix]) > 3:
		edgePixelIndices.append(i_pix)
		edgePixels.append(square)
	# else:
	# if pixel[0] > 0:
	pixels.append(square)
	# else:
	# 	pixels2.append(square)
	for i in neighbours[i_pix]:
		d = np.sqrt((x2[i_pix] - x2[i]) ** 2)
		if m.isnan(d):
			continue
		else:
			gaps2.append(d-pixelDiameter)
	for i in neighbours[i_pix]:
		d = np.sqrt((y2[i_pix] - y2[i]) ** 2)
		if m.isnan(d):
			continue
		else:
			gaps2.append(d - pixelDiameter)
	if pixID[i_pix] < 51:
		plt.text(pixel[0] + pixelDiameter * 0.5,
				 pixel[1] + pixelDiameter * 0.5,
				 pixID[i_pix],
				 horizontalalignment='center',
				 verticalalignment='center',
				 fontsize=4)
		#
    #
	# if pixID[i_pix] < 51:
	# 	posX = 0.5
	# 	posY = 0.5
	# 	plt.text(pixel[0] + pixelDiameter * posX,
	# 			 pixel[1] + pixelDiameter * posY,
	# 			 pixID[i_pix],
	# 			 horizontalalignment='center',
	# 			 verticalalignment='center',
	# 			 fontsize=4)

ax3.add_collection(PatchCollection(pixels, facecolor='none', edgecolor='k'))

# plotAxesDef(plt, 0)

# ax3.add_collection(PatchCollection(edgePixels, facecolor='brown', edgecolor='C0'))
# print(gaps)
# ax3.add_collection(PatchCollection(edgePixels,
# 										 facecolor=mcolors.to_rgb('brown') + (0.5,),
# 										 edgecolor=mcolors.to_rgb('black') + (1,)))
# ax3.add_collection(PatchCollection(offPixels, facecolor='black', edgecolor='black'))

# legendObjects = [legH.pixelObject(), legH.edgePixelObject()]
# legendLabels = ['Pixel', 'Edge pixel']
# ax3.set_aspect('equal', 'datalim')
# plt.tight_layout()
xTitle = 'Horizontal scale [cm] (Y pix)'
yTitle = 'Vertical scale [cm] (X pix)'
plt.axis('equal')
plt.grid(True)

plt.gca().text(0.02, 0.02, 'For an observer facing the camera', transform=plt.gca().transAxes, color='black',
			   fontsize=12)

# plt.set_axisbelow(True)
plt.axis([min(xyPixPos[:, 0]), max(xyPixPos[:, 0]), min(xyPixPos[:, 1]) * 1.4, max(xyPixPos[:, 1]) * 1.4])
plt.xlabel(xTitle, fontsize=18, labelpad=0)
plt.ylabel(yTitle, fontsize=18, labelpad=0)
plt.title('Pixels layout in camera', fontsize=15, y=1.02)
plt.tick_params(axis='both', which='major', labelsize=15)



fig3 =plt.figure(3)
plt.hist(gaps, bins=200, label='MC', alpha = 0.5, color='C1')
plt.hist(gaps2, bins=200, label='Duncan', alpha = 0.5, color='C0')
plt.xlabel(r'$\Delta pos - dPix$')
plt.xlim(0,0.3)
plt.yscale('log')
plt.legend()



fig5=plt.figure(5)
ax5 = fig5.add_subplot(111)
#ax5.scatter(r,dz)
ax5.scatter(r2,dz2)
ax5.scatter(r3,dz3)




# diff = []
# for i in range(len(x)-1):
# 	# print(y[i], y[i+1])
# 	di= abs(x[i+1] -x[i]) -0.623
# 	print(x[i+1], x[i], 0.623, abs(x[i+1] -x[i]))
# 	if di <10:
# 		diff.append(di)
# diff2 = []
# for i in range(len(x2)-1):
# 	# print(y[i], y[i+1])
# 	di= abs(x2[i+1] -x2[i]) -0.623
# 	print(x2[i+1], x2[i], 0.623, abs(x2[i+1] -x[i]))
# 	if di <10:
# 		diff2.append(di)
#
# plt.hist(diff, bins=30,color='C1')
# plt.hist(diff2, bins=30, color='C2')


# # ax.scatter(x2,y2,z2,marker='x')
# # plt.text(x,y, s=np.asarray(pixn_list))
# # plt.text(0,0, 'test')
# nn = 8*8
#
# # mc_cam_one.write('MajorityTrigger * of ')
# # for i in range(nn):
# # 	if i % 2 != 0:
# # 		continue
# # 	mc_cam_one.write("%s[%s,%s,%s] " % (i,i+1,i+8, i+9) )
#
# # print(arr)
#
#
# R = np.linspace(0, 100*10, 100)
# h = (356.108-305.038)*10
# u = np.linspace(0,  2*np.pi, 100)
#
# x = np.outer(R, np.cos(u))
# y = np.outer(R, np.sin(u))
#
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x,y,h, alpha =0.3)
#
# # for i in range(0,10):
# R = np.random.uniform(0,100*10, 10)
# u = np.random.uniform(0,2*np.pi, 10)
# # xscd = np.outer(R, np.cos(u))
# xscd = R*np.cos(u)
# yscd = R*np.sin(u)
# # yscd = np.outer(R, np.sin(u))
#
# xr = np.random.uniform(-150, 150, 10)
# yr = np.random.uniform(-150, 150, 10)
# zr = 10*focal_plane(np.sqrt((xr/10)**2+(yr/10)**2))
# # print(xscd)
# # print(xr)
#
# for i in range(10):
# # 	print([xscd[i], xr[i]],[yscd[i], yr[i]], [h, 0])
# 	plt.plot([xscd[i], xr[i]],[yscd[i], yr[i]], [h, zr[i]], color='r')



plt.show()


