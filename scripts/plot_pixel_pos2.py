import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math as m
from scipy.spatial import cKDTree as KDTree
from matplotlib.collections import PatchCollection
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
import argparse
import astropy.units as u


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


def cart2sph(x, y, z):
    l = x ** 2 + y ** 2
    r = np.sqrt(l + z ** 2)
    elev = m.atan2(z, np.sqrt(l))
    az = m.atan2(y, x)
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


parser = argparse.ArgumentParser(description='plot pixel positions and gaps')
parser.add_argument('--mcam1', help='monte carlo file', default='/Users/armstrongt/Documents/CTA/MonteCarlo/MCVarification/DateForNewModel/GCT/camera/02_cameraConfigFile/camera_CHEC-S_GCT.dat')
parser.add_argument('--mcam2', help='monte carlo file',
                    default='/Users/armstrongt/Documents/CTA/MonteCarlo/MCVarification/DateForNewModel/GCT/camera/02_cameraConfigFile/checs_pixel_mapping_v2.dat')
parser.add_argument('--pixd1', help='pixel diameter', default=0.62, type=float)
parser.add_argument('--pixd2', help='pixel diameter', default=0.623, type=float)
args = parser.parse_args()

mc_cam1 = open(args.mcam1, 'r')
mc_cam2 = open(args.mcam2, 'r')


fig = plt.figure(1)
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ps1 = args.pixd1 / 2
pixelDiameter1 = args.pixd1
ps2 = args.pixd2 / 2
pixelDiameter2 = args.pixd2

pxid1 = []
x1 = []
y1 = []
z1 = []
z13 = []
arr1 = []
pixn1 = 0
pixn1_list = []
r1 = []
r1_error = []
dz1 = []
dz1_error = []
neighbours ={}

for line in mc_cam1:
    # continue
    if line.startswith('Pixel'):
        # if pixn > 4*8*8 + 6*8*8 and pixn < 4*8*8 + 7*8*8:
        # 	continue
        data = line.strip().split('\t')
        pxid1.append(int(data[1].split(' ')[0]))
        xi = float(data[2])
        yi = float(data[3])
        module = float(data[4].split(' ')[0])
        r_dist = np.sqrt(xi ** 2 + yi ** 2)
        zi = focal_plane(r_dist)
        r1.append(r_dist)
        dz1.append(zi)
        x1.append(xi)
        y1.append(yi)
        z1.append(zi)
        one_face_y = np.array([yi - ps1, yi - ps1, yi + ps1, yi + ps1,
                               yi - ps1])
        one_face_x = np.array([xi - ps1, xi + ps1, xi + ps1, xi - ps1, xi - ps1, ])
        one_face_z = np.array([zi, zi, zi, zi, zi])
        arr1.append([float(data[2]), float(data[3])])
        # mc_cam_one.write("%s\t%s 1\t%.3f\t%.3f\t%s\t%s\n" % (data[0],pixn,float(data[2])-2.6715625,float(data[3])-2.6715625,data[4],data[5]))
        pixn1_list.append(str(pixn1))
        # plt.text(float(data[2]),float(data[3]), s=str(pixn))
        pixn1 = pixn1 + 1
    elif line.startswith('MajorityTrigger'):
        continue
    else:
        # mc_cam_one.write(line)
        continue

mc_cam1 = open(args.mcam1, 'r')

gaps11 = []

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)

for line in mc_cam1:
    # continue
    if line.startswith('MajorityTrigger'):
        #fig=plt.figure(1)
        #ax1=fig.add_subplot(111)
        data = line.split(' ')
        # print(data)
        # exit()

        d0 = int(data[3].split('[')[0])
        d1 = int(data[3].split('[')[1].split(',')[0])
        d2 = int(data[3].split('[')[1].split(',')[1])
        d3 = int(data[3].split('[')[1].split(',')[2].split(']')[0])

        dray = [d0,d1,d2,d3]

        data2 = data[3:-1]

        if len(data2) < 9:
            continue
        for i in data2:
            #print(len(i))
            if len(i)==0:
                continue
            dd0 = int(i.split('[')[0])
            if abs(x1[d0] - x1[dd0]) > 0.62 and abs(y1[d0] - y1[dd0]) > 0.62:
                continue
            else:

                dd1 = int(i.split('[')[1].split(',')[0])
                dd2 = int(i.split('[')[1].split(',')[1])
                dd3 = int(i.split('[')[1].split(',')[2].split(']')[0])

                dray2 = [dd0,dd1,dd2,dd3]

                dist_tmp = []
                ids1 = []
                ids2 = []
                for di in dray:
                    for dj in dray2:
                        dist_tmp.append(np.sqrt((x1[di] - x1[dj]) ** 2 + (y1[di] - y1[dj]) ** 2))
                        ids1.append(di)
                        ids2.append(dj)
                mn = dist_tmp.index(min(dist_tmp))

                gaps11.append(np.sqrt((x1[ids1[mn]] - x1[ids2[mn]])**2 + (y1[ids1[mn]] - y1[ids2[mn]])**2 + (z1[ids1[mn]] - z1[ids2[mn]])**2) -pixelDiameter1)
            #     # plt.scatter([x1[d0], x1[dd0]],[y1[d0], y1[dd0]])
            # # #
            # #     s1pixels = []
            # #     s2pixels = []
            # #
            # #     s1quare = mpatches.Rectangle((float(x1[d0]) - pixelDiameter1 / 2, float(y1[d0]) - pixelDiameter1 / 2),
            # #                                  width=pixelDiameter1, height=pixelDiameter1)
            # #     s2quare = mpatches.Rectangle((float(x1[dd0]) - pixelDiameter1 / 2, float(y1[dd0]) - pixelDiameter1 / 2),
            # #                                  width=pixelDiameter1, height=pixelDiameter1)
            # #     s1pixels.append(s1quare)
            # #     s2pixels.append(s2quare)
            #
            #     # for i in neighbours[i_pix]:
            #     d = np.sqrt((x1[d0] - x1[dd0]) ** 2 + (y1[d0] - y1[dd0]) ** 2)
            #     if m.isnan(d):
            #         continue
            #     else:
            #         gaps11.append(d - 2*pixelDiameter1)
            #     # for i in neighbours[i_pix]:
            #     # d = np.sqrt((y1[d0] - y1[dd0]) ** 2)
            #     # if m.isnan(d):
            #     #     continue
            #     # else:
            #     #     gaps11.append(d - 2*pixelDiameter1)
            #     # plt.scatter(x1,y1)
            #     # ax1.add_collection(PatchCollection(s1pixels, facecolor='r', edgecolor='k'))
            #     # ax1.add_collection(PatchCollection(s2pixels, facecolor='none', edgecolor='k'))
            #     # plt.xlim(-0.15,0.15)
            #     # plt.ylim(-0.15,0.15)
            #     #
            #     # xTitle = 'Horizontal scale [cm]'
            #     # yTitle = 'Vertical scale [cm]'
            #     # plt.axis('equal')
            #     # plt.grid(True)
            #     # plt.xlabel(xTitle)
            #     # plt.ylabel(yTitle)
            #     # fig1.canvas.draw()
            #     # plt.pause(0.2)
            #     # # if plt.waitforbuttonpress():
            #     # #     continue
            #     # # plt.savefig('./animation/slice%04d.png' % counter)
            #     # fig.canvas.flush_events()
            #     # plt.cla()
            #
            #         # dd1 = int(i.split('[')[1].split(',')[0])
            #     # dd2 = int(i.split('[')[1].split(',')[1])
            #     # dd3 = int(i.split('[')[1].split(',')[2].split(']')[0])
            #
            #







pxid2 = []
x2 = []
y2 = []
z2 = []
z23 = []
arr2 = []
pixn2 = 0
pixn2_list = []
r2 = []
r2_error = []
dz2 = []
dz2_error = []

for line in mc_cam2:
    # continue
    if line.startswith('Pixel'):
        # if pixn > 4*8*8 + 6*8*8 and pixn < 4*8*8 + 7*8*8:
        # 	continue
        data = line.strip().split('\t')
        # print(data)
        # exit()
        pxid2.append(int(data[1].split(' ')[0]))
        xi = float(data[3])
        yi = float(data[4])
        module = float(data[5])
        r_dist = np.sqrt(xi ** 2 + yi ** 2)
        zi = focal_plane(r_dist)
        r2.append(r_dist)
        dz2.append(zi)
        x2.append(xi)
        y2.append(yi)
        z2.append(zi)
        one_face_y = np.array([yi - ps2, yi - ps2, yi + ps2, yi + ps2,
                               yi - ps2])
        one_face_x = np.array([xi - ps2, xi + ps2, xi + ps2, xi - ps2, xi - ps2, ])
        one_face_z = np.array([zi, zi, zi, zi, zi])
        arr2.append([float(data[2]), float(data[3])])
        # mc_cam_one.write("%s\t%s 1\t%.3f\t%.3f\t%s\t%s\n" % (data[0],pixn,float(data[2])-2.6715625,float(data[3])-2.6715625,data[4],data[5]))
        pixn2_list.append(str(pixn2))
        # plt.text(float(data[2]),float(data[3]), s=str(pixn))
        pixn2 = pixn2 + 1
    elif line.startswith('MajorityTrigger'):
        continue
    else:
        # mc_cam_one.write(line)
        continue

# print(x2)
# print(y2)
# exit()

mc_cam2 = open(args.mcam2, 'r')

gaps22 = []

for line in mc_cam2:
    # continue
    if line.startswith('MajorityTrigger'):
        # fig=plt.figure(1)
        # ax1=fig.add_subplot(111)
        data = line.split(' ')
        # print(data)
        # exit()
        d0 = int(data[3].split('[')[0])
        d1 = int(data[3].split('[')[1].split(',')[0])
        d2 = int(data[3].split('[')[1].split(',')[1])
        d3 = int(data[3].split('[')[1].split(',')[2].split(']')[0])
        data2 = data[3:-1]

        dray = [d0,d1,d2,d3]


        if len(data2) < 9:
            continue
        for i in data2:
            # print(len(i))
            if len(i) == 0:
                continue
            dd0 = int(i.split('[')[0])
            if abs(x2[d0] - x2[dd0]) > 0.62 and abs(y2[d0] - y2[dd0]) > 0.62:
                continue
            else:

                dd1 = int(i.split('[')[1].split(',')[0])
                dd2 = int(i.split('[')[1].split(',')[1])
                dd3 = int(i.split('[')[1].split(',')[2].split(']')[0])

                dray2 = [dd0, dd1, dd2, dd3]

                # dist_tmp = []
                # for di in dray:
                #     for dj in dray2:
                #         dist_tmp.append(np.sqrt((x2[di] - x2[dj]) ** 2 + (y2[di] - y2[dj]) ** 2))
                # gaps22.append(min(dist_tmp) - pixelDiameter2)

                dist_tmp = []
                ids1 = []
                ids2 = []
                for di in dray:
                    for dj in dray2:
                        dist_tmp.append(np.sqrt((x2[di] - x2[dj]) ** 2 + (y2[di] - y2[dj]) ** 2))
                        ids1.append(di)
                        ids2.append(dj)
                mn = dist_tmp.index(min(dist_tmp))

                gaps22.append(np.sqrt((x2[ids1[mn]] - x2[ids2[mn]]) ** 2 + (y2[ids1[mn]] - y2[ids2[mn]]) ** 2 + (z2[ids1[mn]] - z2[ids2[mn]]) ** 2) - pixelDiameter2)

                #
                # # s1pixels = []
                # # s2pixels = []
                # #
                # # s1quare = mpatches.Rectangle((float(x2[d0]) - pixelDiameter2 / 2, float(y2[d0]) - pixelDiameter2 / 2),
                # #                              width=pixelDiameter2, height=pixelDiameter2)
                # # s2quare = mpatches.Rectangle((float(x2[dd0]) - pixelDiameter2 / 2, float(y2[dd0]) - pixelDiameter2 / 2),
                # #                              width=pixelDiameter2, height=pixelDiameter2)
                # # s1pixels.append(s1quare)
                # # s2pixels.append(s2quare)
                #
                #
                # # for i in neighbours[i_pix]:
                # d1 = np.sqrt((x2[d0] - x2[dd0]) ** 2 + (y2[d0] - y2[dd0]) ** 2)
                # if m.isnan(d1):
                #     continue
                # else:
                #     gaps22.append(d1 - 2 * pixelDiameter2)
                # # for i in neighbours[i_pix]:
                # # d2 = np.sqrt((y2[d0] - y2[dd0]) ** 2)
                # # if m.isnan(d2):
                # #     continue
                # # else:
                # #     gaps22.append(d2 - 2 * pixelDiameter2)
                # # plt.scatter(x2,y2)
                # # if d1 > 0.245:
                # #     ax1.add_collection(PatchCollection(s1pixels, facecolor='r', edgecolor='k'))
                # #     ax1.add_collection(PatchCollection(s2pixels, facecolor='r', edgecolor='k'))
                # # else:
                # #     ax1.add_collection(PatchCollection(s1pixels, facecolor='g', edgecolor='k'))
                # #     ax1.add_collection(PatchCollection(s2pixels, facecolor='y', edgecolor='k'))
                # #
                # # # plt.xlim(-0.15,0.15)
                # # # plt.ylim(-0.15,0.15)
                # #
                # # xTitle = 'Horizontal scale [cm]'
                # # yTitle = 'Vertical scale [cm]'
                # # plt.axis('equal')
                # # plt.grid(True)
                # # plt.xlabel(xTitle)
                # # plt.ylabel(yTitle)
                # # fig1.canvas.draw()
                # # plt.pause(0.1)
                # # # if plt.waitforbuttonpress():
                # # #     continue
                # # # plt.savefig('./animation/slice%04d.png' % counter)
                # # fig.canvas.flush_events()
                # # plt.cla()


# x2 = []
# y2 = []
# z2 = []
# module2 = []
# # arr=[]
# # pixn=0
# # pixn_list=[]
# n1 = 0
# n2 = 1
# # cmap = plt.cm.viridis
# pixID = []
# r2 = []
# dz2 = []
#
# for line in cad_cam2:
#     # print(line)
#     if line.startswith('Pixel'):
#         # if pixn > 4*8*8 + 6*8*8 and pixn < 4*8*8 + 7*8*8:
#         # 	continue
#         data = line.strip().split('\t')
#         # print(data)
#         # exit()
#         # if float(data[2]) > 0 and float(data[2]) < 5 and float(data[3]) >0 and float(data[3]) < 5:
#         xi = float(data[3])
#         yi = float(data[4])
#         module = data[5]
#         module2.append(int(module))
#         r_dist = np.sqrt(xi ** 2 + yi ** 2)
#         zi = focal_plane(r_dist)
#         r2.append(r_dist)
#         dz2.append(zi)
#         # ax.scatter(xi, yi, zi, marker='x', color='C%s' % int(module))
#         x2.append(xi)
#         y2.append(yi)
#         # z2.append(zi)
#         # one_face_y = np.array([yi - ps, yi - ps, yi + ps, yi + ps,
#         # 					   yi - ps])
#         # one_face_x = np.array([xi - ps, xi + ps, xi + ps, xi - ps, xi - ps, ])
#         # one_face_z = np.array([zi, zi, zi, zi, zi])
#         # one_face = np.array([[xi-ps, yi-ps, zi], [xi+ps, yi-ps, zi], [xi+ps, yi+ps, zi], [xi-ps, yi+ps, zi], [xi-ps, yi-ps, zi]])
#         # plt.Polygon(one_face)
#         # arr.append([float(data[2]),float(data[3])])
#         # mc_cam_one.write("%s\t%s 1\t%.3f\t%.3f\t%s\t%s\n" % (data[0],pixn,float(data[2])-2.6715625,float(data[3])-2.6715625,data[4],data[5]))
#         pixn_list.append(str(pixn))
#         # plt.text(float(data[2]),float(data[3]), s=str(pixn))
#         pixn = pixn + 1
#     elif line.startswith('MajorityTrigger'):
#         continue
#     else:
#         # mc_cam_one.write(line)
#         continue
#
# r3 = []
# dz3 = []
#
# for n, line in enumerate(cad_cam):
#     pixID.append(n)
#     n1 += 1
#     if n1 == 64:
#         print(n2)
#         n2 += 1
#         n1 = 0
#     data = line.strip().split('\t')
#     xi = float(data[0]) / 10
#     yi = float(data[1]) / 10
#     zi = float(data[2]) / 10
#     r3.append(np.sqrt((float(data[0]) / 10) ** 2 + (float(data[1]) / 10) ** 2))
#     dz3.append(zi)
#     one_face_y = np.array([yi - ps, yi - ps, yi + ps, yi + ps,
#                            yi - ps])
#     one_face_x = np.array([xi - ps, xi + ps, xi + ps, xi - ps, xi - ps, ])
#     one_face_z = np.array([zi, zi, zi, zi, zi])
#     zi3 = focal_plane(np.sqrt(xi ** 2 + yi ** 2))
#     z2.append(zi)
#     z3.append(zi3)
#

# start = 4 * 8 * 8 + 6 * 8 * 8 + 6 * 8 * 8 + 16
# end = 4 * 8 * 8 + 6 * 8 * 8 + 7 * 8 * 8 - 16
# ax.scatter(x2[start:end], z3[start:end], marker='s', label='focal plane parameter')
# ax.scatter(x2[start:end], z2[start:end], marker='x', label='from Duncan')
# ax.scatter(x[start:end], z[start:end], marker='.', label='Old mc values', color='r')
# ax.legend()
#
# for i in range(len(z3[start:end])):
#     ax2.scatter(x2[start:end][i], z3[start:end][i] - z2[start:end][i], marker='x', color='k')

# ax2.set_xlabel('x [cm]')
# ax2.set_ylabel(r'$z_{meas}- z_{focal}$')
# ax.set_ylabel('z [cm]')
# plt.legend()

fig2 = plt.figure(2)
ax3 = fig2.add_subplot(111)
# pixelDiameter = 0.62


xyPixPos = np.column_stack((x1, y1))
neighbours = find_neighbour_pixels(xyPixPos[:, 0], xyPixPos[:, 1], 1.4 * pixelDiameter1)
gaps = []
pixels, edgePixels, offPixels = list(), list(), list()
pixels2, edgePixels2, offPixels2 = list(), list(), list()
edgePixelIndices = list()
for i_pix, pixel in enumerate(xyPixPos):
    square = mpatches.Rectangle((pixel[0], pixel[1]), width=pixelDiameter1, height=pixelDiameter1)
    # if len(neighbours[i_pix]) == 4:
    # 	edgePixelIndices.append(i_pix)
    # 	edgePixels.append(square)
    # else:
    # if pixel[0] < 0:
    pixels.append(square)
    # else:
    # 	pixels2.append(square)
    for i in neighbours[i_pix]:
        d = np.sqrt((x1[i_pix] - x1[i]) ** 2)
        if m.isnan(d):
            continue
        else:
            gaps.append(d - pixelDiameter1)
    for i in neighbours[i_pix]:
        d = np.sqrt((y1[i_pix] - y1[i]) ** 2)
        if m.isnan(d):
            continue
        else:
            gaps.append(d - pixelDiameter1)


xyPixPos = np.column_stack((y2, x2))
neighbours = find_neighbour_pixels(xyPixPos[:, 0], xyPixPos[:, 1], 1 * pixelDiameter2)
# print(neighbours)
gaps2 = []
# print(neighbours)
# exit()
pixels, edgePixels, offPixels = list(), list(), list()
pixels2, edgePixels2, offPixels2 = list(), list(), list()
edgePixelIndices = list()
for i_pix, pixel in enumerate(xyPixPos):
    square = mpatches.Rectangle((pixel[0], pixel[1]), width=pixelDiameter2, height=pixelDiameter2)
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
            gaps2.append(d - pixelDiameter2)
    for i in neighbours[i_pix]:
        d = np.sqrt((y2[i_pix] - y2[i]) ** 2)
        if m.isnan(d):
            continue
        else:
            gaps2.append(d - pixelDiameter2)
    # if pixID[i_pix] < 51:
    #     plt.text(pixel[0] + pixelDiameter * 0.5,
    #              pixel[1] + pixelDiameter * 0.5,
    #              pixID[i_pix],
    #              horizontalalignment='center',
    #              verticalalignment='center',
    #              fontsize=4)


# print(pixels)
# print(module2)

from matplotlib import cm

# ax3.add_collection(PatchCollection(pixels, facecolor=cm.Set1(module2), edgecolor='k'))

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
# plt.axis('equal')
# plt.grid(True)
#
# plt.gca().text(0.02, 0.02, 'For an observer facing the camera', transform=plt.gca().transAxes, color='black',
#                fontsize=12)

# plt.set_axisbelow(True)
# plt.axis([min(xyPixPos[:, 0]), max(xyPixPos[:, 0]), min(xyPixPos[:, 1]) * 1.4, max(xyPixPos[:, 1]) * 1.4])
# plt.xlabel(xTitle, fontsize=18, labelpad=0)
# plt.ylabel(yTitle, fontsize=18, labelpad=0)
# plt.title('Pixels layout in camera', fontsize=15, y=1.02)
# plt.tick_params(axis='both', which='major', labelsize=15)

fig3 = plt.figure(3)
plt.hist(gaps11, bins=200, label='Old', alpha=0.5, color='C1')
plt.hist(gaps22, bins=200, label='New', alpha=0.5, color='C0')
plt.xlabel(r'$\Delta pos - dPix$')
plt.xlim(0, 0.3)
plt.yscale('log')
plt.legend()

# fig5 = plt.figure(5)
# ax5 = fig5.add_subplot(111)
# # ax5.scatter(r,dz)
# ax5.scatter(r2, dz2)
# ax5.scatter(r3, dz3)




plt.show()
