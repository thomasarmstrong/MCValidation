import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import patches
import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math as m
from ctapipe.io.hessio import hessio_event_source
from ctapipe.image import tailcuts_clean
from ctapipe.calib import pedestals
from ctapipe.calib import CameraCalibrator, CameraDL1Calibrator
from ctapipe.analysis.camera.chargeresolution import ChargeResolutionCalculator
from scipy.interpolate import interp1d
import argparse
import astropy.units as u
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Circle, PathPatch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import art3d
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




PIXEL_EPSILON = 0.0002

def focal_plane(x):
    x=100*x
    p2 = -5.0e-3
    p4 = -1.25e-7
    p6 = -6.25e-12
    p8 = -3.90625e-16
    p10 = -2.734375e-20
    # return 0
    return (p2 *x**2 + p4 *x**4 + p6 *x**6 + p8 *x **8 +  p10 *x**10)/100.0

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


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


class Geometry:
    def __init__(self):
        # TODO Make this more flexible for lab setup
        self.fig = plt.figure(1,figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)
        self.required_pe = 100
        self.camera = "CHEC"
        self.camera_curve_center_x = 1
        self.camera_curve_center_y = 0
        self.camera_curve_radius = 1
        self.total_scale = 0
        self.pixel_mask = 0
        self.ang_dist_file_string = "ang_dist_2.dat"
        self.geom = None
        self.npix = 0
        self.pix_size = None
        self.fiducial_center = None
        self.lightsource_distance = None
        self.fiducial_radius = 0.2
        self.lightsource_distance = 1
        self.lightsource_x = 0
        self.lightsource_y = 0
        self.ang_dist_file = None
        self.lightsource_angle = None
        self.lightsource_sa = None
        self.fiducial_sa = None
        self.fiducial_percent = None
        self.pix_sa = None
        self.pix_percent = None
        self.lamda = None
        self.pde = None
        self.photons_pixel = None
        self.photons_ls = None
        self.photons_fiducial = None

    def setup_geom(self):

        self.geom = CameraGeometry.from_name(self.camera)
        self.npix = len(self.geom.pix_id)
        self.total_scale = np.ones(self.npix)
        self.pixel_mask = np.ones(self.npix)
        # pixfile = np.loadtxt("/Users/armstrongt/Workspace/CTA/MCValidation/src/MCValidation/configs/checs_pixel_mapping_fixedgainvar_justpix.dat", unpack=True)
        # self.geom.pix_x = pixfile[2]/100*u.m
        # self.geom.pix_y = pixfile[3]/100*u.m
        # self.geom.pix_id = pixfile[0]
        # self.geom.pix_size = 0.623/100*u.m
        # print(self.geom)

        self.pix_size = np.sqrt(self.geom.pix_area[0]).value
        self.fiducial_center = self.camera_curve_center_x + self.camera_curve_radius * np.cos(np.pi) # This is a little confusing, is it going to be set up to change the position of the lightsource
        self.ang_dist_file = np.loadtxt(self.ang_dist_file_string, unpack=True)
        theta_f = np.pi / 2 - np.arccos(self.fiducial_radius / self.lightsource_distance)
        # theta_ls = theta_f#self.lightsource_angle * np.pi / 180
        theta_p = np.arctan(self.pix_size / (2 * self.lightsource_distance))
        # self.lightsource_angle = theta_ls * 180 / np.pi
        self.lightsource_angle = max(self.ang_dist_file[0])
        theta_ls = self.lightsource_angle * np.pi / 180
        self.lightsource_sa = 2 * np.pi * (1 - np.cos(theta_ls))
        self.fiducial_sa = 2 * np.pi * (1 - np.cos(theta_f))
        self.fiducial_percent = self.fiducial_sa / self.lightsource_sa
        if self.fiducial_percent > 1: self.fiducial_percent = 1
        self.pix_sa = 2 * np.pi * (1 - np.cos(theta_p))
        self.pix_percent = self.pix_sa / self.lightsource_sa
        if self.pix_percent > 1: self.pix_percent = 1

        # photons = self.set_illumination(self.required_pe)
        # print(photons)
        # print(self.lightsource_angle)

    def set_illumination(self, lambda_, tote_trans):
        self.lamda = lambda_
        self.pde = tote_trans
        self.photons_pixel = self.lamda / self.pde
        self.photons_ls = self.photons_pixel / self.pix_percent
        self.photons_fiducial = self.photons_ls * self.fiducial_percent
        return self.photons_ls

    def plot_camera_circle(self):
        x = self.camera_curve_center_x
        y = self.camera_curve_center_y
        r = self.camera_curve_radius
        self.ax.add_artist(plt.Circle((x, y), r, color='b', fill=False))

    def plot_pixels(self):
        pix_x = np.unique(np.reshape(self.geom.pix_x, (32, 64))[10:16])
        pix_cy = np.reshape(pix_x, (6, 8))
        module_centers = pix_cy.mean(1)
        x0 = self.camera_curve_center_x
        y0 = self.camera_curve_center_y
        r = self.camera_curve_radius

        mod_y = module_centers
        theta = np.arcsin((mod_y - y0)/r) + np.pi
        x = x0 + r * np.cos(theta)
        mr = (mod_y - y0) / (x - x0)
        mt = -1 / mr
        pix_cx = ((pix_cy - mod_y[:, None]) / mt[:, None]) + x[:, None]
        pix_x1 = pix_cx - (self.pix_size / 2) / np.sqrt(1 + mt[:, None] ** 2)
        pix_x2 = pix_cx + (self.pix_size / 2) / np.sqrt(1 + mt[:, None] ** 2)
        pix_y1 = (pix_x1 - x[:, None]) * mt[:, None] + mod_y[:, None]
        pix_y2 = (pix_x2 - x[:, None]) * mt[:, None] + mod_y[:, None]
        pix_x = np.vstack([pix_x1.ravel(), pix_x2.ravel()])
        pix_y = np.vstack([pix_y1.ravel(), pix_y2.ravel()])
        self.ax.plot(pix_x, pix_y, color='r')

    def plot_focal_plane(self):
        self.ax.axvline(self.fiducial_center)

    def plot_fiducial_sphere(self):
        x0 = self.fiducial_center
        y0 = self.camera_curve_center_y
        r = self.fiducial_radius
        self.ax.add_artist(plt.Circle((x0, y0), r, color='b', fill=False))

    def plot_lightsource(self):
        x = self.fiducial_center - self.lightsource_distance
        y = self.camera_curve_center_y
        self.ax.plot(x, y, 'x')
        angle = self.lightsource_angle
        wedge = patches.Wedge((x, y), 2*self.lightsource_distance, -angle, angle, alpha=0.5)
        self.ax.add_patch(wedge)

    # def plot_values(self):
    #     text = "Lightsource = {} photons \n" \
    #            "Fiducial = {} photons \n" \
    #            "Pixel = {} photons \n" \
    #            "Pixel = {} p.e.".format(self.photons_ls, self.photons_fiducial,
    #                                     self.photons_pixel, self.lamda)
    #     self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes)

    def plot_values(self):
        text = "Pixel = %s p.e\n" \
               "Pixel = %.2f photons \n" \
               "Lightsource = %.2f photons." % (self.lamda,
                                        self.photons_pixel, self.photons_ls)
        self.ax.text(0.1, 0.9, text, transform=self.ax.transAxes)


    def plot_values3d(self):
        text = "Lightsource = {} photons \n" \
               "Fiducial = {} photons \n" \
               "Pixel = {} photons \n" \
               "Pixel = {} p.e.".format(self.photons_ls, self.photons_fiducial,
                                        self.photons_pixel, self.lamda)
        self.ax.text(0.5, 0.5, 'test')

    def plot(self):
        self.plot_camera_circle()
        self.plot_pixels()
        self.plot_focal_plane()
        self.plot_fiducial_sphere()
        self.plot_lightsource()
        self.plot_values()
        plt.ylabel('Y')
        plt.xlabel('Z')
        # self.ax.set_xlim([-1, 0.05])
        # self.ax.set_ylim([-0.4, 0.4])

    def get_pixel_angscale(self, xc, yc):
        r, e, a = cart2sph(xc-self.lightsource_x, yc-self.lightsource_y, focal_plane(
            np.sqrt((xc) ** 2 + (yc) ** 2)) - self.lightsource_distance)
        index = np.searchsorted(self.ang_dist_file[0], e*180/np.pi+90)
        if index==0: index=0
        else: index = index -1
        return self.ang_dist_file[1][index]

    def get_2dverts(self, xc, yc):
        verts = []
        verts.append((xc - self.pix_size / 2 -  PIXEL_EPSILON, yc - self.pix_size / 2 - PIXEL_EPSILON))
        verts.append((xc + self.pix_size / 2 + PIXEL_EPSILON, yc - self.pix_size / 2 - PIXEL_EPSILON))
        verts.append((xc + self.pix_size / 2 + PIXEL_EPSILON, yc + self.pix_size / 2 + PIXEL_EPSILON))
        verts.append((xc - self.pix_size / 2 - PIXEL_EPSILON, yc + self.pix_size / 2 + PIXEL_EPSILON))
        return verts

    def get_3dverts(self, xc, yc):
        # if xc>0 and yc>0:
        verts = []
        verts.append((focal_plane(np.sqrt((xc) ** 2 + (yc) ** 2)),
                       xc - self.pix_size / 2, yc - self.pix_size / 2))
        verts.append((focal_plane(np.sqrt((xc) ** 2 + (yc) ** 2)),
                       xc + self.pix_size / 2, yc - self.pix_size / 2))
        verts.append((focal_plane(np.sqrt((xc) ** 2 + (yc) ** 2)),
                       xc + self.pix_size / 2, yc + self.pix_size / 2))
        verts.append((focal_plane(np.sqrt((xc) ** 2 + (yc) ** 2)),
                       xc - self.pix_size / 2, yc + self.pix_size / 2))
        return verts

    def get_pixel_scaledarea(self, xc, yc):
        dist_to_pix = np.sqrt((xc-self.lightsource_x) ** 2 + (yc-self.lightsource_y) ** 2 + (self.lightsource_distance - focal_plane(np.sqrt(xc ** 2 + yc ** 2))) ** 2)

        r1, e1, a1 = cart2sph(xc - self.pix_size / 2 - self.lightsource_x, yc - self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc - self.pix_size / 2) ** 2 + (yc - self.pix_size / 2) ** 2)) - self.lightsource_distance)
        r2, e2, a2 = cart2sph(xc + self.pix_size / 2 - self.lightsource_x, yc - self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc + self.pix_size / 2) ** 2 + (yc - self.pix_size / 2) ** 2)) - self.lightsource_distance)
        r3, e3, a3 = cart2sph(xc + self.pix_size / 2 - self.lightsource_x, yc + self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc + self.pix_size / 2) ** 2 + (yc + self.pix_size / 2) ** 2)) - self.lightsource_distance)
        r4, e4, a4 = cart2sph(xc - self.pix_size / 2 - self.lightsource_x, yc + self.pix_size / 2 - self.lightsource_y, focal_plane(
            np.sqrt((xc - self.pix_size / 2) ** 2 + (yc + self.pix_size / 2) ** 2)) - self.lightsource_distance)

        x1, y1, z1 = sph2cart(dist_to_pix, e1, a1)
        x2, y2, z2 = sph2cart(dist_to_pix, e2, a2)
        x3, y3, z3 = sph2cart(dist_to_pix, e3, a3)
        x4, y4, z4 = sph2cart(dist_to_pix, e4, a4)

        area = PolyArea([x1, x2, x3, x4], [y1, y2, y3, y4])
        return area

    def plot3dcamera(self, verts, scale=None):
        fig = plt.figure()
        ax1 = fig.add_subplot(111,projection='3d')
        poly3d = Poly3DCollection(verts)

        for i in range(2048):
            x=self.geom.pix_x[i].value
            y=self.geom.pix_y[i].value
            if x >0.1 and y >0:
        #
                z=focal_plane(np.sqrt((x) ** 2 + (y) ** 2))
        #
        #         verts.append((x - g.pix_size / 2, y - g.pix_size / 2, z))
        #         verts.append((x + g.pix_size / 2, y - g.pix_size / 2, z))
        #         verts.append((x + g.pix_size / 2, y + g.pix_size / 2, z))
        #         verts.append((x - g.pix_size / 2, y + g.pix_size / 2, z))
        #
        #         plt.scatter(x,y,z, color='k', marker=',')
                ax1.quiver(z, x, y , 1, 0, 0, length=0.01)

        if not scale:
            poly3d.set_array(np.array(self.total_scale)/max(self.total_scale))
        else:
            poly3d.set_array(np.array(scale) / max(scale))
        ax1.add_collection3d(poly3d)




        theta = np.linspace(0, 2 * np.pi, 90)
        r = -np.arange(0, self.fiducial_radius, 0.01)
        T, R = np.meshgrid(theta, r)
        X = R * np.cos(T)
        Y = R * np.sin(T)
        Z = -np.sqrt(X ** 2 + Y ** 2)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = self.fiducial_radius* np.cos(u) * np.sin(v)
        y = self.fiducial_radius*np.sin(u) * np.sin(v)
        z = self.fiducial_radius* np.cos(v)
        # ax1.plot_wireframe(self.lightsource_distance+Z*(self.lightsource_distance/0.2),X+ self.lightsource_x,  Y+ self.lightsource_y, alpha = 0.1, color='C0')
        # ax1.plot_wireframe(x, y, z, color="C0", alpha = 0.1)

        lim = max(self.geom.pix_x).value

        ax1.set_ylim3d(0.1, lim)
        ax1.set_xlim3d(-0.02, 0.0)
        ax1.set_zlim3d(0, 0.1)
        ax1.set_xlabel('Z')
        ax1.set_ylabel('X')
        ax1.set_zlabel('Y')
        fig.colorbar(poly3d, ax=ax1)

        colors = ['0.6', '0.5', '0.2', 'c', 'm', 'y']
        # for i, (z, zdir) in enumerate(product([self.lightsource_distance, self.lightsource_distance+0.3], ['x', 'y', 'z'])):
        # side = Rectangle((self.lightsource_distance, -0.01+self.lightsource_x), 0.1, 0.02, facecolor=colors[0])
        # ax1.add_patch(side)
        # pathpatch_2d_to_3d(side, z=0.01+self.lightsource_x, zdir='z')
        #
        # side = Rectangle((self.lightsource_distance, -0.01+self.lightsource_x), 0.1, 0.02, facecolor=colors[0])
        # ax1.add_patch(side)
        # pathpatch_2d_to_3d(side, z=-0.01+self.lightsource_x, zdir='z')
        #
        # side = Rectangle((self.lightsource_distance, -0.01+ self.lightsource_y), 0.1, 0.02, facecolor=colors[1])
        # ax1.add_patch(side)
        # pathpatch_2d_to_3d(side, z=0.01+self.lightsource_x, zdir='y')
        #
        # side = Rectangle((self.lightsource_distance, -0.01 + self.lightsource_y), 0.1, 0.02, facecolor=colors[1])
        # ax1.add_patch(side)
        # pathpatch_2d_to_3d(side, z=-0.01+self.lightsource_x, zdir='y')
        #
        # side = Rectangle((-0.01+self.lightsource_x, -0.01+ self.lightsource_y), 0.02, 0.02, facecolor=colors[1])
        # ax1.add_patch(side)
        # pathpatch_2d_to_3d(side, z=self.lightsource_distance, zdir='x')
        #
        # side = Rectangle((-0.01 + self.lightsource_x, -0.01 + self.lightsource_y), 0.02, 0.02, facecolor=colors[1])
        # ax1.add_patch(side)
        # pathpatch_2d_to_3d(side, z=self.lightsource_distance+0.1, zdir='x')
        logfile = open('lines3.txt', 'r')

        d = 0
        z0 = []
        x1 = []
        y1 = []
        z1 = []
        x2 = 0
        y2 = 0
        z2 = 0
        x3 = []
        y3 = []
        z3 = []
        dist = []
        angleTh = []
        angle = []
        nangleTh = []
        nangle = []

        for line in logfile:
            if line.startswith('Starting position w.r.t. camera front:'):
                data = line.split(' ')[5].split('=')[1].split(',')
                # x1 = float(data[0])
                # y1 = float(data[1])
                # z1 = float(data[2])
                x1.append(float(data[0]))
                y1.append(float(data[1]))
                z1.append(float(data[2]))
            if line.startswith('Distance of emission point'):
                # z0 = float(line.split(' ')[5])
                z0.append(float(line.split(' ')[5]))
            if line.startswith('Return: '):
                data = line.split(' ')
                data2 = [x for x in data if x]
                # print(data2)
                # exit()
                x2 = float(data2[1])
                y2 = float(data2[2])
                z2 = float(data2[3])

                # ax0.plot([0,x1], [0,y1], [z0,z1], color='r')
                # ax0.plot([x1,x2], [y1,y2], [z1,z2], color='r')
                # ax0.scatter(x2,y2,z2, color='k', marker='.')
                # plt.pause(0.001)
            if line.startswith('Camera 1 (custom):'):
                if 'ipix=-1' in line:
                    # x3.append(0)
                    # y3.append(0)
                    # z3.append(0)
                    # dist.append(0)
                    # angleTh.append(0)
                    # nangleTh.append(0)
                    continue
                else:
                    z3.append(x2/100)
                    y3.append(y2/100)
                    x3.append(z2/100)
                    d = np.sqrt(x2 ** 2 + y2 ** 2)
                    dist.append(d)
                    angleTh.append(
                        90 + np.rad2deg(np.arcsin(d / 100) - np.arctan((155.2 + np.sqrt(100 ** 2 + d ** 2)) / d)))
                    nangleTh.append(np.rad2deg(np.arcsin(d / 100)))
            if line.startswith('  Angle to normal to focal surface'):
                angle.append(float(line.split(' ')[8]))

            if line.startswith('  Angle to normal to focal plane'):
                nangle.append(float(line.split(' ')[8]))

            if len(x3) > 10000: break
            # angleTh.append(90-np.rad2deg(np.arcsin(d / 100) - np.arctan((155.2 + np.sqrt(100 ** 2 + d ** 2)) / d)))
        # ax0.scatter(x3,y3, color='k', marker='.')
        # print(len(angleTh), len(angle))
        # ax1.scatter(angle, angleTh, color='C0')
        # ax1.scatter(nangle, nangleTh, color='C1')
        # plt.ylabel('Theoretical Angle')
        # plt.xlabel('simtel value')
        # print(x3)
        # ax1.scatter(x3 , y3 , z3 , color='k', marker='.')

        self.plot_values3d()

    def plot2dscales(self, verts, area2, scale2):
        fig = plt.figure(10)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        poly2d1 = PolyCollection(verts)
        poly2d1.set_array(np.array(area2)/max(area2))

        poly2d2 = PolyCollection(verts)
        poly2d2.set_array(np.array(scale2)/max(scale2))

        poly2d3 = PolyCollection(verts)
        poly2d3.set_array(np.array(self.total_scale)/max(self.total_scale))

        poly2d4 = PolyCollection(verts)
        poly2d4.set_array(self.required_pe * np.array(self.total_scale)/max(self.total_scale))

        lim = max(self.geom.pix_x).value

        ax1.add_collection(poly2d1)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Geometry Scale')

        ax2.add_collection(poly2d2)
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Angular Pulse Scale')

        ax3.add_collection(poly2d3)
        ax3.set_xlim(-lim, lim)
        ax3.set_ylim(-lim, lim)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Total Scale')

        ax4.add_collection(poly2d4)
        ax4.set_xlim(-lim, lim)
        ax4.set_ylim(-lim, lim)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('p.e. per pixel')

        fig.colorbar(poly2d1, ax=ax1)
        fig.colorbar(poly2d2, ax=ax2)
        fig.colorbar(poly2d3, ax=ax3)
        fig.colorbar(poly2d4, ax=ax4)

        fig20 = plt.figure(20)
        ax20 = fig20.add_subplot(111)
        dist = np.sqrt(self.geom.pix_x**2+ self.geom.pix_y**2)
        plt.scatter(dist, self.required_pe * np.array(self.total_scale)/max(self.total_scale))

        # # print(dist)
        #
        # tmpout=open('tmp_true_pe_dist.txt', 'w')
        # arr = np.array(self.total_scale)/max(self.total_scale)
        # for i in range(len(dist)):
        #     tmpout.write('%s\t%s\n' %(dist[i].value, arr[i]))
        # # print(self.required_pe * np.array(self.total_scale)/max(self.total_scale))

        ax20.set_xlabel('distance')
        ax20.set_ylabel('number of p.e.')

    def getscale(self, plot=False):

        area2 = []
        scale2 = []
        verts2d=[]
        verts3d=[]
        for ipix in range(self.npix):
            xc = self.geom.pix_x[ipix].value
            yc = self.geom.pix_y[ipix].value

            if xc>0.1 and yc>0:
                verts2d.append(self.get_2dverts(xc, yc))
                verts3d.append(self.get_3dverts(xc, yc))

                if self.pixel_mask[ipix] == 0:
                    self.total_scale[ipix] = 0
                    scale2.append(0)
                    area2.append(0)
                    continue

                area = self.get_pixel_scaledarea(xc, yc)
                scale = self.get_pixel_angscale(xc, yc)
                scale2.append(scale)
                area2.append(area)
                self.total_scale[ipix] = area*scale

        if plot:
            self.plot3dcamera(verts3d)
            # self.plot2dscales(verts2d, area2, scale2)

        return 0

# plt.ion()

def rotation_matrix(d):
    sin_angle = np.linalg.norm(d)
    if sin_angle == 0:return np.identity(3)
    d /= sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def plot_vector(fig, orig, v, color='blue'):
   ax = fig.gca(projection='3d')
   orig = np.array(orig); v=np.array(v)
   ax.quiver(orig[0], orig[1], orig[2], v[0], v[1], v[2],color=color)
   ax.set_xlim(0,10);ax.set_ylim(0,10);ax.set_zlim(0,10)
   ax = fig.gca(projection='3d')
   return fig

def pathpatch_2d_to_3d(pathpatch, z, normal):
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += delta
    return pathpatch

def plot_plane(ax, g,  point, normal, size=10, color='y'):
#    p = Circle((0, 0), size, facecolor = color, alpha = .2)
    p = Rectangle((0-g.pix_size/2,0-g.pix_size/2), width =g.pix_size , height =g.pix_size)
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathp = pathpatch_translate(p, (point[0], point[1], point[2]))
    return pathp


def transform(x, y, z):




def main():


    print('WARNING THIS NEEDS WORK, mostly hardcoded')
    parser = argparse.ArgumentParser(description='Get number of photons for LightEmission package for a requested p.e. per pixel')
    parser.add_argument('--runfile', help='Path to runfile that contains run number and requested/estimated p.e.')
    parser.add_argument('--outfile', help='file to write out data', default=None)
    parser.add_argument('--pdefile', help='Path to PDE file, used to obtain efficiency at given wavelength')
    parser.add_argument('--transmission', help='Path to additional transmission file, used to obtain efficiency at given wavelength', default=None)
    parser.add_argument('--wavelength', default=405, type=float, help='wavelength of lightsource used')
    parser.add_argument('--camera', default="CHEC", help='ctapipe camera geometery to use')
    parser.add_argument('--ls_distance', default=1, type=float, help='distance of lightsource to camera focal plane [m]')
    parser.add_argument('--angular_distribution', default="ang_dist_2.dat", help='file containing angular distribution')
    args = parser.parse_args()

    # To convert range of desired pe to Lightsource values
    try:
        fl = np.loadtxt(args.runfile, unpack=True)
        pe = fl[3]
        run = fl[0]
    except OSError:
        pe = [float(args.runfile)]
        run =[1]

    out=None
    if args.outfile != None:
        out = open(args.outfile, 'w')
        out.write('#run_number fw_pos fw_atten pe_expected ph_required n_events\n')
    ph = []
    pe2 =[]
    # run = []

    pde = np.loadtxt(args.pdefile,unpack=True)
    fpde = interp1d(pde[0], pde[1])
    if args.transmission != None:
        trans = np.loadtxt(args.transmission, unpack=True)
        ftrans = interp1d(trans[0], trans[1])

        tote_trans = fpde(args.wavelength) * ftrans(args.wavelength)
    else:
        tote_trans = fpde(args.wavelength)

    # print(tote_trans)
    # exit()

    g = Geometry()
    g.lightsource_distance = args.ls_distance
    g.ang_dist_file_string = args.angular_distribution
    g.camera = args.camera
    g.setup_geom()
    # pe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 28, 30, 32, 35, 37, 40, 43, 46, 49, 53, 57, 61, 65, 70, 75, 81, 86, 93, 100, 107, 114, 123, 132, 141, 151, 162, 174, 187, 200, 215, 231, 247, 265, 284, 305, 327, 351, 376, 403, 432, 464, 497, 533, 572, 613, 657, 705, 756, 811, 869, 932, 1000]
    # pe = [   1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   12, 14,   16,   19,   22,   25,   29,   33,   39,   44,   51,   59, 68,   79,   91,  104,  120,  138,  159,  184,  212,  244,  281, 323,  372,  429,  494,  568,  655,  754,  868, 1000]

    try:
        for n,i in enumerate(pe):
            # g.required_pe = i
            photons = g.set_illumination(i, tote_trans)
            # print(i, ' ', photons)
            ph.append(round(photons,2))
            pe2.append(round(i,2))
            # run.append(int(fl[0][n]))
            # g.getscale(plot=False)
            if args.outfile != None:
                out.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (int(fl[0][n]),int(fl[1][n]),fl[2][n],fl[3][n],round(photons,2),int(fl[4][n])))
    except TypeError:
        # g.required_pe = i
        photons = g.set_illumination(pe, tote_trans)
        # print(i, ' ', photons)
        ph.append(round(photons, 2))
        pe2.append(round(pe, 2))
        # run.append(int(fl[0][n]))
        # g.getscale(plot=False)
        if args.outfile != None:
            out.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
            int(fl[0][n]), int(fl[1][n]), fl[2][n], fl[3][n], round(photons, 2), int(fl[4][n])))

    print('number of requested photoelectrons')
    print(pe2)
    print('number of required emitted photons')
    print(ph)
    print('Run number')
    print(run)

    # g.getscale(plot=True)
    # plt.show()


    #
    fig0=plt.figure(0)
    ax0 = fig0.add_subplot(111, projection='3d')
    verts = []
    # print(g.geom.pix_x)
    # exit()
    for i in range(2048):
        x=g.geom.pix_x[i].value
        y=g.geom.pix_y[i].value
        if x >0.14 and y >0.1:

            z=focal_plane(np.sqrt((x) ** 2 + (y) ** 2))
            #
            dist = np.sqrt(x**2+y**2+(z+1)**2)
            # vector = [x/dist,y/dist,1/dist]
            # print(vector)
            # # exit()
            #
            # verts.append((x - g.pix_size / 2, y - g.pix_size / 2, z))
            # verts.append((x + g.pix_size / 2, y - g.pix_size / 2, z))
            # verts.append((x + g.pix_size / 2, y + g.pix_size / 2, z))
            # verts.append((x - g.pix_size / 2, y + g.pix_size / 2, z))
            #
            # plt.scatter(x,y,z, color='k', marker=',')
            # ax0.quiver(x, y, z , 0, 0, 1, length=0.005, color='C0')
            # ax0.quiver(x, y, z, x/dist,y/dist,1/dist, length=0.005, color='C1')


            point = np.array([x, y, z])
            normal = np.array([x/dist,y/dist,1/dist])

            # o = np.array([2, 2, 2])
            # v = np.array([3, 3, 3])
            # n = [1, 0, 1]

            from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # pathp = plot_plane(ax0, g, point, normal, size=3)
            # print(pathp.get_verts())
            # print(pathp.get_xy())
            # ax0.set_xlim(0.14, 0.16)
            # ax0.set_ylim(0.1, 0.12)
            # ax0.set_zlim(-0.02, 0)

            # a plane is a*x+b*y+c*z+d=0
            # [a,b,c] is the normal. Thus, we have to calculate
            # d and we're set
            d = -point.dot(normal)
            #
            # # create x,y
            xx, yy = np.meshgrid([-x- g.pix_size / 2 - PIXEL_EPSILON, x + g.pix_size / 2 + PIXEL_EPSILON],
                                 [y -g.pix_size / 2 - PIXEL_EPSILON, y + g.pix_size / 2 + PIXEL_EPSILON])
            #
            # # calculate corresponding z
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            #
            # # plot the surface
            # plt3d = plt.figure().gca(projection='3d')
            ax0.plot_surface(xx, yy, z)
            # plt.show()

    # print(verts)
    # poly3d = Poly3DCollection(verts)
    # poly3d.set_array(np.array(range(len(verts))))
    # ax0.add_collection3d(poly3d)
    # ax0=fig0.add_subplot(111)
    # fig1=plt.figure(1)
    # ax1 = fig1.add_subplot(111)

    plt.show()

if __name__ == '__main__':
    main()